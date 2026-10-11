"""Fine-tune the KMRL depot copilot with LoRA and export a merged, offline-ready model.

    cd backend
    uv sync --group train
    uv run python ../ml/generate_dataset.py
    uv run python ../ml/train.py              # ~16 min per epoch on an RTX 4060 laptop GPU (8 GB)
    uv run python ../ml/evaluate.py --engine model

If a run is interrupted, the adapter saved after the last finished epoch can still be exported:

    uv run python ../ml/train.py --merge-from ../backend/models/kmrl-copilot-checkpoints/epoch-1

Base model: Qwen2.5-0.5B-Instruct (Apache-2.0). It is downloaded once from the
Hugging Face Hub; after training, everything runs offline from
backend/models/kmrl-copilot.

Only the assistant's JSON answer is trained on (prompt tokens are masked), so
the model learns the extraction task without memorising the prompt.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODELS = ROOT.parent / "backend" / "models"


def encode(tok, row: dict, max_len: int) -> dict | None:
    prompt = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(row["messages"] + [{"role": "assistant", "content": row["target"]}], tokenize=False)
    assert full.startswith(prompt), "chat template must render the prompt as a prefix"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"]
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "labels": [-100] * len(p_ids) + ids[len(p_ids):]}


def batches(examples: list[dict], size: int, rng: random.Random, shuffle: bool = True):
    """Group examples of similar length to minimise padding."""
    idx = list(range(len(examples)))
    if shuffle:
        rng.shuffle(idx)
    chunks = [sorted(idx[i:i + size * 50], key=lambda j: len(examples[j]["input_ids"])) for i in range(0, len(idx), size * 50)]
    groups = [c[i:i + size] for c in chunks for i in range(0, len(c), size)]
    if shuffle:
        rng.shuffle(groups)
    return groups


def collate(examples: list[dict], pad_id: int, device: str) -> dict:
    n = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), n), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), n), -100, dtype=torch.long)
    mask = torch.zeros((len(examples), n), dtype=torch.long)
    for i, e in enumerate(examples):
        k = len(e["input_ids"])
        ids[i, :k] = torch.tensor(e["input_ids"])
        labels[i, :k] = torch.tensor(e["labels"])
        mask[i, :k] = 1
    return {"input_ids": ids.to(device), "labels": labels.to(device), "attention_mask": mask.to(device)}


def answer_loss(model, batch: dict) -> torch.Tensor:
    """Cross-entropy on the answer tokens only.

    Equivalent to the model's own loss, but the vocabulary projection (151k logits per token, in fp32)
    runs only where labels exist instead of over the whole prompt. That tensor is what overflows 8 GB.
    """
    base = model.get_base_model()                      # LoRA layers are injected in place, so they still apply
    hidden = base.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    labels = batch["labels"][:, 1:]
    keep = labels != -100
    logits = base.lm_head(hidden[:, :-1][keep]).float()
    return torch.nn.functional.cross_entropy(logits, labels[keep])


@torch.no_grad()
def eval_loss(model, examples: list[dict], pad_id: int, device: str, size: int) -> float | None:
    """Validation loss, or None if the GPU runs out of memory. It is a progress signal and must never end a run."""
    model.eval()
    if device == "cuda":
        torch.cuda.empty_cache()   # hand fragmented cached blocks back before allocating different shapes
    total, count = 0.0, 0
    try:
        for group in batches(examples, size, random.Random(0), shuffle=False):
            batch = collate([examples[i] for i in group], pad_id, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                total += answer_loss(model, batch).item() * len(group)
            count += len(group)
        return total / max(1, count)
    except torch.OutOfMemoryError:
        print("WARNING: out of GPU memory while computing validation loss; skipped (training is unaffected)", flush=True)
        return None
    finally:
        if device == "cuda":
            torch.cuda.empty_cache()
        model.train()


def export(model, tok, out: Path, meta: dict) -> None:
    """Merge the LoRA adapter into the base weights and save one plain, offline-loadable model."""
    merged = model.merge_and_unload().to(torch.bfloat16)
    # The base model ships chat-sampling defaults (temperature, top_p, repetition_penalty 1.1). The copilot
    # emits JSON, which repeats keys by design, so save plain deterministic decoding as the default.
    gc = merged.generation_config
    gc.do_sample, gc.repetition_penalty, gc.temperature, gc.top_p, gc.top_k = False, 1.0, None, None, None
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "kmrl_model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved to {out}")


def merge_only(args) -> None:
    """Export a model from an adapter checkpoint (e.g. after an interrupted run)."""
    ckpt = args.merge_from
    info = json.loads((ckpt / "kmrl_checkpoint.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(ckpt)
    base = AutoModelForCausalLM.from_pretrained(info["base_model"], dtype=torch.float32)
    model = PeftModel.from_pretrained(base, ckpt)
    export(model, tok, args.out, {**info, "name": "kmrl-copilot", "exported_from": str(ckpt)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--epochs", type=float, default=1, help="loss flattens within one pass over the synthetic data")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--no-grad-checkpointing", action="store_true", help="faster, but needs more than 8 GB of VRAM")
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--out", type=Path, default=MODELS / "kmrl-copilot")
    ap.add_argument("--checkpoints", type=Path, default=MODELS / "kmrl-copilot-checkpoints")
    ap.add_argument("--merge-from", type=Path, help="skip training; export a model from this adapter checkpoint")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.merge_from:
        merge_only(args)
        return

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA GPU found, training on CPU will be very slow.", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = {s: [json.loads(l) for l in open(ROOT / "data" / f"{s}.jsonl", encoding="utf-8")] for s in ("train", "val")}
    data = {s: [e for e in (encode(tok, r, args.max_len) for r in rs) if e] for s, rs in rows.items()}
    print(f"train {len(data['train'])}  val {len(data['val'])} examples "
          f"(dropped {len(rows['train']) - len(data['train'])} over {args.max_len} tokens)")

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
    if not args.no_grad_checkpointing:
        # Recompute activations in the backward pass instead of storing them all: fits an 8 GB card.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    for p in model.parameters():          # keep the trainable adapter in fp32 for stable updates
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(data["train"]) / (args.batch * args.accum))
    total_steps = int(steps_per_epoch * args.epochs)
    warmup = max(1, int(0.03 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    run_info = {
        "base_model": args.base, "train_examples": len(data["train"]), "epochs": args.epochs, "lr": args.lr,
        "lora_r": args.lora_r, "effective_batch": args.batch * args.accum,
    }

    history: list[dict] = []
    started = time.perf_counter()
    step, micro, running = 0, 0, 0.0
    model.train()
    print(f"training {total_steps} optimiser steps on {device}", flush=True)
    epoch = 0
    while step < total_steps:
        for group in batches(data["train"], args.batch, rng):
            batch = collate([data["train"][i] for i in group], tok.pad_token_id, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = answer_loss(model, batch) / args.accum
            loss.backward()
            running += loss.item()
            micro += 1
            if micro % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0 or step == 1:
                elapsed = time.perf_counter() - started
                peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0
                print(f"step {step:>4}/{total_steps}  loss {running / (25 if step > 1 else 1):.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  peak {peak:.1f} GB  {elapsed / 60:.1f} min", flush=True)
                history.append({"step": step, "loss": running / (25 if step > 1 else 1)})
                running = 0.0
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
            if step >= total_steps:
                break
        epoch += 1
        # Save the adapter first, so nothing that follows can lose a finished epoch.
        ckpt = args.checkpoints / f"epoch-{epoch}"
        model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
        (ckpt / "kmrl_checkpoint.json").write_text(json.dumps({**run_info, "epoch": epoch, "step": step}, indent=2), encoding="utf-8")
        vl = eval_loss(model, data["val"], tok.pad_token_id, device, args.batch)
        print(f"epoch {epoch} done, adapter saved to {ckpt}, validation loss "
              f"{'skipped' if vl is None else f'{vl:.4f}'}", flush=True)
        history.append({"step": step, "val_loss": vl})

    minutes = (time.perf_counter() - started) / 60
    print(f"training finished in {minutes:.1f} min, merging adapter into the base weights", flush=True)
    export(model, tok, args.out, {
        "name": "kmrl-copilot", **run_info, "trained_at": datetime.now().isoformat(timespec="seconds"),
        "minutes": round(minutes, 1),
        "final_val_loss": next((h["val_loss"] for h in reversed(history) if h.get("val_loss") is not None), None),
        "history": history,
    })


if __name__ == "__main__":
    main()
