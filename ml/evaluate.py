"""Score an engine on the synthetic validation set and the hand-written real-world set.

    uv run python ../ml/evaluate.py --engine rules
    uv run python ../ml/evaluate.py --engine model [--model-dir ../backend/models/kmrl-copilot]

Parse metrics compare the *facts* extracted (train, action, department, date,
job card, km, priority); free-text fields (description, source_text) are not
scored. "Relaxed" ignores priority, which is partly a judgement call.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))
from app.ai import rules  # noqa: E402
from app.ai.local_format import ask_messages, ground_and_resolve, load_json, parse_messages  # noqa: E402
from app.priority import apply_policy  # noqa: E402

DEFAULTS = {"expiring_certificates": {"days": 7}, "highest_wear": {"limit": 5}}


# ------------------------------------------------------------------ loading
def load_sets() -> dict[str, list[dict]]:
    sets: dict[str, list[dict]] = {"synthetic": [], "realworld": []}
    for line in open(ROOT / "data" / "val.jsonl", encoding="utf-8"):
        r = json.loads(line)
        content = r["messages"][1]["content"]
        target = json.loads(r["target"])
        if r["task"] == "parse":
            head, text = content.split("\nLogbook:\n", 1)
            today = head.removeprefix("Today: ")
            expected, _ = ground_and_resolve(target["updates"], text, date.fromisoformat(today))   # date_text -> date
            sets["synthetic"].append({"task": "parse", "today": today, "text": text, "expected": expected})
        else:
            sets["synthetic"].append({"task": "ask", "question": content, "expected": target})
    for line in open(ROOT / "data" / "realworld_test.jsonl", encoding="utf-8"):
        sets["realworld"].append(json.loads(line))
    return sets


# ------------------------------------------------------------------ engines
class RulesEngine:
    name = "rule engine"

    def run(self, rows: list[dict]) -> list[object]:
        out = []
        for r in rows:
            if r["task"] == "parse":
                ups = rules.parse_logbook(r["text"], date.fromisoformat(r["today"]))
                out.append([u.model_dump(exclude_none=True) for u in ups])
            else:
                out.append(rules.interpret_question(r["question"]).model_dump(exclude_none=True))
        return out


class ModelEngine:
    def __init__(self, model_dir: Path, batch: int = 16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.padding_side = "left"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(self.device).eval()
        self.batch = batch
        self.name = f"trained model ({model_dir.name}, {self.device})"

    def run(self, rows: list[dict]) -> list[object]:
        prompts = []
        for r in rows:
            msgs = parse_messages(r["text"], date.fromisoformat(r["today"])) if r["task"] == "parse" else ask_messages(r["question"])
            prompts.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))   # similar lengths batch better
        raw: dict[int, str] = {}
        for s in range(0, len(order), self.batch):
            idx = order[s:s + self.batch]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
            longest = max(len(rows[i].get("text", "")) for i in idx)
            with self.torch.inference_mode():
                gen = self.model.generate(**enc, max_new_tokens=96 + 4 * longest, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
            texts = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            raw.update(zip(idx, texts))
        out = []
        for i, r in enumerate(rows):
            try:
                data = load_json(raw[i])
                if r["task"] == "parse":
                    # Exactly what the backend does: resolve expiry phrases, drop ungrounded suggestions.
                    proposed = [{"train_name": "", **u} for u in data.get("updates", [])]
                    out.append(ground_and_resolve(proposed, r["text"], date.fromisoformat(r["today"]))[0])
                else:
                    out.append(data)
            except ValueError:
                out.append(None)  # invalid JSON
        return out


# ------------------------------------------------------------------ scoring
def with_policy(rows: list[dict], preds: list[object]) -> list[object]:
    """Apply the depot priority policy exactly as the backend does before showing updates."""
    out = []
    for r, p in zip(rows, preds):
        if r["task"] != "parse" or p is None:
            out.append(p)
            continue
        out.append([
            {**u, "priority": apply_policy(u.get("description") or "", u.get("source_text"), u.get("priority"))[0]}
            if u.get("action") == "open_job_card" else u
            for u in p
        ])
    return out


def fact_key(u: dict, strict: bool) -> tuple:
    if u.get("action") == "close_job_card":
        return ("close_job_card", u.get("job_card_id"))   # the job card id decides the train
    key = (u.get("train_name"), u.get("action"), u.get("department"), u.get("date"), u.get("km"))
    return key + ((u.get("priority"),) if strict else ())


def norm_intent(d: dict) -> dict:
    d = {k: v for k, v in (d or {}).items() if v is not None}
    return {**DEFAULTS.get(d.get("intent"), {}), **d}


def score(rows: list[dict], preds: list[object]) -> dict:
    res = {"parse_n": 0, "parse_valid": 0, "parse_exact": 0, "parse_exact_relaxed": 0,
           "tp": 0, "fp": 0, "fn": 0, "ask_n": 0, "ask_exact": 0, "failures": []}
    for r, p in zip(rows, preds):
        if r["task"] == "parse":
            res["parse_n"] += 1
            if p is None:
                res["fn"] += len(r["expected"])
                res["failures"].append({"text": r["text"], "got": "INVALID JSON"})
                continue
            res["parse_valid"] += 1
            exp_s, got_s = {fact_key(u, True) for u in r["expected"]}, {fact_key(u, True) for u in p}
            exp_r, got_r = {fact_key(u, False) for u in r["expected"]}, {fact_key(u, False) for u in p}
            res["parse_exact"] += exp_s == got_s
            res["parse_exact_relaxed"] += exp_r == got_r
            res["tp"] += len(exp_r & got_r)
            res["fp"] += len(got_r - exp_r)
            res["fn"] += len(exp_r - got_r)
            if exp_s != got_s:
                res["failures"].append({"text": r["text"], "expected": sorted(map(str, exp_s)), "got": sorted(map(str, got_s))})
        else:
            res["ask_n"] += 1
            ok = p is not None and norm_intent(p) == norm_intent(r["expected"])
            res["ask_exact"] += ok
            if not ok:
                res["failures"].append({"question": r["question"], "expected": r["expected"], "got": p})
    prec = res["tp"] / max(1, res["tp"] + res["fp"])
    rec = res["tp"] / max(1, res["tp"] + res["fn"])
    res.update(
        parse_valid_rate=res["parse_valid"] / max(1, res["parse_n"]),
        parse_exact_rate=res["parse_exact"] / max(1, res["parse_n"]),
        parse_exact_relaxed_rate=res["parse_exact_relaxed"] / max(1, res["parse_n"]),
        fact_precision=prec, fact_recall=rec, fact_f1=2 * prec * rec / max(1e-9, prec + rec),
        ask_exact_rate=res["ask_exact"] / max(1, res["ask_n"]),
    )
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["rules", "model"], required=True)
    ap.add_argument("--model-dir", type=Path, default=ROOT.parent / "backend" / "models" / "kmrl-copilot")
    args = ap.parse_args()

    engine = RulesEngine() if args.engine == "rules" else ModelEngine(args.model_dir)
    report = {"engine": engine.name}
    for name, rows in load_sets().items():
        t0 = time.perf_counter()
        preds = engine.run(rows)
        raw = score(rows, preds)
        s = score(rows, with_policy(rows, preds))   # what the supervisor actually sees
        s["raw_parse_exact_rate"] = raw["parse_exact_rate"]
        s["seconds"] = round(time.perf_counter() - t0, 1)
        report[name] = s
        print(f"\n== {engine.name} on {name} set ({s['parse_n']} logbook, {s['ask_n']} questions, {s['seconds']}s)")
        print(f"  logbook  valid JSON {s['parse_valid_rate']:.1%}   exact {s['parse_exact_rate']:.1%} "
              f"(raw engine {raw['parse_exact_rate']:.1%})   exact ignoring priority {s['parse_exact_relaxed_rate']:.1%}")
        print(f"           fact precision {s['fact_precision']:.1%}  recall {s['fact_recall']:.1%}  F1 {s['fact_f1']:.1%}")
        print(f"  questions exact {s['ask_exact_rate']:.1%}")
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    path = out / f"eval_{args.engine}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull report with failures: {path}")


if __name__ == "__main__":
    main()
