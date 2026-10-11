"""Terminal demo of the on-device model (no web app needed).

    cd backend
    uv run python ../ml/demo.py                                   # interactive
    uv run python ../ml/demo.py "TS-07 A-check done. Closed JC 1071."
    uv run python ../ml/demo.py --ask "why is TS-09 on standby?"

Shows the raw model output, then what the backend makes of it: expiry phrases resolved to dates,
ungrounded suggestions dropped, the depot priority policy applied, and job cards matched to trains.
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

from app.ai.local_format import ask_messages, ground_and_resolve, load_json, parse_messages  # noqa: E402
from app.models import FleetUpdate  # noqa: E402
from app.seed import generate_fleet  # noqa: E402
from app.updates import validate_updates  # noqa: E402

SAMPLES = [
    "Handover 22:40 - signalling cert for TS 09 extended for 6 months. JC 1071 brake cylinder sorted on TS-19, closed. "
    "TS-12 needs a deep clean, graffiti on car 2. TS-20 CCTV not recording in car 3.",
    "B check of TS-24 over, TS-24 good to go. Rolling stock FC for rake 14 renewed for a fortnight.",
    "Good night all, meeting at 10 tomorrow.",
]


class LogReader:
    def __init__(self, model_dir: Path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not (model_dir / "config.json").exists():
            sys.exit(f"No model at {model_dir}. Run run.bat once (downloads it) or train.bat.")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        t = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(self.device).eval()
        params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"kmrl-logreader loaded on {self.device.upper()} in {time.perf_counter() - t:.1f}s ({params:.0f}M parameters)\n")

    def generate(self, messages: list[dict], max_new_tokens: int) -> tuple[str, float, int]:
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        t = time.perf_counter()
        with self.torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        new = out[0, enc["input_ids"].shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True), time.perf_counter() - t, len(new)


def show_parse(bot: LogReader, text: str, today: date) -> None:
    raw, secs, n = bot.generate(parse_messages(text, today), max_new_tokens=min(1024, 160 + 4 * len(text)))
    print(f"Raw model output ({n} tokens, {secs:.1f}s):\n  {raw}\n")
    try:
        proposed = [{"train_name": "", **u} for u in load_json(raw).get("updates", [])]
    except ValueError:
        print("  The model did not return valid JSON; the app would fall back to the rule engine.\n")
        return
    kept, dropped = ground_and_resolve(proposed, text, today)
    fleet = generate_fleet(today)
    valid, warnings = validate_updates([FleetUpdate.model_validate(u) for u in kept], fleet, today)
    print("After grounding, date resolution and the depot priority policy:")
    if not valid:
        print("  (no fleet updates)")
    for u in valid:
        fields = {k: v for k, v in u.model_dump().items() if v is not None and k not in ("train_name", "action", "source_text")}
        print(f"  + {u.train_name}  {u.action:<20} {json.dumps(fields)}")
    for w in dropped + warnings:
        print(f"  ! {w}")
    print()


def show_ask(bot: LogReader, question: str) -> None:
    raw, secs, n = bot.generate(ask_messages(question), max_new_tokens=64)
    print(f"Structured query ({n} tokens, {secs:.2f}s): {raw}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="logbook text to read")
    ap.add_argument("--ask", help="a supervisor question instead")
    ap.add_argument("--model-dir", type=Path, default=ROOT.parent / "backend" / "models" / "kmrl-logreader")
    ap.add_argument("--today", default=date.today().isoformat())
    args = ap.parse_args()
    today = date.fromisoformat(args.today)
    bot = LogReader(args.model_dir)

    if args.ask:
        show_ask(bot, args.ask)
    elif args.text:
        show_parse(bot, args.text, today)
    else:
        print("Type a logbook message (or '?question' to ask, 's' for samples, 'q' to quit).")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if line in ("q", "quit", "exit"):
                break
            if line == "s":
                for s in SAMPLES:
                    print(f"\nLogbook: {s}")
                    show_parse(bot, s, today)
            elif line.startswith("?"):
                show_ask(bot, line[1:])
            elif line:
                show_parse(bot, line, today)


if __name__ == "__main__":
    main()
