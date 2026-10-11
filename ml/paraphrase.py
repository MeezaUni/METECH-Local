"""Teacher-paraphrased training data from a local Ollama model (step 2 of the data recipe).

The template generator gets the *labels* right by construction but its *wording* is
limited to the templates we wrote. A larger local model rewrites each generated message
the way different depot staff would actually type it. The labels are NOT taken from the
teacher: they stay the generator's ground truth, and any rewrite that drops or changes a
protected string (train, job card, date, km, department, urgency word) or invents a new
number is rejected. So the data gets more natural without getting less correct.

    cd backend
    uv run python ../ml/paraphrase.py                      # ~45 min with qwen3.5 on an RTX 4060
    uv run python ../ml/paraphrase.py --messages 20 --questions 10   # quick look first
    uv run python ../ml/generate_dataset.py                # merges data/paraphrased.jsonl into train/val

Runs fully offline against Ollama (http://127.0.0.1:11434). Safe to stop and re-run:
finished items are kept and skipped.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))
sys.path.insert(0, str(ROOT))
import generate_dataset as gen  # noqa: E402
from app.ai.local_format import ask_messages, dump_updates, parse_messages  # noqa: E402

OUT = ROOT / "data" / "paraphrased.jsonl"

SYSTEM = (
    "You rewrite short messages written by staff at a Kochi Metro (KMRL) train depot. "
    "Write it the way different real people would type it: WhatsApp style, terse shift-handover notes, "
    "Indian English, sometimes a Malayalam word (kazhinju = finished, cheythu = done, sheri = ok), "
    "abbreviations, missing punctuation. Keep the meaning exactly the same. "
    "Copy every protected string character for character. Never add trains, numbers, dates or new facts. "
    "Each rewrite is a single line."
)
SCHEMA = {
    "type": "object",
    "properties": {"variants": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3}},
    "required": ["variants"],
}
DIGITS = re.compile(r"\d+")


def ask_teacher(host: str, model: str, text: str, protect: list[str]) -> list[str]:
    prompt = (f"Message: {text}\nProtected strings (copy exactly): {json.dumps(protect, ensure_ascii=False)}\n"
              "Give 3 different rewrites.")
    body = {"model": model, "system": SYSTEM, "prompt": prompt, "format": SCHEMA, "stream": False, "think": False,
            "options": {"temperature": 0.9, "num_predict": 320}}
    req = urllib.request.Request(f"{host}/api/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        reply = json.loads(r.read())["response"]
    try:
        return [v.strip() for v in json.loads(reply).get("variants", []) if isinstance(v, str)]
    except json.JSONDecodeError:
        return []


def acceptable(variant: str, original: str, protect: list[str]) -> bool:
    v = variant.lower()
    return (
        0 < len(variant) <= max(220, 3 * len(original))
        and "\n" not in variant and "{" not in variant and "updates" not in v
        and all(p.lower() in v for p in protect if p)
        and set(DIGITS.findall(variant)) <= set(DIGITS.findall(original))   # no invented numbers
        and v.strip(" .") != original.lower().strip(" .")
    )


def rewrite(host: str, model: str, text: str, protect: list[str], stats: dict) -> list[str]:
    try:
        variants = ask_teacher(host, model, text, protect)
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        stats["errors"] += 1
        print(f"  ! teacher error: {exc}", file=sys.stderr)
        return []
    ok = [v for v in variants if acceptable(v, text, protect)]
    stats["proposed"] += len(variants)
    stats["accepted"] += len(ok)
    return list(dict.fromkeys(ok))   # de-duplicate, keep order


def question_protect(q: str, target: dict) -> list[str]:
    protect = [m.group(0) for m in gen_train_re.finditer(q)] + DIGITS.findall(q)
    if target.get("intent") in ("count", "list"):
        protect += [w for w in gen.STATUS_WORDS.get(target.get("status"), []) if w.lower() in q.lower()][:1]
    if target.get("priority"):
        protect.append(target["priority"].lower())
    return protect


gen_train_re = re.compile(r"\b(?:KMRL[-\s]?)?(?:TS|trainset|train|rake|set|unit|T)[-\s#.]*\d{1,3}\b|\bTS\d{2,3}\b", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:latest")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--messages", type=int, default=400, help="base logbook messages to rewrite (3 rewrites each)")
    ap.add_argument("--questions", type=int, default=200, help="base questions to rewrite (3 rewrites each)")
    ap.add_argument("--seed", type=int, default=314)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {json.loads(l)["base_id"] for l in open(args.out, encoding="utf-8")}
    stats = {"proposed": 0, "accepted": 0, "errors": 0, "rows": 0}
    started = time.perf_counter()
    rng = random.Random(args.seed)
    jobs = [("m", i) for i in range(args.messages)] + [("q", i) for i in range(args.questions)]

    with open(args.out, "a", encoding="utf-8") as out:
        for n, (kind, i) in enumerate(jobs, 1):
            base_id = f"{kind}{i}"
            item_rng = random.Random(f"{args.seed}-{base_id}")   # same base item on every run, so resuming is exact
            if base_id in done:
                continue
            if kind == "m":
                today = date(2025, 6, 1) + timedelta(days=item_rng.randint(0, 900))
                k = item_rng.choices([1, 2, 3], [50, 35, 15])[0]
                facts = []
                for t in item_rng.sample(range(1, gen.FLEET_MAX + 1), k):
                    fn = item_rng.choices([f for f, _ in gen.FACTS], [w for _, w in gen.FACTS])[0]
                    facts.append(fn(t, today, item_rng))
                options = [rewrite(args.host, args.model, clause, protect, stats) for clause, _, protect in facts]
                for v in range(3):   # up to 3 rewritten messages per base message
                    if not all(len(o) > v for o in options):
                        break
                    clauses, updates = [], []
                    for (clause, ups, _), opts in zip(facts, options):
                        clauses.append(opts[v])
                        updates += [{**u, "source_text": opts[v]} for u in ups]
                    text = rng.choice(gen.NOISE_PREFIX) + rng.choice([". ", ".\n", "\n", "; "]).join(clauses) + rng.choice(gen.NOISE_SUFFIX)
                    out.write(json.dumps({"base_id": base_id, "task": "parse", "messages": parse_messages(text, today),
                                          "target": dump_updates(updates), "source": f"teacher:{args.model}"}, ensure_ascii=False) + "\n")
                    stats["rows"] += 1
            else:
                ex = gen.ask_example(item_rng)
                q = ex["messages"][1]["content"]
                target = json.loads(ex["target"])
                for variant in rewrite(args.host, args.model, q, question_protect(q, target), stats):
                    out.write(json.dumps({"base_id": base_id, "task": "ask", "messages": ask_messages(variant),
                                          "target": ex["target"], "source": f"teacher:{args.model}"}, ensure_ascii=False) + "\n")
                    stats["rows"] += 1
            out.flush()
            if n % 20 == 0 or n == len(jobs):
                rate = stats["accepted"] / max(1, stats["proposed"])
                print(f"{n}/{len(jobs)} items  {stats['rows']} new rows  {rate:.0%} of rewrites accepted  "
                      f"{stats['errors']} errors  {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    print(f"done: {stats['rows']} new rows in {args.out} ({stats['accepted']}/{stats['proposed']} rewrites accepted)")


if __name__ == "__main__":
    main()
