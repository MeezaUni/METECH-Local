"""Score an engine on the synthetic validation set and the hand-written real-world set.

    uv run python ../ml/evaluate.py --engine rules
    uv run python ../ml/evaluate.py --engine model [--model-dir ../backend/models/kmrl-logreader] [--tag round2]
    uv run python ../ml/evaluate.py --engine base     # Qwen2.5-0.5B-Instruct before fine-tuning (few-shot prompt)
    uv run python ../ml/evaluate.py --engine ollama --sets realworld   # a general 10B model via Ollama, no fine-tuning

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
    def __init__(self, model_dir: Path | str, batch: int = 16, name: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.padding_side = "left"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(self.device).eval()
        self.batch = batch
        self.name = name or f"trained model ({Path(model_dir).name}, {self.device})"

    def messages(self, r: dict) -> list[dict]:
        return parse_messages(r["text"], date.fromisoformat(r["today"])) if r["task"] == "parse" else ask_messages(r["question"])

    def run(self, rows: list[dict]) -> list[object]:
        prompts = [self.tok.apply_chat_template(self.messages(r), tokenize=False, add_generation_prompt=True) for r in rows]
        order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))   # similar lengths batch better
        raw: dict[int, str] = {}
        for s in range(0, len(order), self.batch):
            idx = order[s:s + self.batch]
            enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
            longest = max(len(rows[i].get("text", "")) for i in idx)
            with self.torch.inference_mode():
                gen = self.model.generate(**enc, max_new_tokens=96 + 4 * longest, do_sample=False, repetition_penalty=1.0,
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


BASE_PARSE_SYSTEM = """You extract fleet updates for the Kochi Metro (KMRL) train depot from logbook text.
Reply with JSON only: {"updates": [ ... ]}. One object per fact, with these keys:
- train_name: "KMRL-TS-NNN" (three digits). "train 4", "TS-04", "rake 4" all mean "KMRL-TS-004". "" if not stated.
- action: one of renew_certificate, close_job_card, open_job_card, flag_cleaning, mark_cleaned, complete_maintenance, update_mileage
- department (renew_certificate): "Rolling-Stock", "Signalling" or "Telecom"
- date_text (renew_certificate): the new expiry exactly as written in the text, e.g. "for 6 months" or "10/10/2027"
- job_card_id (close_job_card): the job card / work order number
- description and priority (open_job_card): short defect text; priority "Critical", "High" or "Routine"
- km (update_mileage): kilometres since the last inspection
- source_text: the exact words from the text that support the update
Return {"updates": []} if the text contains no fleet update."""

BASE_PARSE_EXAMPLES = [
    ("Today: 2026-03-02\nLogbook:\nTelecom cert for TS-15 renewed till 5 March 2027. Closed JC 1071.",
     '{"updates":[{"train_name":"KMRL-TS-015","action":"renew_certificate","department":"Telecom","date_text":"5 March 2027",'
     '"source_text":"Telecom cert for TS-15 renewed till 5 March 2027"},{"train_name":"","action":"close_job_card","job_card_id":1071,'
     '"source_text":"Closed JC 1071"}]}'),
    ("Today: 2026-03-02\nLogbook:\nTrain 7 A-check completed. Rake 22 door 3 not closing.",
     '{"updates":[{"train_name":"KMRL-TS-007","action":"complete_maintenance","source_text":"Train 7 A-check completed"},'
     '{"train_name":"KMRL-TS-022","action":"open_job_card","description":"Door 3 not closing","priority":"High",'
     '"source_text":"Rake 22 door 3 not closing"}]}'),
    ("Today: 2026-03-02\nLogbook:\nGood night all, meeting at 10.", '{"updates":[]}'),
]

BASE_ASK_SYSTEM = """You convert a train depot supervisor's question into a JSON query. Reply with JSON only.
Keys: intent (one of count, list, train_detail, explain_assignment, expiring_certificates, open_jobs,
branding_status, plan_summary, highest_wear, help), and where relevant status (Service, Standby,
Inspection Bay, Cleaning, Held, Fit, Warning, Unfit; green=Fit, yellow=Warning, red=Unfit),
train_name ("KMRL-TS-NNN"), days, priority (Critical, High, Routine), limit."""

BASE_ASK_EXAMPLES = [
    ("How many trains are on standby?", '{"intent":"count","status":"Standby"}'),
    ("why is TS-9 in the inspection bay", '{"intent":"explain_assignment","train_name":"KMRL-TS-009"}'),
    ("certs expiring in 10 days", '{"intent":"expiring_certificates","days":10}'),
]


class BaseModelEngine(ModelEngine):
    """The base model *before* fine-tuning, given a detailed instruction prompt and worked examples (few-shot).

    The fair baseline for "how much did fine-tuning help?": same model, same decoding, same scoring,
    but it has to learn the task from the prompt instead of from training data.
    """

    def __init__(self, base: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        super().__init__(base, batch=8, name=f"base model, few-shot ({base.split('/')[-1]}, no fine-tuning)")

    def messages(self, r: dict) -> list[dict]:
        if r["task"] == "parse":
            system, examples = BASE_PARSE_SYSTEM, BASE_PARSE_EXAMPLES
            user = f"Today: {r['today']}\nLogbook:\n{r['text']}"
        else:
            system, examples, user = BASE_ASK_SYSTEM, BASE_ASK_EXAMPLES, r["question"]
        msgs = [{"role": "system", "content": system}]
        for q, a in examples:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        return msgs + [{"role": "user", "content": user}]


class OllamaEngine:
    """A general-purpose local LLM served by Ollama (no fine-tuning), with the long instruction prompt."""

    def __init__(self, model: str = "qwen3.5:latest"):
        from app.ai.base import AIContext, AIUnavailable
        from app.ai.ollama import OllamaProvider

        self.provider, self.ctx_cls, self.unavailable = OllamaProvider("http://127.0.0.1:11434", model), AIContext, AIUnavailable
        self.name = f"general LLM via Ollama ({model}, no fine-tuning)"

    def run(self, rows: list[dict]) -> list[object]:
        out = []
        for r in rows:
            try:
                if r["task"] == "parse":
                    today = date.fromisoformat(r["today"])
                    ups = self.provider.parse_logbook(r["text"], self.ctx_cls(today, [], [], {}))
                    proposed = [u.model_dump(exclude_none=True) for u in ups]
                    out.append(ground_and_resolve(proposed, r["text"], today)[0])
                else:
                    out.append(self.provider.interpret_question(r["question"], None).model_dump(exclude_none=True))
            except self.unavailable:
                out.append(None)
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
           "tp": 0, "fp": 0, "fn": 0, "ask_n": 0, "ask_exact": 0, "failures": [],
           "per_action": {}, "per_intent": {}}
    for r, p in zip(rows, preds):
        if r["task"] == "parse":
            res["parse_n"] += 1
            exp_r = {fact_key(u, False) for u in r["expected"]}
            got_r = {fact_key(u, False) for u in p} if p is not None else set()
            for u in r["expected"]:   # recall per action type: was this expected fact extracted?
                a = res["per_action"].setdefault(u["action"], {"expected": 0, "found": 0})
                a["expected"] += 1
                a["found"] += fact_key(u, False) in got_r
            if p is None:
                res["fn"] += len(r["expected"])
                res["failures"].append({"text": r["text"], "got": "INVALID JSON"})
                continue
            res["parse_valid"] += 1
            exp_s, got_s = {fact_key(u, True) for u in r["expected"]}, {fact_key(u, True) for u in p}
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
            it = res["per_intent"].setdefault(r["expected"]["intent"], {"expected": 0, "found": 0})
            it["expected"] += 1
            it["found"] += ok
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
    ap.add_argument("--engine", choices=["rules", "model", "base", "ollama"], required=True)
    ap.add_argument("--model-dir", type=Path, default=ROOT.parent / "backend" / "models" / "kmrl-logreader")
    ap.add_argument("--ollama-model", default="qwen3.5:latest")
    ap.add_argument("--sets", nargs="+", choices=["synthetic", "realworld"], default=["synthetic", "realworld"])
    ap.add_argument("--tag", help="report name suffix, e.g. round1 -> reports/eval_model_round1.json")
    args = ap.parse_args()

    engine = {
        "rules": RulesEngine,
        "model": lambda: ModelEngine(args.model_dir),
        "base": BaseModelEngine,
        "ollama": lambda: OllamaEngine(args.ollama_model),
    }[args.engine]()
    report = {"engine": engine.name}
    for name, rows in load_sets().items():
        if name not in args.sets:
            continue
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
    path = out / f"eval_{args.engine}{'_' + args.tag if args.tag else ''}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull report with failures: {path}")


if __name__ == "__main__":
    main()
