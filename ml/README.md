# Training the on-device copilot

The local edition replaces Gemini with a small language model fine-tuned for exactly two jobs:

1. **Logbook reading**: free text in, fleet updates out, as JSON
   (`renew_certificate`, `close_job_card`, `open_job_card`, `flag_cleaning`, `mark_cleaned`, `complete_maintenance`, `update_mileage`).
2. **Question understanding**: a supervisor's question in, a structured query out
   (`count`, `list`, `train_detail`, `explain_assignment`, ...). The backend computes the answer from live data.

Run everything with `train.bat` in the project root, or step by step from `backend/`:

```bat
uv sync --group train
uv run python ..\ml\paraphrase.py              :: optional, ~20-30 min: teacher rewrites via local Ollama (qwen3.5)
uv run python ..\ml\generate_dataset.py        :: 8,000 template examples (+ teacher rewrites if present)
uv run python ..\ml\train.py                   :: LoRA fine-tune (~16 min), merge, save to backend\models\kmrl-copilot
uv run python ..\ml\evaluate.py --engine rules :: baseline
uv run python ..\ml\evaluate.py --engine model :: trained model
```

A run that stops after an epoch can still be exported:
`uv run python ..\ml\train.py --merge-from ..\backend\models\kmrl-copilot-checkpoints\epoch-1`

## Why this approach

**Base model: Qwen2.5-0.5B-Instruct** (Apache-2.0, 0.5B parameters). Large enough to read messy text,
small enough to fine-tune in minutes on an 8 GB laptop GPU and to run on a CPU if there is no GPU (~1 GB on disk).

**Fine-tuning: LoRA** on every attention and MLP projection (rank 32), then merged into the base weights,
so the backend loads one ordinary model with plain `transformers`: no adapters and no internet at runtime.
Only the JSON answer tokens are trained on; the prompt is masked out, and the vocabulary projection runs only
on answer tokens, which keeps peak VRAM around 3 GB.

**The model reads; code calculates.** Round 1 showed the classic small-model weakness: it read
"renewed for 6 months" correctly but did the calendar arithmetic wrong, and once it invented a certificate
renewal that wasn't in the message. So certificate updates carry `date_text`, the expiry *exactly as written*
("for 6 months", "10/10/2027"), and the backend turns it into a date with tested code (`rules.find_date`,
checked against 20,000 generated phrasings). Every suggestion is also grounded before anyone sees it: its train
number, job card number, km figure and expiry phrase must actually occur in the message, otherwise it is dropped
and the supervisor is told why (`local_format.ground_and_resolve`).

**Data, step 1: synthesised, labels correct by construction.** There is no public corpus of KMRL logbooks, so
`generate_dataset.py` starts from the *answer* (a list of updates or a query) and renders it into the text
depot staff actually write:

- many ways to name a train (`TS-04`, `train 4`, `rake 04`, `KMRL-TS-004`, `T04` ...), departments (`RS`, `S&T`, `comms` ...) and dates
  (`10/10/2027`, `5th Mar 2027`, `Jan 5`, `for 6 months`, `for a fortnight` ...), resolved against the message's date
- 60 defect descriptions; their priority is labelled by the depot priority policy (`backend/app/priority.py`),
  the same code the backend applies to every engine's output, so training data and the app can never disagree
- 1 to 4 facts per message, several job cards closed in one sentence, shift-handover prefixes, sign-offs,
  timestamps, typos, lower-casing, Malayalam-English fillers (`kazhinju`, `cheythu`, `ok aanu`)
- 8% pure chatter that must produce an empty list, and chatter mixed into real updates

Train numbers go up to 40 so the model keeps working as the fleet grows beyond today's 25.

**Data, step 2 (optional, recommended): teacher rewrites.** Templates get the labels right but limit the wording.
`paraphrase.py` asks a larger local model through Ollama (default `qwen3.5`, ~10B parameters) to rewrite generated
messages the way different staff would type them. The teacher never labels anything: the generator's labels are
kept, and a rewrite is rejected if any protected string (train, job card, date, km, department, urgency word) is
missing or changed, or if it contains a number that wasn't in the original. Fully offline, and safe to stop and
re-run. `generate_dataset.py` merges the rewrites in, holding 10% of them out for validation.

**Honest evaluation.** Besides the synthetic validation split, `data/realworld_test.jsonl` has 43 logbook
messages and 25 questions written by hand, in phrasings that do not come from the generator's templates.
Scores on that set are the ones to trust. `evaluate.py` compares extracted *facts* (train, action, department,
date, job card, km, priority) after the priority policy, as the supervisor sees them, and also reports the raw
engine score. A test guarantees that no test sentence ever appears in generated training data.

## Results

Hand-written real-world test set (43 logbook messages, 25 questions), scored as the supervisor sees it:

| Engine | Logbook fully correct | Fact precision / recall | Questions | Synthetic logbook / questions |
|---|---|---|---|---|
| Qwen2.5-0.5B, **no fine-tuning** (detailed prompt + 3 examples) | 4.7% | 15.0% / 6.1% | 24.0% | 7.5% / 19.1% |
| qwen3.5 9.7B via Ollama, no fine-tuning | 48.8% | 89.3% / 51.0% | 60.0% | not run |
| Rule engine (regular expressions) | 76.7% | 97.4% / 77.6% | 84.0% | 46.9% / 82.3% |
| kmrl-copilot round 1 | 90.7% | 91.8% / 91.8% | 88.0% | 88.1% / 94.7% |
| **kmrl-copilot round 2** | **97.7%** | **100% / 98.0%** | **100%** | **98.7% / 97.6%** |
| Gemini (cloud edition, reference) | 100% | 100% / 100% | 100% | not run |

Fine-tuning is what makes the difference: the same 0.5B model goes from 4.7% to 97.7%, and it beats a general
model 20x its size. Recall by update type, training curves and data composition: `reports/summary.md` and
`reports/figures/` (rebuild with `uv run --group report python ../ml/make_figures.py`). The terminal demo,
`uv run python ../ml/demo.py`, shows the raw model output next to what the app makes of it.

- **Round 1**: 8,000 template examples, model computes dates. Failed mostly on calendar arithmetic and once
  invented a certificate renewal.
- **Round 2**: + 1,336 teacher rewrites (qwen3.5), expiry copied as written and resolved by code, grounding
  check, priority policy labels. The one remaining miss is a job-card closure the model skipped; it no longer
  reports anything that isn't in the message (100% precision).

Training: ~12.5 min per round on an RTX 4060 laptop GPU, peak 3.3 GB VRAM. Inference on that GPU: a typical
handover note takes 4-6 s (Hugging Face `generate` runs at ~28 tokens/s on Windows), a question under 1 s.
Converting the model to GGUF and serving it with llama.cpp/Ollama would be several times faster; Ollama 0.40
cannot import this architecture from safetensors directly, so that conversion is a possible next step.

Reports (one per engine, each with every failure listed): `reports/eval_rules.json`, `eval_base.json`,
`eval_ollama.json`, `eval_model_round1.json`, `eval_model_round2.json`; training curves in
`reports/training_history_round{1,2}.json`.

## Improving it

- Add real (anonymised) logbook lines to `realworld_test.jsonl` first, so you can measure progress.
- When the model misses a phrasing, add a template or surface form to `generate_dataset.py` (never the test sentence
  itself), regenerate and retrain.
- More capacity: `train.py --base Qwen/Qwen2.5-1.5B-Instruct --batch 2 --accum 8` (fits in 8 GB, ~3x slower).
- The backend checks every model output against the fleet (unknown trains, closed job cards and past dates are
  rejected) and shows each update for confirmation, so a model mistake cannot silently change the fleet.
