# METECH Local — KMRL Depot Planner (fully offline AI)

AI-driven train induction planning for Kochi Metro Rail Limited, with **our own fine-tuned language model**
running on your machine: no API key, no cloud calls. Smart India Hackathon 2025, problem statement
**SIH25081**, team **METECH**.

Every night a KMRL depot supervisor decides, for all 25 trainsets, which go into revenue service at 05:30,
which wait on standby, and which go for inspection or deep cleaning. Today that is a two-hour spreadsheet
exercise. This app does it in under a second and explains every decision.

> The same app with Google Gemini instead of the local model lives at
> [MeezaUni/METECH-Cloud](https://github.com/MeezaUni/METECH-Cloud).

## Run it

**Windows:** double-click `run.bat` (or run it from a terminal).
**macOS / Linux:** `./run.sh`

On the first run it installs the dependencies (including PyTorch, ~3 GB), downloads the trained model
(~1 GB) from this repo's GitHub release, builds the dashboard and opens <http://127.0.0.1:8020>.
After that everything works offline. Requirements: [Node.js 20+](https://nodejs.org/) and
[uv](https://docs.astral.sh/uv/) (the script offers to install uv for you). An NVIDIA GPU makes the model
fast; without one it runs on the CPU, more slowly.

If the model can't be downloaded, the app still runs with its built-in rule engine, or you can train the
model yourself (below).

## What it does

- **Induction planner** (Google OR-Tools CP-SAT). Assigns every train to Service, Standby, Inspection Bay,
  Cleaning or Held, balancing the six factors in the problem statement:
  1. Fitness certificates: a train with *any* expired Rolling-Stock, Signalling or Telecom certificate is never inducted
  2. Job cards: open critical work keeps a train out; other open work prefers the inspection bay
  3. Branding: trains behind on contracted advertising hours are preferred for service
  4. Mileage balancing: trains with less wear since their last inspection are preferred
  5. Cleaning slots: overdue trains compete for the limited deep-cleaning bays
  6. Stabling geometry: inducting a train parked behind one that stays costs a shunting move
- **Explanations**: each decision lists its reasons and the score that tipped it.
- **What-if simulation**: stage changes as a draft, see the re-solved plan with before/after badges, then save or discard.
- **3D depot view**: stabling lines, inspection bay and cleaning bay; trains glide to their new track when the plan changes.
- **Logbook reader**: paste shift notes or WhatsApp messages; the on-device model turns them into fleet updates
  you confirm one by one. Every suggestion is grounded: its train, job card, km and expiry date must actually
  appear in the message, or it is dropped and you are told why.
- **Ask**: questions in plain English ("why is TS-09 on standby?"). The model only interprets the question;
  every number in the answer is computed from live data.
- **Priority policy** (`backend/app/priority.py`) and a full **audit trail**.

## The on-device model

**kmrl-logreader**: Qwen2.5-0.5B-Instruct fine-tuned with LoRA on 9,199 examples (8,000 synthetic depot
messages plus 1,199 natural rewrites from a larger local teacher model). It reads; code calculates: expiry
phrases like "for 6 months" are copied as written and turned into dates by tested code.

Hand-written real-world test set (43 logbook messages, 25 questions), scored as the supervisor sees it:

| | Rule engine | **kmrl-logreader** | Gemini (cloud edition) |
|---|---|---|---|
| Logbook messages fully correct | 76.7% | **97.7%** | 100% |
| Of the updates it suggests, correct | 97.4% | **100%** | 100% |
| Questions understood | 84.0% | **100%** | 100% |
| Time per logbook message (RTX 4060) | instant | ~4-6 s | ~2 s |

How the data is made, how it is trained and how it is evaluated: [ml/README.md](ml/README.md).

### Train it yourself

```bat
train.bat                 :: Windows: dataset + fine-tune (~13 min on an RTX 4060) + evaluation
```

or step by step from `backend/` (any OS):

```bash
uv sync --group train
uv run python ../ml/generate_dataset.py     # includes the committed teacher rewrites
uv run python ../ml/train.py                # writes backend/models/kmrl-logreader
uv run python ../ml/evaluate.py --engine model
```

Regenerating the teacher rewrites (`ml/paraphrase.py`) needs [Ollama](https://ollama.com) with `qwen3.5`;
the rewrites from our run are committed in `ml/data/paraphrased.jsonl`, so it is optional.

## Configuration

Optional, in `backend/.env` (see `.env.example`):

| Setting | |
|---|---|
| `KMRL_LOCAL_ENGINE=trained` | the fine-tuned kmrl-logreader model (default) |
| `KMRL_LOCAL_ENGINE=ollama` | a general LLM from a local Ollama install instead, e.g. `OLLAMA_MODEL=qwen3:14b` |
| `KMRL_MODEL_DEVICE=cpu` | force CPU inference |

Planner settings are in `backend/app/config.py`. The demo fleet is regenerated from **History → Reset demo data**.

## Develop

```bash
cd backend  && uv run uvicorn app.main:app --reload --port 8020
cd frontend && npm run dev          # http://localhost:5174, proxies /api to :8020
cd backend  && uv run pytest        # 56 tests
```

The API is described at <http://127.0.0.1:8020/docs>.

## Layout

```
backend/app/        FastAPI app; ai/local_model.py (on-device engine), ai/local_format.py (prompt, grounding)
backend/scripts/    get_model.py: downloads the trained model
backend/tests/      56 tests
frontend/src/       React + TypeScript + three.js dashboard
ml/                 dataset generator, teacher rewrites, training, evaluation, real-world test set
run.bat, run.sh, train.bat
```

## What changed from the hackathon prototype (v1.2)

| v1.2 (Sept 2025) | Now |
|---|---|
| Gemini called from the browser with the key in the bundle | No API at all: our own model, offline |
| Backend grounded a train only if *all* certificates had expired (the frontend said *any*) | One readiness check used everywhere, with a regression test |
| Edit dialog never showed job cards (`train.id` typo) | Fixed; job cards can be closed from the train panel |
| CSV files overwritten on save, no history | SQLite with an audit log and approved-plan history |
| 3 factors and a branding bonus | All 6 factors, depot capacities, explanations, shortfall alerts |
| No tests | 56 backend tests |
