# KMRL Depot Planner — fully local AI edition

The induction planner with **our own fine-tuned language model** running on this PC. No API key and no
cloud calls; after a one-time download of the base model for training, it works fully offline.
See the [top-level README](../README.md) for what the app does.

## Run

```bat
train.bat     :: once: generate data, fine-tune (~15 min on an RTX 4060), evaluate
run.bat       :: http://127.0.0.1:8020
```

Until a model has been trained, the app uses the built-in rule engine and says so in the top bar.
The trained model lives in `backend/models/kmrl-copilot` (~1 GB, not committed to git).
How it is built and evaluated: [ml/README.md](ml/README.md).

## Engines

Set in `backend/.env` (see `.env.example`):

| `KMRL_LOCAL_ENGINE` | |
|---|---|
| `trained` (default) | The fine-tuned 0.5B copilot. Fast (well under a second per request on the GPU), ~1 GB of VRAM, also runs on CPU. |
| `ollama` | A general LLM from a local [Ollama](https://ollama.com) install, e.g. `OLLAMA_MODEL=qwen3:14b`. No training, but it needs a much bigger model and is much slower. |

`KMRL_MODEL_DEVICE=cpu` forces CPU inference.

## Develop

```bat
cd backend  &  uv run uvicorn app.main:app --reload --port 8020
cd frontend &  npm run dev          :: http://localhost:5174, proxies /api to :8020
cd backend  &  uv run pytest
```

The API is identical to the cloud edition's; interactive docs at <http://127.0.0.1:8020/docs>.
