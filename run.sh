#!/usr/bin/env bash
# METECH Local - KMRL Depot Planner with an on-device AI model. Serves http://127.0.0.1:8020
set -euo pipefail
cd "$(dirname "$0")"

command -v npm >/dev/null || { echo "Node.js 20 or newer is required: https://nodejs.org/"; exit 1; }
if ! command -v uv >/dev/null; then
  read -r -p "uv (the Python package manager) is not installed. Install it now from https://astral.sh/uv? [y/N] " ok
  [[ "$ok" =~ ^[Yy] ]] || { echo "Install it from https://docs.astral.sh/uv/ and run this again."; exit 1; }
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

(cd frontend && { [ -d node_modules ] || npm ci; } && npm run build)
cd backend
echo "Installing Python packages (the first run downloads PyTorch)..."
uv sync --inexact
if [ ! -f models/kmrl-copilot/config.json ]; then
  uv run python scripts/get_model.py || echo "  Continuing with the built-in rule engine. See README to train the model yourself."
fi
echo; echo "  METECH Local is running at http://127.0.0.1:8020   (Ctrl+C to stop)"; echo
[ -n "${KMRL_NO_BROWSER:-}" ] || ( sleep 5; (command -v xdg-open >/dev/null && xdg-open http://127.0.0.1:8020) || (command -v open >/dev/null && open http://127.0.0.1:8020) || true ) &
exec uv run uvicorn app.main:app --host 127.0.0.1 --port 8020
