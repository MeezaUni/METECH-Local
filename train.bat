@echo off
rem Generate the dataset, fine-tune the on-device copilot model and score it.
setlocal
cd /d "%~dp0backend"
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
uv sync --group train || exit /b 1
uv run python ..\ml\generate_dataset.py || exit /b 1
uv run python ..\ml\train.py %* || exit /b 1
uv run python ..\ml\evaluate.py --engine rules
uv run python ..\ml\evaluate.py --engine model
