@echo off
rem KMRL Depot Planner - fully local AI edition. Serves everything on http://127.0.0.1:8020
setlocal
cd /d "%~dp0"
where uv >nul 2>nul || (echo [!] uv is required: https://docs.astral.sh/uv/getting-started/installation/ & exit /b 1)
where npm >nul 2>nul || (echo [!] Node.js is required: https://nodejs.org/ & exit /b 1)

pushd frontend
if not exist node_modules call npm install || exit /b 1
call npm run build || exit /b 1
popd

pushd backend
uv sync || exit /b 1
if not exist models\kmrl-copilot\config.json (
  echo.
  echo   [!] No trained model found in backend\models\kmrl-copilot.
  echo       Run train.bat once ^(about 15 minutes on an RTX 4060^). Until then the rule engine is used.
)
echo.
echo   KMRL Depot Planner (local AI) on http://127.0.0.1:8020
echo.
start "" http://127.0.0.1:8020
uv run uvicorn app.main:app --host 127.0.0.1 --port 8020
