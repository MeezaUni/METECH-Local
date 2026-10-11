@echo off
rem METECH Local - KMRL Depot Planner with an on-device AI model. Serves http://127.0.0.1:8020
setlocal
cd /d "%~dp0"
title METECH Local - KMRL Depot Planner

rem ---- prerequisites -------------------------------------------------------
where npm >nul 2>nul
if errorlevel 1 (
  echo [!] Node.js 20 or newer is required. Install it from https://nodejs.org/ and run this again.
  pause & exit /b 1
)
where uv >nul 2>nul
if errorlevel 1 (
  echo uv ^(the Python package manager^) is not installed.
  choice /c YN /m "Install uv now from https://astral.sh/uv"
  if errorlevel 2 ( echo Install it from https://docs.astral.sh/uv/ and run this again. & pause & exit /b 1 )
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
where uv >nul 2>nul || ( echo [!] uv is still not available. Open a new terminal and run this again. & pause & exit /b 1 )

rem ---- build ----------------------------------------------------------------
pushd frontend
if not exist node_modules ( call npm ci || ( popd & pause & exit /b 1 ) )
call npm run build || ( popd & pause & exit /b 1 )
popd

pushd backend
echo Installing Python packages (the first run downloads PyTorch, about 3 GB)...
uv sync --inexact || ( popd & pause & exit /b 1 )

rem ---- the trained model (downloaded once) ----------------------------------
if not exist models\kmrl-copilot\config.json call :get_model

echo.
echo   METECH Local is running at http://127.0.0.1:8020   (close this window to stop)
echo.
if not defined KMRL_NO_BROWSER start "" http://127.0.0.1:8020
uv run uvicorn app.main:app --host 127.0.0.1 --port 8020
popd
exit /b

:get_model
uv run python scripts\get_model.py
if errorlevel 1 (
  echo.
  echo   Continuing with the built-in rule engine. Run train.bat to train the model yourself.
  echo.
)
exit /b
