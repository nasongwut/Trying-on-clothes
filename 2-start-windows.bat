@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Not installed yet - double-click 1-install-windows.bat first
  pause
  exit /b 1
)
set "HF_HOME=%~dp0hf-cache"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
chcp 65001 >nul
echo Starting AI Try-On...  Close this window to stop.
start "" cmd /c "timeout /t 4 >nul & start http://127.0.0.1:8765/"
.venv\Scripts\python.exe app.py
pause
