@echo off
setlocal
cd /d "%~dp0"
echo ==========================================================
echo   AI Try-On  -  Windows setup  (FASHN VTON v1.5)
echo   Run once. Needs internet and about 8 GB free disk.
echo ==========================================================

rem ---- 1) uv : Python manager (installs its own Python 3.11) ----
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
  echo [1/4] Installing uv...
  powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex" || goto :fail
) else (
  echo [1/4] uv already installed
)

rem ---- 2) FASHN VTON source code ----
if not exist fashn-vton-1.5 (
  echo [2/4] Downloading FASHN VTON code...
  curl -L -o fashn.zip https://github.com/fashn-AI/fashn-vton-1.5/archive/refs/heads/main.zip || goto :fail
  tar -xf fashn.zip || goto :fail
  ren fashn-vton-1.5-main fashn-vton-1.5 || goto :fail
  del fashn.zip
) else (
  echo [2/4] Code already downloaded
)

rem ---- 3) Python env + libraries ----
echo [3/4] Installing Python 3.11 and libraries - this can take a while...
if not exist .venv\Scripts\python.exe (
  uv venv --python 3.11 .venv || goto :fail
)
rem  PyTorch: uv picks the CUDA build automatically if an NVIDIA GPU is found, otherwise CPU
uv pip install --python .venv\Scripts\python.exe torch torchvision --torch-backend=auto || goto :fail
uv pip install --python .venv\Scripts\python.exe -r requirements.txt || goto :fail
uv pip install --python .venv\Scripts\python.exe --no-deps -e .\fashn-vton-1.5 || goto :fail

rem ---- 4) Model weights ----
echo [4/4] Downloading model weights - about 2.3 GB...
set "HF_HOME=%~dp0hf-cache"
.venv\Scripts\python.exe fashn-vton-1.5\scripts\download_weights.py --weights-dir .\weights || goto :fail
.venv\Scripts\python.exe -c "from fashn_human_parser import FashnHumanParser; FashnHumanParser(device='cpu'); print('human parser OK')" || goto :fail
.venv\Scripts\python.exe -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not found - will run on CPU (slow)')"

echo.
echo DONE!  Now double-click  2-start-windows.bat
pause
exit /b 0

:fail
echo.
echo *** SETUP FAILED - please copy the error message above ***
pause
exit /b 1
