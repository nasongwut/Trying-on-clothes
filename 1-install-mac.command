#!/bin/bash
# AI Fitting Room - Mac setup (Flask + FASHN VTON v1.5). Run once.
set -e
cd "$(dirname "$0")"
echo "=== AI Fitting Room - Mac setup (FASHN VTON v1.5) ==="

# 1) uv: Python manager (installs its own Python 3.11, leaves the system Python alone)
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "[1/4] Installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
else
  echo "[1/4] uv already installed"
fi

# 2) FASHN VTON v1.5 source code
if [ ! -d fashn-vton-1.5 ]; then
  echo "[2/4] Downloading FASHN VTON v1.5 code ..."
  curl -L -o fashn.zip https://github.com/fashn-AI/fashn-vton-1.5/archive/refs/heads/main.zip
  unzip -q fashn.zip && mv fashn-vton-1.5-main fashn-vton-1.5 && rm fashn.zip
else
  echo "[2/4] Code already downloaded"
fi

# 3) Python environment + libraries (plain onnxruntime instead of onnxruntime-gpu, which has no Mac build)
echo "[3/4] Installing libraries (PyTorch etc., may take several minutes) ..."
[ -d .venv ] || uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-deps -e ./fashn-vton-1.5

# 4) Model weights (~2 GB) + human parser (~244 MB)
echo "[4/4] Downloading model weights (~2.3 GB) ..."
export HF_HOME="$(pwd)/hf-cache"
.venv/bin/python fashn-vton-1.5/scripts/download_weights.py --weights-dir ./weights
.venv/bin/python -c "from fashn_human_parser import FashnHumanParser; FashnHumanParser(device='cpu'); print('human parser OK')"

echo ""
echo "DONE! Now double-click '2-start-mac.command'"
read -n 1 -s -r -p "Press any key to close"
