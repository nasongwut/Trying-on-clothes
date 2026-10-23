#!/bin/bash
# ติดตั้งห้องลองชุด AI (Flask + FASHN VTON v1.5) — รันครั้งเดียว
set -e
cd "$(dirname "$0")"
echo "=== ติดตั้งห้องลองชุด AI (FASHN VTON v1.5) ==="

# 1) uv: ตัวจัดการ Python (ติดตั้ง Python 3.11 แยกให้เอง ไม่ยุ่งกับ Python ของระบบ)
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "[1/4] ติดตั้ง uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
else
  echo "[1/4] มี uv แล้ว"
fi

# 2) ดาวน์โหลดโค้ด FASHN VTON v1.5
if [ ! -d fashn-vton-1.5 ]; then
  echo "[2/4] ดาวน์โหลดโค้ด FASHN VTON v1.5 ..."
  curl -L -o fashn.zip https://github.com/fashn-AI/fashn-vton-1.5/archive/refs/heads/main.zip
  unzip -q fashn.zip && mv fashn-vton-1.5-main fashn-vton-1.5 && rm fashn.zip
else
  echo "[2/4] มีโค้ดแล้ว"
fi

# 3) Python environment + ไลบรารี (ใช้ onnxruntime ธรรมดาแทน onnxruntime-gpu ซึ่งไม่มีบน Mac)
echo "[3/4] ติดตั้งไลบรารี (PyTorch ฯลฯ อาจใช้เวลาหลายนาที) ..."
[ -d .venv ] || uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-deps -e ./fashn-vton-1.5

# 4) น้ำหนักโมเดล (~2 GB) + human parser (~244 MB)
echo "[4/4] ดาวน์โหลดโมเดล (~2.3 GB) ..."
export HF_HOME="$(pwd)/hf-cache"
.venv/bin/python fashn-vton-1.5/scripts/download_weights.py --weights-dir ./weights
.venv/bin/python -c "from fashn_human_parser import FashnHumanParser; FashnHumanParser(device='cpu'); print('human parser OK')"

echo ""
echo "✅ ติดตั้งเสร็จแล้ว — ดับเบิลคลิก '2-เปิดแอป.command' เพื่อเริ่มใช้งาน"
read -n 1 -s -r -p "กดปุ่มใดก็ได้เพื่อปิด"
