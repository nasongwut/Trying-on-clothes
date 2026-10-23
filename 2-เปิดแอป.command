#!/bin/bash
# เปิดห้องลองชุด AI แล้วเปิดเบราว์เซอร์ — ปิดหน้าต่างนี้เมื่อเลิกใช้
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "ยังไม่ได้ติดตั้ง — ดับเบิลคลิก '1-ติดตั้ง.command' ก่อน"
  read -n 1 -s -r; exit 1
fi
export HF_HOME="$(pwd)/hf-cache"
(sleep 3; open "http://127.0.0.1:8765/") &
.venv/bin/python app.py
