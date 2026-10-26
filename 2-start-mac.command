#!/bin/bash
# Start AI Fitting Room and open the browser. Close this window to stop.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Not installed yet - double-click '1-install-mac.command' first"
  read -n 1 -s -r; exit 1
fi
export HF_HOME="$(pwd)/hf-cache"
(sleep 3; open "http://127.0.0.1:8765/") &
.venv/bin/python app.py
