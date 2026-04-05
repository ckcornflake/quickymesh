#!/bin/bash
# startup.sh — Start ComfyUI and the quickymesh API server in the same container.
#
# ComfyUI runs in the background on :8190.
# quickymesh API server runs in the background on :8000.
# If either process exits, both are stopped and the container exits.

set -e

# ---- signal handler ----------------------------------------------------------
# Forward SIGTERM/SIGINT to both children so they can shut down gracefully.
_shutdown() {
    echo "[startup] Received shutdown signal. Stopping children..."
    [ -n "$COMFYUI_PID" ] && kill "$COMFYUI_PID" 2>/dev/null || true
    [ -n "$QM_PID"      ] && kill "$QM_PID"      2>/dev/null || true
    wait
    exit 0
}
trap _shutdown SIGTERM SIGINT

# ---- ComfyUI -----------------------------------------------------------------
echo "[startup] Starting ComfyUI on :8190 ..."
cd /app
python main.py --listen 0.0.0.0 --port 8190 &
COMFYUI_PID=$!

# ---- quickymesh API server ---------------------------------------------------
echo "[startup] Starting quickymesh API server on :8000 ..."
cd /quickymesh
python api_server.py --host 0.0.0.0 --port 8000 &
QM_PID=$!

echo "[startup] Both services started."
echo "[startup]   ComfyUI      PID=$COMFYUI_PID  →  http://localhost:8190"
echo "[startup]   quickymesh   PID=$QM_PID        →  http://localhost:8000"

# ---- wait for either child to exit -------------------------------------------
# bash 5+ (Ubuntu 24.04 ships bash 5.2): wait -n exits when any child exits.
wait -n "$COMFYUI_PID" "$QM_PID"
EXIT_CODE=$?

echo "[startup] A child process exited (code=$EXIT_CODE). Stopping all..."
kill "$COMFYUI_PID" "$QM_PID" 2>/dev/null || true
wait
exit $EXIT_CODE
