#!/usr/bin/env bash
set -euo pipefail

cd /opt/armani_flp/pl

ENGINE_PATH="./pretrained_weights/tensorrt/unet_work.engine"

# Ensure dirs exist (safe even if already present)
mkdir -p "$(dirname "${ENGINE_PATH}")"

# Build the TensorRT engine only if it doesn't exist
if [ ! -f "${ENGINE_PATH}" ]; then
  echo "TensorRT engine not found at ${ENGINE_PATH}. Building..."
  python torch2trt.py
else
  echo "TensorRT engine found at ${ENGINE_PATH}. Skipping build."
fi

# Start FastAPI (Runpod load balancer expects it to listen on PORT)
exec uvicorn ws_server:app --host 0.0.0.0 --port "${PORT:-80}"
