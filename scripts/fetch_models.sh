#!/usr/bin/env bash
# Download upscaling weights into ./models (ignored by git).
set -euo pipefail
mkdir -p models
get() { [ -f "models/$1" ] || curl -fL -o "models/$1" "$2"; echo "models/$1"; }
get 4x-UltraSharp.pth       https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth
get RealESRGAN_x4plus.pth   https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
