#!/usr/bin/env bash
# em680: Turnaround to 3D on the RX 7600M XT eGPU, http://0.0.0.0:8091 (Mesh Studio is configured for 8090).
# Code, models and the shape venv live in ~/turnaround. The paint step uses ~/Wan2GP's ROCm venv unchanged.
T="$HOME/turnaround"

# GPU environment, as ~/src/pixal3d/run_pixal3d.sh sets it: only the eGPU (ROCm agent 1), no gfx override.
export ROCR_VISIBLE_DEVICES=1 ROCPROFILER_REGISTER_ENABLED=0 MIOPEN_FIND_MODE=FAST TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
unset HSA_OVERRIDE_GFX_VERSION

# Models were copied in, so nothing is downloaded.
export HF_HUB_OFFLINE=1 HF_HOME="$T/models/hf" HY3DGEN_MODELS="$T/models/hy3dgen" U2NET_HOME="$T/models/rembg"

export TURNAROUND_HY3DGEN_DIR="$T" TURNAROUND_SHAPE_PYTHON="$T/.venv/bin/python" TURNAROUND_WAN_DIR="$HOME/Wan2GP"
export TURNAROUND_MIN_FREE_VRAM_MIB=6500  # the card has 8 GiB; the shape step peaks at 5.7 GiB
# pymeshlab's plugins need libOpenGL.so.0, which is not installed system-wide; a copy unpacked from Ubuntu's libopengl0 is in lib/.
export LD_LIBRARY_PATH="$T/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
exec "$T/.venv/bin/python" "$T/app/server.py" --port "${TURNAROUND_PORT:-8091}"
