#!/usr/bin/env bash
# Polaris / RX 590 resurrection env.
# Source before training / pricing:  source scripts/env_polaris.sh
#
# Two different GPU stories on this card:
# 1) MascotRL Layer-2 pricing → Mesa rusticl OpenCL (needs RUSTICL_ENABLE=radeonsi)
# 2) Local LLMs (Ollama) → Vulkan + HSA spoof (Villorente); not used by PyTorch here
#    ROCm dropped gfx803; PyTorch training stays on CPU.

# Mesa rusticl: without this, clinfo shows platform but 0 devices
export RUSTICL_ENABLE="${RUSTICL_ENABLE:-radeonsi}"

# ROCm/HIP spoof (harmless for rusticl; required for any ROCm/Ollama-Vulkan path)
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-8.0.3}"
export OLLAMA_VULKAN="${OLLAMA_VULKAN:-1}"

# Prefer discrete AMD GPU when multiple OpenCL devices exist
export MASCOTRL_OPENCL_VENDOR="${MASCOTRL_OPENCL_VENDOR:-AMD}"
export MASCOTRL_OPENCL_DEVICE_HINT="${MASCOTRL_OPENCL_DEVICE_HINT:-RX 590|polaris10|Ellesmere|gfx803|radeonsi|Radeon}"

# Use OpenCL only when batch is large enough to beat AVX2 (override with 1 to force)
export MASCOTRL_OCL_MIN_N="${MASCOTRL_OCL_MIN_N:-256}"

if [[ -d /dev/dri ]]; then
  export MASCOTRL_DRI_PRESENT=1
fi

echo "[mascotrl] RUSTICL_ENABLE=${RUSTICL_ENABLE}"
echo "[mascotrl] HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}"
echo "[mascotrl] MASCOTRL_OPENCL_DEVICE_HINT=${MASCOTRL_OPENCL_DEVICE_HINT}"
echo "[mascotrl] MASCOTRL_OCL_MIN_N=${MASCOTRL_OCL_MIN_N}"
