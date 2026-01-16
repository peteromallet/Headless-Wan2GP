#!/usr/bin/env bash
set -euo pipefail

TAG="[NVMLInstallBreak]"

echo "${TAG} timestamp: $(date -Iseconds)"
echo "${TAG} user: $(id -u -n 2>/dev/null || true) uid=$(id -u) gid=$(id -g)"
echo "${TAG} pwd: $(pwd)"
echo "${TAG} uname: $(uname -a)"
echo "${TAG} os-release:"
cat /etc/os-release 2>/dev/null | sed "s/^/${TAG}   /" || true

echo
echo "${TAG} /dev/nvidia*:"
ls -la /dev/nvidia* 2>/dev/null | sed "s/^/${TAG}   /" || echo "${TAG}   (no /dev/nvidia* devices visible)"

echo
echo "${TAG} /proc/driver/nvidia/version:"
if [[ -f /proc/driver/nvidia/version ]]; then
  sed "s/^/${TAG}   /" /proc/driver/nvidia/version
else
  echo "${TAG}   (missing /proc/driver/nvidia/version)"
fi

echo
echo "${TAG} nvidia-smi location/version:"
command -v nvidia-smi >/dev/null 2>&1 && {
  echo "${TAG}   which: $(command -v nvidia-smi)"
  nvidia-smi --version 2>&1 | sed "s/^/${TAG}   /" || true
} || {
  echo "${TAG}   (nvidia-smi not found in PATH)"
}

echo
echo "${TAG} nvidia-smi output:"
nvidia-smi 2>&1 | sed "s/^/${TAG}   /" || true

echo
echo "${TAG} dpkg NVIDIA/CUDA packages (if any):"
if command -v dpkg >/dev/null 2>&1; then
  dpkg -l | grep -Ei '(^ii\s+nvidia|^ii\s+cuda|^ii\s+libnvidia)' | sed "s/^/${TAG}   /" || echo "${TAG}   (none found)"
else
  echo "${TAG}   (dpkg not available)"
fi

echo
echo "${TAG} ldconfig libnvidia-ml (if any):"
if command -v ldconfig >/dev/null 2>&1; then
  ldconfig -p 2>/dev/null | grep -i 'libnvidia-ml' | sed "s/^/${TAG}   /" || echo "${TAG}   (not found via ldconfig)"
else
  echo "${TAG}   (ldconfig not available)"
fi

echo
echo "${TAG} python/torch cuda check (if torch is installed):"
if command -v python >/dev/null 2>&1; then
  python - <<'PY' 2>&1 | sed "s/^/'"${TAG}   "'/"
import os

print("python:", __import__("sys").version.replace("\n", " "))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

try:
    import torch
except Exception as e:
    print("torch import failed:", repr(e))
    raise SystemExit(0)

print("torch:", getattr(torch, "__version__", "<unknown>"))
print("torch.version.cuda:", getattr(getattr(torch, "version", None), "cuda", None))
try:
    avail = torch.cuda.is_available()
except Exception as e:
    print("torch.cuda.is_available() raised:", repr(e))
    raise SystemExit(0)

print("torch.cuda.is_available():", avail)
if avail:
    print("torch.cuda.device_count():", torch.cuda.device_count())
    print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
PY
else
  echo "${TAG}   (python not found in PATH)"
fi
