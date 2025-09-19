#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d venv_headless_wan2gp ]; then
  python3 -m venv venv_headless_wan2gp
fi
source venv_headless_wan2gp/bin/activate

python -m pip install --upgrade pip wheel setuptools >/dev/null
python -m pip install -r Wan2GP/requirements.txt -q || true
python -m pip install -r requirements.txt -q || true

python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
PY

mkdir -p outputs
OUTDIR="outputs/baseline_real_$(date +%Y%m%d_%H%M%S)"
ARGS=(--output-dir "$OUTDIR")
if [ -n "${FRAMES:-}" ]; then ARGS+=(--frames "$FRAMES"); fi
if [ -n "${STEPS:-}" ]; then ARGS+=(--steps "$STEPS"); fi
python test_model_comparison.py "${ARGS[@]}"
echo "Baseline complete. Output dir: $OUTDIR"
