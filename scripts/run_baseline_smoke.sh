#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d venv_headless_wan2gp ]; then
  python3 -m venv venv_headless_wan2gp
fi
source venv_headless_wan2gp/bin/activate

python -m pip install --upgrade pip wheel setuptools >/dev/null
# Skip heavy upstream requirements in smoke mode unless explicitly requested
if [ "${INSTALL_WAN2GP_REQS:-0}" = "1" ]; then
  python -m pip install -r Wan2GP/requirements.txt -q || true
fi
python -m pip install -r requirements.txt -q || true

mkdir -p outputs
export HEADLESS_WAN2GP_SMOKE=1
export HEADLESS_WAN2GP_FORCE_CPU=1
python test_model_comparison.py --output-dir "outputs/smoke_$(date +%Y%m%d_%H%M%S)"
