# Reigh-Worker

**GPU worker for [Reigh](https://github.com/banodoco/Reigh)** — handles video generation tasks using [Wan2GP](https://github.com/deepbeepmeep/Wan2GP).

This is a headless wrapper around the Wan2GP engine that enables programmatic video generation and queue-based processing. It connects to the Reigh application to claim and execute video generation tasks on GPU infrastructure.

## Architecture

- **[Reigh](https://github.com/banodoco/Reigh)** — The main application (frontend + API)
- **Reigh-Worker** (this repo) — GPU worker that processes video generation jobs using Wan2GP

## Powered by Wan2GP

This worker is built on top of [Wan2GP](https://github.com/deepbeepmeep/Wan2GP), a powerful video generation engine. The `Wan2GP/` directory contains the upstream engine code.

---

## Quick Start (GPU)

1. Create venv: `python3 -m venv venv_headless_wan2gp && source venv_headless_wan2gp/bin/activate`
2. Install deps:
   - `python -m pip install --upgrade pip wheel setuptools`
   - `python -m pip install -r Wan2GP/requirements.txt`
   - `python -m pip install -r requirements.txt`
3. Run baseline tests (real generation):
   - `python test_model_comparison.py --output-dir outputs/baseline_real_$(date +%Y%m%d_%H%M%S)`

## Smoke Mode (CI/Headless)

When no GPU is available, you can validate the integration path and end‑to‑end flow without heavy generation.

- Usage:
  - `HEADLESS_WAN2GP_SMOKE=1 HEADLESS_WAN2GP_FORCE_CPU=1 python test_model_comparison.py --output-dir outputs/smoke_$(date +%Y%m%d_%H%M%S)`
- Behavior:
  - Skips Wan2GP import and model loading
  - Produces placeholder outputs under `Wan2GP/outputs/` and returns those paths
  - Leaves the upstream `Wan2GP/` code untouched

## Helper Scripts

- `scripts/run_baseline_smoke.sh` – Runs the smoke-mode baseline tests
- `scripts/run_baseline_real.sh` – Runs the real baseline tests (requires GPU)

## CI

- GitHub Actions workflow `Smoke Test` runs the smoke-mode baseline on push/PR to validate end-to-end integration without GPU.
- `GPU Baseline` (workflow_dispatch) runs the real baseline on a self-hosted GPU runner. You can optionally set `frames` and `steps` inputs to avoid OOM.

## Notes

- If you encounter CUDA OOM during baseline, reduce `video_length` in `test_model_comparison.py` (e.g., 65 → 33) and retry.
- The wrapper prefers upstream‑first fixes to maintain clean Wan2GP updates.
