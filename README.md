# Headless-Wan2GP

A headless wrapper around the Wan2GP engine that enables programmatic video generation and queue-based processing without the Gradio UI.

## Quick Start (GPU)
- Create venv: `python3 -m venv venv_headless_wan2gp && source venv_headless_wan2gp/bin/activate`
- Install deps:
  - `python -m pip install --upgrade pip wheel setuptools`
  - `python -m pip install -r Wan2GP/requirements.txt`
  - `python -m pip install -r requirements.txt`
- Run baseline tests (real generation):
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

## Maintenance Docs
- `AI_AGENT_MAINTENANCE_GUIDE.md` – Complete agent procedure for updates and validation
- `maintenance_analysis/` – Living documents during maintenance cycles:
  - `wan2gp_diff_analysis.md`, `system_integration_analysis.md`, `update_checklist.md`, `milestone_progress.txt`, `attempt_log.txt`

## Notes
- If you encounter CUDA OOM during baseline, reduce `video_length` in `test_model_comparison.py` (e.g., 65 → 33) and retry.
- The wrapper prefers upstream‑first fixes to maintain clean Wan2GP updates.
