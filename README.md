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
3. Run the worker: `python worker.py`

## Runpod / container notes (NVIDIA / NVML)

If `nvidia-smi` goes from working to:

- `Failed to initialize NVML: Unknown Error`

…that almost always means **the container’s NVML user-space library no longer matches the host driver** (commonly caused by installing Ubuntu `nvidia-*` / `libnvidia-*` packages inside the container).

- **Do not install NVIDIA drivers inside the container**: avoid `apt-get install nvidia-*`, `cuda-*`, `libnvidia-*`.
- **Use `--no-install-recommends`** for system packages (to avoid pulling in GPU driver libraries as “recommended” deps).
- **If it breaks**: the most reliable fix is to **restart the pod/container**, then re-run your install but keep `dpkg -l | grep -Ei '(^ii\\s+nvidia|^ii\\s+cuda|^ii\\s+libnvidia)'` empty.

Quick debugging:

- Run `./scripts/gpu_diag.sh` **before** and **after** each step to find the exact command that flips NVML.

## Helper Scripts

- `scripts/gpu_diag.sh` – Prints GPU/NVML diagnostics to help debug `nvidia-smi` issues in containers
- `debug.py` – Debug tool for investigating tasks and worker state

## Debugging

Use `python debug.py` to investigate tasks and worker state:

```bash
python debug.py task <task_id>          # Investigate a specific task
python debug.py tasks --status Failed   # List recent failures
```

## Notes

- The wrapper prefers upstream‑first fixes to maintain clean Wan2GP updates.
