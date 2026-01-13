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

> **Note:** Running `worker.py` requires API credentials from [reigh.art](https://reigh.art/) to connect to the task queue. For standalone usage without credentials, see [Standalone Usage](#standalone-usage-without-workerpy) below.

## Standalone Usage (without worker.py)

You can use the generation engine directly without connecting to Reigh's database. This is useful for local testing, scripting, or building custom pipelines.

### Using the Examples

The `examples/` directory contains self-contained scripts:

```bash
# Join two video clips with a smooth AI-generated transition
python examples/join_clips_example.py \
    --clip1 scene1.mp4 \
    --clip2 scene2.mp4 \
    --output transition.mp4 \
    --prompt "smooth camera glide between scenes"

# Regenerate a range of frames within a video (fix corrupted frames, etc.)
python examples/inpaint_frames_example.py \
    --video my_video.mp4 \
    --start-frame 45 \
    --end-frame 61 \
    --output fixed_video.mp4 \
    --prompt "smooth continuous motion"
```

### Using HeadlessTaskQueue Directly

For custom scripts, use `HeadlessTaskQueue` to manage model loading and generation:

```python
import os, sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
wan_root = project_root / "Wan2GP"

from headless_model_management import HeadlessTaskQueue, GenerationTask

# Initialize queue (loads models on first use)
queue = HeadlessTaskQueue(wan_dir=str(wan_root), max_workers=1)
queue.start()

try:
    # Create a generation task
    task = GenerationTask(
        id="my_task_001",
        model="wan_2_2_vace_lightning_baseline_2_2_2",
        prompt="a cat walking through a garden",
        parameters={
            "video_length": 81,
            "resolution": "896x512",
            "num_inference_steps": 6,
            "guidance_scale": 3.0,
            "seed": 42,
        }
    )
    
    # Submit and wait for result
    queue.submit_task(task)
    result = queue.wait_for_completion(task.id, timeout=600)
    
    if result.get("success"):
        print(f"Generated: {result['output_path']}")
    else:
        print(f"Failed: {result.get('error')}")
finally:
    queue.stop()
```

### Available Models

Common model presets (see `Wan2GP/defaults/*.json` for full list):

| Model Key | Type | Description |
|-----------|------|-------------|
| `wan_2_2_vace_lightning_baseline_2_2_2` | VACE | Fast video inpainting/editing (6 steps) |
| `wan_2_2_i2v_lightning_baseline_2_2_2` | I2V | Image-to-video with Lightning LoRAs |
| `wan_2_2_t2v_14B` | T2V | Text-to-video (14B params) |
| `qwen_image_edit_20B` | Image | Qwen-based image editing |

## Helper Scripts

- `scripts/gpu_diag.sh` – Prints GPU/NVML diagnostics to help debug `nvidia-smi` issues in containers
- `debug.py` – Debug tool for investigating tasks and worker state
- `create_test_task.py` – Create test tasks in Supabase for debugging

## Debugging

Use `python debug.py` to investigate tasks and worker state:

```bash
python debug.py task <task_id>          # Investigate a specific task
python debug.py tasks --status Failed   # List recent failures
```

## Notes

- The wrapper prefers upstream‑first fixes to maintain clean Wan2GP updates.
