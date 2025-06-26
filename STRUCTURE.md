# Project Structure

```
<repo-root>
├── add_task.py
├── generate_test_tasks.py
├── headless.py
├── source/
│   ├── __init__.py
│   ├── common_utils.py
│   ├── db_operations.py
│   ├── specialized_handlers.py
│   ├── video_utils.py
│   ├── wgp_utils.py
│   └── sm_functions/
│       ├── __init__.py
│       ├── travel_between_images.py
│       ├── different_perspective.py
│       └── single_image.py
├── logs/               # runtime logs (git-ignored)
├── outputs/            # generated videos/images (git-ignored)
├── samples/            # example inputs for docs & tests
├── tests/              # pytest suite
├── test_outputs/       # artefacts written by tests (git-ignored)
├── Wan2GP/  ← third-party video-generation engine (keep high-level only)
└── STRUCTURE.md  (this file)
```

## Top-level scripts

* **headless.py** – Headless service that polls the `tasks` database, claims work, and drives the Wan2GP generator (`wgp.py`). Includes extra handlers for OpenPose and RIFE interpolation tasks and can upload outputs to Supabase storage.
* **add_task.py** – Lightweight CLI helper to queue a single new task into SQLite/Supabase. Accepts a JSON payload (or file) and inserts it into the `tasks` table.
* **generate_test_tasks.py** – Developer utility that back-fills the database with synthetic images/prompts for integration testing and local benchmarking.

## source/ package

This is the main application package.

* **common_utils.py** – Reusable helpers (file downloads, ffmpeg helpers, MediaPipe keypoint interpolation, debug utilities, etc.)
* **db_operations.py** – Handles all database interactions for both SQLite and Supabase.
* **specialized_handlers.py** – Contains handlers for specific, non-standard tasks like OpenPose generation and RIFE interpolation.
* **video_utils.py** – Provides utilities for video manipulation like cross-fading, frame extraction, and color matching.
* **wgp_utils.py** – Thin wrapper around `Wan2GP.wgp` that standardises parameter names, handles LoRA quirks (e.g. CausVid, LightI2X), and exposes the single `generate_single_video` helper used by every task handler.

### source/sm_functions/ sub-package

Task-specific wrappers around the bulky upstream logic. These are imported by `headless.py` (and potentially by notebooks/unit tests) without dragging in the interactive Gradio UI shipped with Wan2GP.

* **travel_between_images.py** – Implements the segment-by-segment interpolation pipeline between multiple anchor images. Builds guide videos, queues generation tasks, stitches outputs.
* **different_perspective.py** – Generates a new perspective for a single image using an OpenPose or depth-driven guide video plus optional RIFE interpolation for smoothness.
* **single_image.py** – Minimal handler for one-off image-to-video generation without travel or pose manipulation.
* **__init__.py** – Re-exports public APIs (`run_travel_between_images_task`, `run_single_image_task`, `run_different_perspective_task`) and common utilities for convenient importing.

## Additional runtime artefacts & folders

* **logs/** – Rolling log files captured by `headless.py` and unit tests. The directory is git-ignored.
* **outputs/** – Default location for final video/image results when not explicitly overridden by a task payload.
* **samples/** – A handful of small images shipped inside the repo that are referenced in the README and tests.
* **tests/** – Pytest-based regression and smoke tests covering both low-level helpers and full task workflows.
* **test_outputs/** – Artefacts produced by the test-suite; kept out of version control via `.gitignore`.
* **tasks.db** – SQLite database created on-demand by the orchestrator to track queued, running, and completed tasks.

## Wan2GP/

**Git Submodule** pointing to the upstream [deepbeepmeep/Wan2GP](https://github.com/deepbeepmeep/Wan2GP) repository. This contains the video-generation engine (`wgp.py`) together with model checkpoints, inference helpers, preprocessing code, and assorted assets.

`wgp.py` is capable of the following AI processes:
*   Wan2.1 text2video (1.3B & 14B): Standard text-to-video generation.
*   Wan2.1 image2video (480p & 720p, 14B): Standard image-to-video generation.
*   Fun InP image2video (1.3B & 14B): Alternative image-to-video with end-image fixing.
*   Vace ControlNet (1.3B & 14B): Controllable video generation using pose, depth, or object references.
*   ReCamMaster (1.3B & 14B): Replays a video with different camera movements.
*   Wan2.1 FLF2V (720p, 14B): Image-to-video supporting start and end frames.
*   SkyReels2 Diffusion Forcing (1.3B & 14B, 540p & 720p): Generates long videos and extends existing videos.
*   Wan2.1 Phantom (1.3B & 14B): Transfers people or objects into a generated video.
*   Wan2.1 Fantasy Speaking (720p, 14B): Image-to-video with audio input processing.
*   Wan2.1 MoviiGen (1080p, 14B): Cinematic video generation in 720p or 1080p (21:9).
*   LTX Video (0.9.7, 13B & Distilled 13B): Fast generation of long videos (up to 260 frames).
*   Hunyuan Video text2video (720p, 13B): High-quality text-to-video generation.
*   Hunyuan Video image2video (720p, 13B): Image-to-video generation.
*   Hunyuan Video Custom (720p, 13B): Transfers people (identity-preserving) into videos.
*   Video Mask Creator (MatAnyone & SAM2): For creating masks for inpainting/outpainting.
*   Prompt Enhancer (Florence2 & Llama3_2): Enhances prompts using LLMs for better video generation.
*   Temporal Upsampling (RIFE): Increases video fluidity (frame rate).
*   Spatial Upsampling (Lanczos): Increases video resolution.

The submodule is currently pinned to commit `6706709` ("optimization for i2v with CausVid") and can be updated periodically using standard git submodule commands. Only the entry module `wgp.py` is imported directly; everything else stays encapsulated within the submodule.

## Runtime artefacts

* **tasks.db** – SQLite database created on-demand by the orchestrator/server to track queued, running, and completed tasks.
* **public/files/** – For SQLite mode, all final video outputs are saved directly here with descriptive filenames (e.g., `{run_id}_seg00_output.mp4`, `{run_id}_final.mp4`). No nested subdirectories are created.
* **outputs/** – For non-SQLite modes or when explicitly configured, videos are saved here with task-specific subdirectories.

## End-to-End task lifecycle (1-minute read)

1. **Task injection** – A CLI, API, or test script calls `add_task.py`, which inserts a new row into the `tasks` table (SQLite or Supabase).  Payload JSON is stored in `params`, `status` is set to `pending`.
2. **Worker pickup** – `headless.py` runs in a loop, atomically updates a `pending` row to `running`, and inspects `task_type` to choose the correct handler.
3. **Handler execution**
   * Standard tasks live in `source/sm_functions/…` (see table below).
   * Special one-offs (OpenPose, RIFE, etc.) live in `specialized_handlers.py`.
   * Handlers may queue **sub-tasks** (e.g. travel → N segments + 1 stitch) by inserting new rows with `dependant_on` set, forming a DAG.
4. **Video generation** – Every handler eventually calls `wgp_utils.generate_single_video` which wraps **Wan2GP/wgp.py** and returns a path to the rendered MP4.
5. **Post-processing** – Optional saturation / brightness / colour-match (`video_utils.py`) or upscaling tasks.
6. **DB update** – Handler stores `output_location` (relative in SQLite, absolute otherwise) and marks the row `completed` (or `failed`).  Dependants are now eligible to start.
7. **Cleanup** – Intermediate folders are deleted unless `debug_mode_enabled` or `skip_cleanup_enabled` flags are set in the payload.

## Quick task-to-file reference

| Task type / sub-task | Entrypoint function | File |
|----------------------|---------------------|------|
| Travel orchestrator  | `_handle_travel_orchestrator_task` | `sm_functions/travel_between_images.py` |
| Travel segment       | `_handle_travel_segment_task`      | " " |
| Travel stitch        | `_handle_travel_stitch_task`       | " " |
| Single image video   | `run_single_image_task`            | `sm_functions/single_image.py` |
| Different perspective | `run_different_perspective_task`   | `sm_functions/different_perspective.py` |
| OpenPose mask video  | `handle_openpose_task`             | `specialized_handlers.py` |
| RIFE interpolation   | `handle_rife_task`                 | `specialized_handlers.py` |

All of the above eventually call `wgp_utils.generate_single_video`, which is the single **shared** bridge into Wan2GP.

## Database cheat-sheet

Column | Purpose
-------|---------
`id` | UUID primary key (task_id)
`task_type` | e.g. `travel_segment`, `wgp`, `travel_stitch`
`dependant_on` | Optional FK forming execution DAG
`params` | JSON payload saved by the enqueuer
`status` | `pending` → `running` → `completed`/`failed`
`output_location` | Where the final artefact lives (string)
`updated_at` | Heartbeat & ordering

SQLite keeps the DB at `tasks.db`; Supabase uses the same columns with RLS policies.

## LoRA Support

### Special LoRA Flags

* **`use_causvid_lora`** – Enables CausVid LoRA with 9 steps, guidance 1.0, flow-shift 1.0. Auto-downloads from HuggingFace if missing.
* **`use_lighti2x_lora`** – Enables LightI2X LoRA with 5 steps, guidance 1.0, flow-shift 5.0, Tea Cache disabled. Auto-downloads from HuggingFace if missing.

Both flags automatically configure optimal generation parameters and handle LoRA downloads/activation.

## Environment & config knobs (non-exhaustive)

Variable / flag | Effect
----------------|-------
`SUPABASE_URL / SUPABASE_KEY` | Switches DB operations to Supabase mode.
`WAN2GP_CACHE` | Where Wan2GP caches model weights.
`--debug` | Prevents cleanup of temp folders, extra logs.
`--skip_cleanup` | Keeps all intermediate artefacts even outside debug.

---

Keep this file **brief** – for in-depth developer docs see the `docs/` folder and inline module docstrings. 