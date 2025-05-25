# Project Structure

```
<repo-root>
├── steerable_motion.py
├── headless.py
├── sm_functions/
│   ├── __init__.py
│   ├── common_utils.py
│   ├── travel_between_images.py
│   └── different_pose.py
├── Wan2GP/  ← third-party video-generation engine (keep high-level only)
└── STRUCTURE.md  (this file)
```

## Top-level scripts

**steerable_motion.py** – CLI orchestrator. Parses user arguments for the two primary tasks, ensures the SQLite `tasks` table exists, sets global/debug flags and then delegates to task handlers in `sm_functions`.

**headless.py** – Headless server that continuously polls the `tasks` database, claims work, and drives the Wan2GP generator (`wgp.py`). Includes extra handlers for OpenPose and RIFE interpolation tasks and can upload outputs to Supabase storage.

## sm_functions/ package

A light-weight, testable wrapper around the bulky `steerable_motion.py` logic.  It holds:

* **common_utils.py** – Reusable helpers (DB access, ffmpeg helpers, MediaPipe keypoint interpolation, debug utilities, etc.)
* **travel_between_images.py** – Implements the segment-by-segment interpolation pipeline between multiple anchor images.  Builds guide videos, queues generation tasks, stitches outputs.
* **different_pose.py** – Generates a new pose for a single image using an OpenPose-driven guide video plus optional RIFE interpolation for smoothness.
* **__init__.py** – Re-exports public APIs (`run_travel_between_images_task`, `run_different_pose_task`) and common utilities for convenient importing.

## Wan2GP/

**Git Submodule** pointing to the upstream [deepbeepmeep/Wan2GP](https://github.com/deepbeepmeep/Wan2GP) repository. This contains the video-generation engine (`wgp.py`) together with model checkpoints, inference helpers, preprocessing code, and assorted assets.

The submodule is currently pinned to commit `6706709` ("optimization for i2v with CausVid") and can be updated periodically using standard git submodule commands. Only the entry module `wgp.py` is imported directly; everything else stays encapsulated within the submodule.

## Runtime artefacts

* **tasks.db** – SQLite database created on-demand by the orchestrator/server to track queued, running, and completed tasks.
* Intermediate segment folders under `steerable_motion_output/` (or user-provided directory) – automatically cleaned unless `--debug`/`--skip_cleanup` is set.

---

This document is auto-generated to give newcomers a concise map of the codebase; update it when new modules or packages are added. 