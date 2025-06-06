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

* **common_utils.py** – Reusable helpers (DB access, file downloads, ffmpeg helpers, MediaPipe keypoint interpolation, debug utilities, etc.)
* **travel_between_images.py** – Implements the segment-by-segment interpolation pipeline between multiple anchor images.  Builds guide videos, queues generation tasks, stitches outputs.
* **different_pose.py** – Generates a new pose for a single image using an OpenPose-driven guide video plus optional RIFE interpolation for smoothness.
* **__init__.py** – Re-exports public APIs (`run_travel_between_images_task`, `run_different_pose_task`) and common utilities for convenient importing.

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
* Intermediate segment folders under `steerable_motion_output/` (or user-provided directory) – automatically cleaned unless `--debug`/`--skip_cleanup` is set.

---

This document is auto-generated to give newcomers a concise map of the codebase; update it when new modules or packages are added. 