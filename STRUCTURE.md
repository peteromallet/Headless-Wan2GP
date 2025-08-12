# Headless-Wan2GP Project Structure

## Overview

Headless-Wan2GP is a queue-based video generation system built around the Wan2GP engine. It provides a scalable, headless interface for automated video generation tasks with support for both local SQLite and cloud Supabase backends.

## Architecture

### **Queue-Based Processing System**
- **HeadlessTaskQueue**: Central task management with persistent model state
- **Model Memory Management**: Efficient model loading/unloading with LoRA handling
- **Priority Scheduling**: Task prioritization with dependency resolution
- **Worker Management**: Multi-worker support with automatic scaling

### **Database Backends**
- **SQLite**: Local file-based database for single-machine deployments
- **Supabase**: Cloud PostgreSQL with Edge Functions, RLS, and storage integration
- **Dual Authentication**: Service role keys (workers) vs PATs (individual users)
- **Edge Function Operations**: Atomic task claiming, completion, and dependency management

### **Video Generation Pipeline**
- **VACE ControlNet**: Automatic detection and activation with preprocessing support
- **Wan 2.2 Integration**: Optimized dual-phase models with advanced CFG presets
- **Advanced LoRAs**: CausVid acceleration, LightI2X distillation, custom LoRA support
- **Frame Management**: Intelligent masking and cross-fade processing

### **Storage and Upload**
- **Local-First**: Files saved locally for reliability, then uploaded to cloud storage
- **Supabase Storage**: Automatic upload to `image_uploads` bucket with public URLs
- **Collision-Free Naming**: Files organized as `{task_id}/{filename}`

# Project Structure

```
<repo-root>
├── add_task.py
├── generate_test_tasks.py
├── worker.py
├── test_supabase_worker.py    # Test script for Supabase functionality
├── SUPABASE_SETUP.md            # Setup guide for Supabase mode
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
├── tasks/                      # Task specifications
│   └── HEADLESS_SUPABASE_TASK.md  # Supabase implementation spec
├── supabase/
│   └── functions/
│       ├── complete_task/         # Edge Function: uploads file & marks task complete
│       ├── create_task/           # Edge Function: queues task from client
│       ├── claim_next_task/       # Edge Function: claims next task (service-role → any, user → own only)
│       ├── get_predecessor_output/ # Edge Function: gets task dependency and its output in single call
│       └── get-completed-segments/ # Edge Function: fetches completed travel_segment outputs for a run_id, bypassing RLS
├── logs/               # runtime logs (git-ignored)
├── outputs/            # generated videos/images (git-ignored)
├── samples/            # example inputs for docs & tests
├── tests/              # pytest suite
├── test_outputs/       # artefacts written by tests (git-ignored)
├── Wan2GP/  ← third-party video-generation engine (keep high-level only)
└── STRUCTURE.md  (this file)
```

## Top-level scripts

* **worker.py** – Headless service that polls the `tasks` database, claims work, and executes tasks via the HeadlessTaskQueue system. Includes specialized handlers for OpenPose and RIFE interpolation tasks with automatic Supabase storage upload. Supports both SQLite and Supabase backends via `--db-type` flag with queue-based processing architecture.
* **add_task.py** – Lightweight CLI helper to queue a single new task into SQLite/Supabase. Accepts a JSON payload (or file) and inserts it into the `tasks` table.
* **generate_test_tasks.py** – Developer utility that back-fills the database with synthetic images/prompts for integration testing and local benchmarking.
* **tests/test_travel_workflow_db_edge_functions.py** – Comprehensive test script to verify Supabase Edge Functions, authentication, and database operations for the headless worker.

## Supabase Upload System

All task types support automatic upload to Supabase Storage when configured:

### How it works
* **Local-first**: Files are always saved locally first for reliability
* **Conditional upload**: If Supabase is configured, files are uploaded to the `image_uploads` bucket
* **Consistent API**: All task handlers use the same two functions:
  * `prepare_output_path_with_upload()` - Sets up local path and provisional DB location
  * `upload_and_get_final_output_location()` - Handles upload and returns final URL/path for DB

### Task type coverage
* **single_image**: Generated images → Supabase bucket with public URLs
* **travel_stitch**: Final stitched videos → Supabase bucket
* **different_perspective**: Final posed images → Supabase bucket  
* **Standard WGP tasks**: All video outputs → Supabase bucket
* **Specialized handlers**: OpenPose masks, RIFE interpolations, etc. → Supabase bucket

### Database behavior
* **SQLite mode**: `output_location` contains relative paths (e.g., `files/video.mp4`)
* **Supabase mode**: `output_location` contains public URLs (e.g., `https://xyz.supabase.co/storage/v1/object/public/image_uploads/task_123/video.mp4`)
* **Object naming**: Files stored as `{task_id}/{filename}` for collision-free organization

## source/ package

This is the main application package.

* **common_utils.py** – Reusable helpers (file downloads, ffmpeg helpers, MediaPipe keypoint interpolation, debug utilities, etc.). Includes generalized Supabase upload functions (`prepare_output_path_with_upload`, `upload_and_get_final_output_location`) used by all task types.
* **db_operations.py** – Handles all database interactions for both SQLite and Supabase. Includes Supabase client initialization, Edge Function integration, and automatic backend selection based on `DB_TYPE`.
* **specialized_handlers.py** – Contains handlers for specific, non-standard tasks like OpenPose generation and RIFE interpolation. Uses Supabase-compatible upload functions for all outputs.
* **video_utils.py** – Provides utilities for video manipulation like cross-fading, frame extraction, and color matching.
* **wgp_utils.py** – Thin wrapper around `Wan2GP.wgp` that standardises parameter names, handles LoRA quirks (e.g. CausVid, LightI2X), and exposes the single `generate_single_video` helper used by every task handler. Includes comprehensive debugging throughout the generation pipeline with detailed frame count validation.

### source/sm_functions/ sub-package

Task-specific wrappers around the bulky upstream logic. These are imported by `worker.py` (and potentially by notebooks/unit tests) without dragging in the interactive Gradio UI shipped with Wan2GP. All task handlers use generalized Supabase upload functions for consistent output handling.

* **travel_between_images.py** – Implements the segment-by-segment interpolation pipeline between multiple anchor images. Builds guide videos, queues generation tasks, stitches outputs. Final stitched videos are uploaded to Supabase when configured. Includes extensive debugging system with `debug_video_analysis()` function that tracks frame counts, file sizes, and processing steps throughout the entire orchestrator → segments → stitching pipeline.
* **different_perspective.py** – Generates a new perspective for a single image using an OpenPose or depth-driven guide video plus optional RIFE interpolation for smoothness. Final posed images are uploaded to Supabase when configured.
* **single_image.py** – Minimal handler for one-off image-to-video generation without travel or pose manipulation. Generated images are uploaded to Supabase when configured.
* **magic_edit.py** – Processes images through Replicate's black-forest-labs/flux-kontext-dev-lora model for scene transformations. Supports conditional InScene LoRA usage via `in_scene` parameter (true for scene consistency, false for creative freedom). Integrates with Supabase storage for output handling.
* **__init__.py** – Re-exports public APIs (`run_travel_between_images_task`, `run_single_image_task`, `run_different_perspective_task`) and common utilities for convenient importing.

## Additional runtime artefacts & folders

* **logs/** – Rolling log files captured by `worker.py` and unit tests. The directory is git-ignored.
* **outputs/** – Default location for final video/image results when not explicitly overridden by a task payload.
* **samples/** – A handful of small images shipped inside the repo that are referenced in the README and tests.
* **tests/** – Pytest-based regression and smoke tests covering both low-level helpers and full task workflows.
* **test_outputs/** – Artefacts produced by the test-suite; kept out of version control via `.gitignore`.
* **tasks.db** – SQLite database created on-demand by the orchestrator to track queued, running, and completed tasks (SQLite mode only).

## Database Configuration

### SQLite (Default)
* Local file-based database (`tasks.db`)
* No authentication required
* Single-machine deployments
* Files stored locally in `public/files/`

### Supabase
* Cloud PostgreSQL with Row-Level Security (RLS)
* Enable with: `--db-type supabase --supabase-url <url> --supabase-access-token <token>`
* Authentication modes:
  * **User JWT**: Processes only user-owned tasks
  * **Service-role key**: Processes all tasks (bypasses RLS)
* Automatic file upload to Supabase Storage
* Edge Function operations for RLS compliance

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

The submodule is updated periodically using standard git submodule commands. Only the entry module `wgp.py` is imported directly; everything else stays encapsulated within the submodule.

## Runtime artefacts

* **tasks.db** – SQLite database created on-demand by the orchestrator/server to track queued, running, and completed tasks (SQLite mode only).
* **public/files/** – For SQLite mode, all final video outputs are saved directly here with descriptive filenames (e.g., `{run_id}_seg00_output.mp4`, `{run_id}_final.mp4`). No nested subdirectories are created.
* **outputs/** – For non-SQLite modes or when explicitly configured, videos are saved here with task-specific subdirectories.

## End-to-End task lifecycle (1-minute read)

1. **Task injection** – A CLI, API, or test script calls `add_task.py`, which inserts a new row into the `tasks` table (SQLite or Supabase).  Payload JSON is stored in `params`, `status` is set to `Queued`.
2. **Worker pickup** – `worker.py` runs in a loop, atomically updates a `Queued` row to `In Progress`, and inspects `task_type` to choose the correct handler.
3. **Handler execution**
   * Standard tasks live in `source/sm_functions/…` (see table below).
   * Special one-offs (OpenPose, RIFE, etc.) live in `specialized_handlers.py`.
   * Handlers may queue **sub-tasks** (e.g. travel → N segments + 1 stitch) by inserting new rows with `dependant_on` set, forming a DAG.
4. **Video generation** – Every handler eventually calls `wgp_utils.generate_single_video` which wraps **Wan2GP/wgp.py** and returns a path to the rendered MP4.
5. **Post-processing** – Optional saturation / brightness / colour-match (`video_utils.py`) or upscaling tasks.
6. **DB update** – Handler stores `output_location` (relative in SQLite, absolute or URL in Supabase) and marks the row `Complete` (or `Failed`).  Dependants are now eligible to start.
7. **Cleanup** – Intermediate folders are deleted unless `debug_mode_enabled` or `skip_cleanup_enabled` flags are set in the payload.

## Quick task-to-file reference

| Task type / sub-task | Entrypoint function | File |
|----------------------|---------------------|------|
| Travel orchestrator  | `_handle_travel_orchestrator_task` | `sm_functions/travel_between_images.py` |
| Travel segment       | `_handle_travel_segment_task`      | " " |
| Travel stitch        | `_handle_travel_stitch_task`       | " " |
| Single image video   | `run_single_image_task`            | `sm_functions/single_image.py` |
| Different perspective | `run_different_perspective_task`   | `sm_functions/different_perspective.py` |
| Magic edit           | `_handle_magic_edit_task`          | `sm_functions/magic_edit.py` |
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
`status` | `Queued` → `In Progress` → `Complete`/`Failed`
`output_location` | Where the final artefact lives (string)
`updated_at` | Heartbeat & ordering
`project_id` | Links to project (required for Supabase RLS)

SQLite keeps the DB at `tasks.db`; Supabase uses the same columns with RLS policies.

## Debugging System

Comprehensive debugging system for video generation pipeline with detailed frame count tracking and validation:

### Debug Functions
* **`debug_video_analysis()`** – Analyzes any video file and reports frame count, FPS, duration, file size with clear labeling
* **Frame count validation** – Compares expected vs actual frame counts at every processing step with ⚠️ warnings for mismatches
* **Processing step tracking** – Logs success/failure of each chaining step (saturation, brightness, color matching, banner overlay)

### Debug Output Categories
* **`[FRAME_DEBUG]`** – Orchestrator frame quantization and overlap calculations
* **`[SEGMENT_DEBUG]`** – Individual segment processing parameters and frame analysis
* **`[WGP_DEBUG]`** – WGP generation parameters, results, and frame count validation
* **`[CHAIN_DEBUG]`** – Post-processing chain (saturation, brightness, color matching) with step-by-step analysis
* **`[STITCH_DEBUG]`** – Path resolution, video collection, and cross-fade analysis
* **`[CRITICAL_DEBUG]`** – Critical stitching calculations and frame count summaries
* **`[STITCH_FINAL_ANALYSIS]`** – Complete final video analysis with expected vs actual comparisons

### Key Features
* **Video analysis at every step** – Frame count, FPS, duration, file size tracked throughout pipeline
* **Path resolution debugging** – Detailed logging of SQLite-relative, absolute, and URL path handling
* **Cross-fade calculation verification** – Step-by-step analysis of overlap processing and frame arithmetic
* **Mismatch highlighting** – Clear warnings when frame counts don't match expectations
* **Processing chain validation** – Success/failure tracking for each post-processing step

This debugging system provides comprehensive visibility into the video generation pipeline to identify exactly where frame counts change and why final outputs might have unexpected lengths.

## LoRA Support

### Special LoRA Flags

* **`use_causvid_lora`** – Enables CausVid LoRA with 9 steps, guidance 1.0, flow-shift 1.0. Auto-downloads from HuggingFace if missing.
* **`use_lighti2x_lora`** – Enables LightI2X LoRA with 6 steps, guidance 1.0, flow-shift 5.0, Tea Cache disabled. Auto-downloads from HuggingFace if missing.

Both flags automatically configure optimal generation parameters and handle LoRA downloads/activation.

## Environment & config knobs (non-exhaustive)

Variable / flag | Effect
----------------|-------
`SUPABASE_URL / SUPABASE_SERVICE_KEY` | Used for Supabase connection (if not provided via CLI).
`POSTGRES_TABLE_NAME` | Table name for Supabase (default: `tasks`).
`SUPABASE_VIDEO_BUCKET` | Storage bucket name for video and image uploads.
`WAN2GP_CACHE` | Where Wan2GP caches model weights.
`--debug` | Prevents cleanup of temp folders, extra logs.
`--skip_cleanup` | Keeps all intermediate artefacts even outside debug.
`--db-type` | Choose between `sqlite` (default) or `supabase`.
`--supabase-url` | Supabase project URL (required for Supabase mode).
`--supabase-access-token` | JWT token or service-role key for authentication.

---

Keep this file **brief** – for in-depth developer docs see the `docs/` folder and inline module docstrings. 