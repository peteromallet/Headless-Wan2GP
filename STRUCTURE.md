# Headless-Wan2GP Project Structure
<!-- NOTE: This file is maintained; run pytest via tests/ only -->

## Overview

Headless-Wan2GP is a queue-based video generation system built around the Wan2GP engine. It provides a scalable, headless interface for automated video generation tasks with support for both local SQLite and cloud Supabase backends.

📋 **For detailed component interactions and responsibilities, see [HEADLESS_SYSTEM_ARCHITECTURE.md](HEADLESS_SYSTEM_ARCHITECTURE.md)**

## Architecture

### Runtime data flow with wgp.py (unified architecture)

**DB → worker.py → HeadlessTaskQueue → WanOrchestrator → wgp.py → Files**

**NEW: Direct Queue Integration** eliminates triple-queue inefficiencies:

1. **worker.py** polls the tasks table (SQLite/Supabase), claims a row
2. **For generation tasks**: worker.py directly submits to HeadlessTaskQueue (bypassing legacy handlers)
3. **For orchestrator tasks**: worker.py delegates to orchestrator handlers (travel_orchestrator, etc.)
4. **HeadlessTaskQueue** processes tasks with model persistence, switches models efficiently
5. **WanOrchestrator** maps parameters safely, calls `wgp.generate_video()` with exact model loading semantics
6. **wgp.py** performs generation, writes files to outputs/, updates `state["gen"]["file_list"]`
7. **Result flows back**: WanOrchestrator → HeadlessTaskQueue → worker.py → DB status update → optional Supabase upload

**Key improvements:**
- ✅ **Eliminated blocking waits** via direct queue integration in task_registry.py
- ✅ **Model persistence** across sequential tasks (especially beneficial for travel segments)
- ✅ **Single worker thread** processing (VRAM-friendly)
- ✅ **Direct queue routing** for simple generation tasks

### Component responsibilities at a glance

- **worker.py**: Polls DB, claims work, routes tasks appropriately:
  - **Generation tasks** (`vace`, `flux`, `t2v`, `i2v`, etc.) → Direct to HeadlessTaskQueue  
  - **Orchestrator tasks** (`travel_orchestrator`) → Orchestrator handlers
  - **Travel segments** → `_handle_travel_segment_via_queue` (eliminates blocking)
  - **Complex tasks** (`travel_stitch`, `dp_final_gen`) → Specialized handlers
- **HeadlessTaskQueue** (`headless_model_management.py`): Enhanced queue system with:
  - Model persistence across tasks (reduces load times)
  - Advanced parameter validation (video guides, masks, image refs)
  - VACE/CausVid/LightI2X optimization auto-detection
  - Enhanced logging and debugging support
- **WanOrchestrator** (`headless_wgp.py`): Enhanced adapter with improved VACE detection and parameter mapping
- **wgp.py** (in `Wan2GP/`): Upstream engine that performs generation and records output paths

### Architectural Improvements (New)

**Before (Triple-Queue Problem):**
```
DB → worker.py → handler.py → HeadlessTaskQueue → WanOrchestrator → wgp.py
                      ↓ BLOCKING WAIT ↓
                   (defeats queue purpose)
```

**After (Unified Architecture):**
```
DB → worker.py → HeadlessTaskQueue → WanOrchestrator → wgp.py
         ↓ Direct routing, no blocking waits
    Model stays loaded between tasks
```

**Benefits:**
1. **Eliminated Blocking Waits**: No more `task_queue.wait_for_completion()` in handlers
2. **Model Persistence**: Same model stays loaded across sequential tasks (huge performance gain)
3. **Simplified Flow**: Direct routing eliminates unnecessary handler layers
4. **VRAM Efficiency**: Single worker thread respects GPU memory constraints
5. **Better Debugging**: Centralized parameter validation and logging

### Key integration details with wgp.py

- **Exact loading pattern**: `WanOrchestrator.load_model()` mirrors `wgp.generate_video`’s load/unload sequence to avoid stale state and ensure VRAM correctness.
- **UI state compatibility**: For LoRAs, orchestrator temporarily pre-populates `state["loras"]` so `wgp` behaves as if driven by its UI, then restores the original state after generation.
- **Model-type routing**: Queue delegates to `generate_vace`, `generate_flux`, or `generate_t2v` based on `wgp`-reported base model type, mapping parameters appropriately (e.g., Flux uses `video_length` as image count).
- **Conservative param pass-through**: Orchestrator forwards only known-safe params; queue applies model defaults and sampler CFG presets when available, while letting explicit task params override.
- **Result handoff**: `wgp` writes files and updates `state.gen.file_list`; orchestrator returns the latest path to the queue, which bubbles back up to `worker.py` for DB updates and optional uploads.

### Supabase and specialized handlers

- **Supabase Edge Functions**: Task lifecycle ops (claim, complete, fetch predecessors) happen via Edge Functions when in Supabase mode, keeping RLS intact. Canonical completion function is `complete-task` (hyphen).
- **Uploads**: `worker.py` and specialized task handlers use `prepare_output_path_with_upload` and `upload_and_get_final_output_location` to save locally first, then upload to Supabase Storage with stable paths `{task_id}/{filename}`.
- **Chaining**: Orchestrators like `travel_between_images` queue sub-tasks (segments/stitch) via DB rows; after each primitive generation, `worker.py` runs chaining logic to advance the DAG.

#### 🚨 Critical Bug Fix: Phantom Task Prevention

**Issue**: The `update-task-status` Edge Function was being misused to set tasks to "In Progress" without proper worker claiming fields (`worker_id`, `claimed_at`). This created phantom tasks that counted toward concurrency limits but couldn't be found by workers, blocking the entire system.

**Root Cause**: External services or misconfigured calls were using `update-task-status` with just `{task_id, status: "In Progress"}`, leaving tasks in a claimed-but-unfindable state.

**Fix Applied**: Added validation to `update-task-status` Edge Function:
- ✅ **Prevents Misuse**: Requires `worker_id` and `claimed_at` when setting status to "In Progress"
- ✅ **Clear Error Messages**: Returns descriptive 400 error directing to proper claiming functions
- ✅ **Proper Field Handling**: Includes worker claiming fields in update payload when provided
- ✅ **Documentation**: Updated function docs to warn against phantom task creation

**Correct Usage**: Use `claim-next-task` Edge Function or `safe_update_task_status` RPC for proper task claiming, not `update-task-status`.


### **Database Backends**
- **SQLite**: Local file-based database for single-machine deployments
- **Supabase**: Cloud PostgreSQL with Edge Functions, RLS, and storage integration
- **Dual Authentication**: Service role keys (workers) vs PATs (individual users)
- **Edge Function Operations**: Atomic task claiming, completion, and dependency management



### **Storage and Upload**
- **Local-First**: Files saved locally for reliability, then uploaded to cloud storage
- **Supabase Storage**: Automatic upload to `image_uploads` bucket with public URLs
- **Collision-Free Naming**: Files organized as `{task_id}/{filename}`

# Project Structure

```
<repo-root>
├── add_task.py
├── debug.py                 # Unified task/worker/system debug CLI
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

│   └── sm_functions/
│       ├── __init__.py
│       ├── travel_between_images.py
│       ├── join_clips.py
│       ├── join_clips_orchestrator.py
├── tasks/                      # Task specifications
│   └── HEADLESS_SUPABASE_TASK.md  # Supabase implementation spec
├── supabase/
│   └── functions/
│       ├── _shared/                # Shared authentication utilities for edge functions
│       ├── claim-next-task/        # Edge Function: claims next task (service-role → any, user → own only)
│       ├── complete-task/          # Edge Function: uploads file & marks task complete
│       ├── create-task/            # Edge Function: queues task from client
│       ├── generate-upload-url/    # Edge Function: generates presigned URLs for file uploads
│       ├── get-predecessor-output/ # Edge Function: gets task dependency and its output in single call
│       ├── get-completed-segments/ # Edge Function: fetches completed travel_segment outputs for a run_id, bypassing RLS
│       ├── task-counts/            # Edge Function: returns task counts and worker statistics
│       └── update-task-status/     # Edge Function: updates task status (use claim_next_task for proper claiming)
├── scripts/
│   └── gpu_diag.sh              # Prints GPU/NVML diagnostics (helpful when nvidia-smi breaks in containers)
├── logs/               # runtime logs (git-ignored)
├── outputs/            # generated videos/images (git-ignored)
├── samples/            # example inputs for docs & tests
├── tests/              # pytest suite
├── test_outputs/       # artefacts written by tests (git-ignored)
├── Wan2GP/  ← third-party video-generation engine (keep high-level only)
└── STRUCTURE.md  (this file)
```

## Top-level scripts

* **worker.py** – Headless service that polls the `tasks` database, claims work, and executes tasks via the HeadlessTaskQueue system. Includes specialized handlers for OpenPose and RIFE interpolation tasks with automatic Supabase storage upload. Includes 5 Qwen image editing task types (qwen_image_edit, qwen_image_hires, qwen_image_style, image_inpaint, annotated_image_edit) handled by `QwenHandler`. All Qwen tasks support optional two-pass hires fix via `hires_scale` parameter. All Qwen tasks use the qwen_image_edit_20B model with automatic LoRA management via typed `LoRAConfig`. Supports both SQLite and Supabase backends via `--db-type` flag with queue-based processing architecture. Features centralized logging that batches logs with heartbeats for orchestrator integration.
* **add_task.py** – Lightweight CLI helper to queue a single new task into SQLite/Supabase. Accepts a JSON payload (or file) and inserts it into the `tasks` table.
* **generate_test_tasks.py** – Developer utility that back-fills the database with synthetic images/prompts for integration testing and local benchmarking.
* **tests/test_travel_workflow_db_edge_functions.py** – Comprehensive test script to verify Supabase Edge Functions, authentication, and database operations for the headless worker.

## Documentation

* **STRUCTURE.md** (this file) – Project structure and component overview
* **AI_AGENT_MAINTENANCE_GUIDE.md** – Guide for AI agents working on this codebase
* **WORKER_LOGGING_IMPLEMENTATION.md** – Centralized logging implementation guide for GPU workers integrating with orchestrator logging system
* **HEADLESS_SYSTEM_ARCHITECTURE.md** – Detailed component interactions and system architecture
* **agent_docs/MULTI_STRUCTURE_VIDEO.md** – Multi-structure-video support (composite guidance + standalone segment support, timeline semantics)
* **agent_docs/uni3c/** – Uni3C integration docs:
  * `STARTING_POINT_AND_STATUS.md` – **Entry point**: dashboard, DoD, risks, phase links
  * `PHASE_1_*.md` through `PHASE_5_*.md` – Phase-by-phase implementation guides
  * `_reference/` – Appendix materials (sense check, Kijai code, param definitions)

## Supabase Upload System

All task types support automatic upload to Supabase Storage when configured:

### How it works
* **Local-first**: Files are always saved locally first for reliability
* **Conditional upload**: If Supabase is configured, files are uploaded to the `image_uploads` bucket
* **Filename sanitization**: All filenames are automatically sanitized to remove invalid storage characters (§, ®, ©, ™, control characters, etc.) before upload
* **Consistent API**: All task handlers use the same two functions:
  * `prepare_output_path_with_upload()` - Sets up local path, sanitizes filename, and provisional DB location
  * `upload_and_get_final_output_location()` - Handles upload and returns final URL/path for DB

### Task type coverage
* **Direct queue tasks**: Generated images/videos → Supabase bucket with public URLs
* **travel_stitch**: Final stitched videos → Supabase bucket
* **join_clips**: Joined video clips with VACE transitions → Supabase bucket  
* **Standard WGP tasks**: All video outputs → Supabase bucket
* **Specialized handlers**: OpenPose masks, RIFE interpolations, etc. → Supabase bucket
* **Qwen Image Edit tasks**: All 5 Qwen task types → Supabase bucket with public URLs
  * **qwen_image_edit**: Basic image editing with optional LoRAs
  * **qwen_image_style**: Style transfer between images (auto-prompt modification, Lightning/Style/Subject LoRAs)
  * **qwen_image_hires**: Dedicated two-pass hires fix generation
  * **image_inpaint**: Inpainting with green mask overlay compositing
  * **annotated_image_edit**: Scene annotation editing with specialized LoRA
  * **Hires fix**: All Qwen task types support optional two-pass hires fix when `hires_scale` param is set (latent upscale + refinement pass)

### Database behavior
* **SQLite mode**: `output_location` contains relative paths (e.g., `files/video.mp4`)
* **Supabase mode**: `output_location` contains public URLs (e.g., `https://xyz.supabase.co/storage/v1/object/public/image_uploads/task_123/video.mp4`)
* **Object naming**: Files stored as `{task_id}/{filename}` for collision-free organization

## Queue-Based Architecture

The system now uses a modern queue-based architecture for video generation:

* **headless_model_management.py** – Core queue system providing the `HeadlessTaskQueue` class with efficient model loading, memory management, and task processing. Handles model switching, quantization, and resource optimization.

* **headless_wgp.py** – Integration layer between the queue system and Wan2GP. Contains the `WanOrchestrator` class that handles parameter mapping, LoRA processing, and VACE-specific optimizations. Provides clean parameter handling to prevent conflicts.

* **worker.py** – Main worker process that polls the database, claims tasks, and routes them to appropriate handlers. All task handlers now use the queue system exclusively for video generation.

### Key Benefits
- **Model Persistence**: Models stay loaded in memory between tasks for faster processing
- **Memory Optimization**: Intelligent model loading and quantization support  
- **Parameter Safety**: Clean parameter mapping prevents conflicts and errors
- **Queue Efficiency**: Tasks are processed through an optimized queue system
- **Modern Architecture**: Robust queue-based processing system with centralized handlers


## source/ package

This is the main application package.

* **common_utils.py** – Reusable helpers (file downloads, ffmpeg helpers, MediaPipe keypoint interpolation, debug utilities, etc.). Includes generalized Supabase upload functions (`prepare_output_path_with_upload`, `upload_and_get_final_output_location`) used by all task types. Contains `extract_orchestrator_parameters()` function that provides centralized parameter extraction from `orchestrator_details` across all task types.
* **db_operations.py** – Handles all database interactions for both SQLite and Supabase. Includes Supabase client initialization, Edge Function integration, and automatic backend selection based on `DB_TYPE`.
* **specialized_handlers.py** – Contains handlers for specific, non-standard tasks like OpenPose generation and RIFE interpolation. Uses Supabase-compatible upload functions for all outputs.
* **video_utils.py** – Provides utilities for video manipulation like cross-fading, frame extraction, and color matching.
* **travel_segment_processor.py** – Shared processor for travel segment handling. Contains unified logic for guide video creation, mask video creation, and video_prompt_type construction used by both `travel_between_images.py` and `worker.py`.
* **lora_utils.py** – LoRA download and cleanup utilities. Contains `_download_lora_from_url()` for HuggingFace/direct URL downloads with collision-safe filenames, and `cleanup_legacy_lora_collisions()` for removing old generic LoRA files.
* **params/** – Typed parameter dataclasses (`TaskConfig`, `LoRAConfig`, `VACEConfig`, etc.) for clean parameter flow. Provides canonical representations that parse once at system boundary and convert to WGP format only at the final WGP call. `LoRAConfig` handles URL detection, deduplication, download tracking, and WGP format conversion.
  * **base.py** – `ParamGroup` ABC with precedence utilities and `flatten_params()` helper.
  * **lora.py** – `LoRAConfig` and `LoRAEntry` for typed LoRA handling. Uses entry objects (not parallel arrays) to preserve ordering.
  * **vace.py** – `VACEConfig` for video guide/mask parameters.
  * **generation.py** – `GenerationConfig` for core generation parameters.
  * **phase.py** – `PhaseConfig` that wraps existing `parse_phase_config()` function.
  * **task.py** – `TaskConfig` combining all param groups with `from_db_task()`, `from_segment_params()`, and `to_wgp_format()` methods.
* **model_handlers/** – Package for model-specific task handlers.
  * **qwen_handler.py** – Qwen-specific preprocessing and parameter transformation. Handles 5 Qwen task types: `qwen_image_edit`, `qwen_image_hires`, `image_inpaint`, `annotated_image_edit`, `qwen_image_style`. Manages resolution capping, composite image creation (green masks), system prompt selection, LoRA coordination, and two-pass hires fix configuration.


### source/sm_functions/ sub-package

Task-specific wrappers around the bulky upstream logic. These are imported by `worker.py` (and potentially by notebooks/unit tests) without dragging in the interactive Gradio UI shipped with Wan2GP. All task handlers use generalized Supabase upload functions for consistent output handling.

* **travel_between_images.py** – Implements the segment-by-segment interpolation pipeline between multiple anchor images. Builds guide videos, queues generation tasks, stitches outputs. Final stitched videos are uploaded to Supabase when configured. Includes extensive debugging system with `debug_video_analysis()` function that tracks frame counts, file sizes, and processing steps throughout the entire orchestrator → segments → stitching pipeline. Uses `TravelSegmentProcessor` for shared travel segment logic.
* **join_clips.py** – Bridges two video clips using VACE generation. Extracts context frames from boundaries, generates transition frames, and stitches with crossfade blending.
* **join_clips_orchestrator.py** – Orchestrates sequential joining of multiple clips. Contains shared core logic (`_create_join_chain_tasks`, `_check_existing_join_tasks`, `_extract_join_settings_from_payload`) used by both `join_clips_orchestrator` and `edit_video_orchestrator`. Supports optional VLM prompt enhancement (Qwen) to generate motion/style/details prompts from boundary frames.
* **edit_video_orchestrator.py** – Regenerates selected portions of a video. Takes a source video and `portions_to_regenerate` (list of frame ranges), extracts "keeper" clips from non-regenerated portions, then uses the shared join_clips infrastructure to regenerate transitions. Reuses all join_clips_orchestrator core logic for task creation.
* **magic_edit.py** – Processes images through Replicate's black-forest-labs/flux-kontext-dev-lora model for scene transformations. Supports conditional InScene LoRA usage via `in_scene` parameter (true for scene consistency, false for creative freedom). Integrates with Supabase storage for output handling.
* **__init__.py** – Re-exports public APIs and common utilities for convenient importing.

### Single-image (single-frame) outputs

Single-frame “image” tasks are handled by the unified queue flow (no separate `single_image.py` handler). When a task produces a single-frame video, the queue converts it to a `.png` via `HeadlessTaskQueue._convert_single_frame_video_to_png()` in `headless_model_management.py`.

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

Updated to upstream commit 9fa267087b2dfdba651fd173325537f031edf91d on 2025-09-12T20:39:26+00:00.

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
4. **Video generation** – Every handler now uses the **HeadlessTaskQueue** system which provides efficient model management, memory optimization, and queue-based processing through **headless_model_management.py** and **headless_wgp.py**.
5. **Post-processing** – Optional saturation / brightness / colour-match (`video_utils.py`) or upscaling tasks.
6. **DB update** – Handler stores `output_location` (relative in SQLite, absolute or URL in Supabase) and marks the row `Complete` (or `Failed`).  Dependants are now eligible to start.
7. **Cleanup** – Intermediate folders are deleted unless `debug_mode_enabled` or `skip_cleanup_enabled` flags are set in the payload.

## Quick task-to-file reference

| Task type / sub-task | Entrypoint function | File |
|----------------------|---------------------|------|
| Travel orchestrator  | `_handle_travel_orchestrator_task` | `sm_functions/travel_between_images.py` |
| Travel segment       | `_handle_travel_segment_via_queue` | `source/task_registry.py` |
| Travel stitch        | `_handle_travel_stitch_task`       | " " |
| Single image video   | Direct queue integration (wan_2_2_t2i) | `worker.py` (direct routing)   |
| Join clips           | `_handle_join_clips_task`          | `sm_functions/join_clips.py` |
| Join clips orchestrator | `_handle_join_clips_orchestrator_task` | `sm_functions/join_clips_orchestrator.py` |
| Edit video orchestrator | `_handle_edit_video_orchestrator_task` | `sm_functions/edit_video_orchestrator.py` |
| Magic edit           | `_handle_magic_edit_task`          | `sm_functions/magic_edit.py` |
| OpenPose mask video  | `handle_openpose_task`             | `specialized_handlers.py` |
| RIFE interpolation   | `handle_rife_task`                 | `specialized_handlers.py` |

All of the above now use the **HeadlessTaskQueue** system, which provides a modern, efficient bridge into Wan2GP with proper model management and queue-based processing.

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

### Typed Parameter System

The system uses typed dataclasses (`source/params/`) for clean parameter handling with a single source of truth:

* **`source/params/lora.py`** – `LoRAConfig` and `LoRAEntry` dataclasses for LoRA handling
* **`source/params/task.py`** – `TaskConfig` that orchestrates all parameter groups
* **`source/lora_utils.py`** – Download utilities (`_download_lora_from_url`) and legacy cleanup

**LoRA Flow:**
1. `TaskConfig.from_db_task()` parses all params at system boundary
2. `LoRAConfig` detects URLs (marks as `PENDING`) vs local files (marks as `LOCAL`)
3. Queue downloads PENDING LoRAs via `_download_lora_from_url()`
4. `config.to_wgp_format()` converts to WGP format, excluding any unresolved URLs

### Key Features

* **URL Detection**: Automatically identifies `http://`/`https://` in `activated_loras` and marks for download
* **Deduplication**: Same LoRA from multiple sources is deduplicated by filename
* **Phase-Config Multipliers**: Preserves `1.2;0.6;0.0` format for multi-phase generation
* **Safe Exclusion**: PENDING entries never reach WGP (prevents "missing LoRA" errors from URLs)
* **Collision-Safe Downloads**: HuggingFace LoRAs with generic names get parent folder prefix
* **WGP Compatibility**: All processing outputs standard WGP-compatible parameter formats

## Adding New Parameters

When adding a new parameter to the system, you need to update multiple locations depending on how the parameter flows through the pipeline. Here's where to add params:

### Parameter Flow Overview

```
Frontend → orchestrator_details → orchestrator handler → segment params → task_registry → generation_params → WGP
```

### 1. Orchestrator-Level Parameters

For parameters that affect how the orchestrator creates segments:

| Location | File | What to Update |
|----------|------|----------------|
| **Orchestrator handler** | `source/sm_functions/travel_between_images.py` | Read from `orchestrator_payload.get("param_name")` in `_handle_travel_orchestrator_task()` |
| **Segment payload creation** | Same file, around line 1740 | Add to `segment_payload = { "param_name": value, ... }` |

### 2. Segment-Level Parameters

For parameters that affect individual segment generation:

| Location | File | What to Update |
|----------|------|----------------|
| **Segment handler** | `source/task_registry.py` | Read from `segment_params.get("param_name")` in `_handle_travel_segment_via_queue()` |
| **Generation params** | Same file, around line 330-440 | Add to `generation_params["param_name"] = value` |

### 3. WGP Generation Parameters

For parameters that WGP needs during generation:

| Location | File | What to Update |
|----------|------|----------------|
| **Generation params dict** | `source/task_registry.py` | Add to `generation_params` dict before WGP submission |
| **Model defaults** (optional) | `Wan2GP/defaults/*.json` | Add default value for specific model configs |

### 4. Centralized Extraction (for common params)

For parameters used across multiple task types:

| Location | File | What to Update |
|----------|------|----------------|
| **extract_orchestrator_parameters()** | `source/common_utils.py` | Add to `extraction_map` dict (around line 68) |

### Example: Adding a New Feature Flag (e.g., `use_feature_x`)

1. **Frontend** sends `use_feature_x: true` in `orchestrator_details`

2. **Orchestrator** (`travel_between_images.py`):
   ```python
   use_feature_x = orchestrator_payload.get("use_feature_x", False)
   ```

3. **Segment payload** (`travel_between_images.py`, ~line 1740):
   ```python
   segment_payload = {
       ...
       "use_feature_x": use_feature_x,
   }
   ```

4. **Segment handler** (`task_registry.py`):
   ```python
   use_feature_x = segment_params.get("use_feature_x", False) or full_orchestrator_payload.get("use_feature_x", False)
   
   if use_feature_x:
       generation_params["feature_x_param"] = value
   ```

### Key Files for Parameter Changes

| File | Purpose |
|------|---------|
| `source/task_registry.py` | **PRIMARY**: Travel segment processing, param → generation_params conversion |
| `source/sm_functions/travel_between_images.py` | Orchestrator logic, segment payload creation |
| `source/common_utils.py` | Centralized param extraction (`extract_orchestrator_parameters`) |
| `source/params/*.py` | Typed param dataclasses (LoRA, VACE, Generation configs) |
| `Wan2GP/defaults/*.json` | Model-specific default values |

### Common Pitfalls

1. **Missing in task_registry.py**: Parameters set in orchestrator but not read in `_handle_travel_segment_via_queue()` won't reach WGP
2. **Wrong precedence**: Always check `segment_params` first, then `full_orchestrator_payload` as fallback
3. **Type coercion**: Use explicit `bool()` or type checks since DB values may be strings
4. **Logging**: Add `dprint_func()` calls to trace parameter flow for debugging

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
