# Headless-Wan2GP Project Structure

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
- ✅ **Eliminated blocking waits** in single_image.py and travel_between_images.py
- ✅ **Model persistence** across sequential tasks (especially beneficial for travel segments)
- ✅ **Single worker thread** processing (VRAM-friendly)
- ✅ **Direct queue routing** for simple generation tasks

### Component responsibilities at a glance

- **worker.py**: Polls DB, claims work, routes tasks appropriately:
  - **Generation tasks** (`single_image`, `vace`, `flux`, `t2v`, etc.) → Direct to HeadlessTaskQueue  
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
DB → worker.py → single_image.py → HeadlessTaskQueue → WanOrchestrator → wgp.py
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

- **Supabase Edge Functions**: Task lifecycle ops (claim, complete, fetch predecessors) happen via Edge Functions when in Supabase mode, keeping RLS intact.
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
* **Filename sanitization**: All filenames are automatically sanitized to remove invalid storage characters (§, ®, ©, ™, control characters, etc.) before upload
* **Consistent API**: All task handlers use the same two functions:
  * `prepare_output_path_with_upload()` - Sets up local path, sanitizes filename, and provisional DB location
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
- **Modern Architecture**: Replaces legacy direct WGP calls with robust queue-based processing
- **Code Deduplication**: Eliminated ~470 lines of duplicated travel segment logic through shared processor pattern

## Code Deduplication Refactoring

### Problem Addressed
The travel segment system previously had nearly identical logic duplicated between two handlers:

1. **Blocking Handler** (`travel_between_images.py`) - Synchronous processing with inline logic
2. **Queue Handler** (`worker.py`) - Asynchronous processing with copied logic

**Duplication Scope:**
- Guide video creation logic (~280 lines)
- Mask video creation logic (~140 lines) 
- Video prompt type construction (~50 lines)
- **Total: ~470 lines of duplicated code**

### Maintenance Burden
Every bug fix, feature addition, or parameter change had to be implemented twice:
- Doubled development time
- Error-prone manual synchronization  
- Risk of behavioral divergence between handlers
- Parameter precedence fixes needed in both places

### Solution: Shared Processor Pattern
Created `TravelSegmentProcessor` class that:
- Consolidates all shared logic in a single location
- Provides identical behavior for both execution paths
- Maintains full compatibility with existing parameter handling
- Supports both VACE and non-VACE model workflows

**Files Modified:**
- `source/travel_segment_processor.py` - **NEW**: Shared processor implementation
- `source/sm_functions/travel_between_images.py` - Refactored to use shared processor
- `worker.py` - Refactored to use shared processor  

**Benefits:**
- ✅ Single source of truth for travel segment logic
- ✅ Consistent behavior across both execution paths
- ✅ Easier maintenance and feature development
- ✅ Reduced codebase size and complexity
- ✅ Preserved existing functionality and parameter precedence

## Centralized Parameter Extraction System

### Problem Addressed
Previously, multiple files manually extracted parameters from `orchestrator_details` with inconsistent logic:

1. **`worker.py`** - Manual extraction for single_image tasks
2. **`travel_between_images.py`** - Different manual extraction for travel segments  
3. **`magic_edit.py`** - Yet another manual extraction implementation

**Issues:**
- Code duplication across multiple task handlers
- Inconsistent parameter precedence rules
- Manual maintenance of parameter mapping in each location
- Risk of missing parameters when adding new task types

### Solution: Centralized Extraction Function
Created `extract_orchestrator_parameters()` function in `common_utils.py` that:
- Provides single source of truth for parameter extraction logic
- Maintains consistent parameter precedence across all task types
- Supports extensible parameter mapping via centralized configuration
- Handles all parameter types (LoRAs, generation settings, task-specific params)

**Implementation:**
```python
# All task types now use the same extraction pattern:
extracted_params = extract_orchestrator_parameters(
    task_params_with_orchestrator_details, 
    task_id=task_id, 
    dprint=dprint
)
```

**Files Modified:**
- `source/common_utils.py` - **NEW**: `extract_orchestrator_parameters()` function
- `worker.py` - Refactored to use centralized extraction
- `source/sm_functions/travel_between_images.py` - Refactored to use centralized extraction (both orchestrator and segment levels)
- `source/sm_functions/magic_edit.py` - Refactored to use centralized extraction

**Benefits:**
- ✅ DRY principle: Single extraction implementation used everywhere
- ✅ Consistent parameter handling across all task types
- ✅ Easy maintenance: Add new parameters in one place
- ✅ Reduced bugs: Eliminates manual parameter extraction errors
- ✅ Automatic support: New task types get parameter extraction for free
- ✅ Fixed custom LoRA support: `additional_loras` now work consistently across single_image and travel_segment tasks

## source/ package

This is the main application package.

* **common_utils.py** – Reusable helpers (file downloads, ffmpeg helpers, MediaPipe keypoint interpolation, debug utilities, etc.). Includes generalized Supabase upload functions (`prepare_output_path_with_upload`, `upload_and_get_final_output_location`) used by all task types. **NEW**: Contains `extract_orchestrator_parameters()` function that provides centralized parameter extraction from `orchestrator_details` across all task types, eliminating code duplication and ensuring consistent parameter handling.
* **db_operations.py** – Handles all database interactions for both SQLite and Supabase. Includes Supabase client initialization, Edge Function integration, and automatic backend selection based on `DB_TYPE`.
* **specialized_handlers.py** – Contains handlers for specific, non-standard tasks like OpenPose generation and RIFE interpolation. Uses Supabase-compatible upload functions for all outputs.
* **video_utils.py** – Provides utilities for video manipulation like cross-fading, frame extraction, and color matching.
* **travel_segment_processor.py** – **NEW**: Shared processor that eliminates code duplication between travel segment handlers. Contains the unified logic for guide video creation, mask video creation, and video_prompt_type construction that was previously duplicated between `travel_between_images.py` and `worker.py`.
* **lora_utils.py** – **NEW**: Centralized LoRA processing system with `process_all_loras()` function. Consolidates all LoRA handling (detection, optimization, download, formatting) into a single pipeline that replaces 200+ lines of scattered code. Supports auto-download of LightI2X/CausVid LoRAs and multiple input formats.


### source/sm_functions/ sub-package

Task-specific wrappers around the bulky upstream logic. These are imported by `worker.py` (and potentially by notebooks/unit tests) without dragging in the interactive Gradio UI shipped with Wan2GP. All task handlers use generalized Supabase upload functions for consistent output handling.

* **travel_between_images.py** – Implements the segment-by-segment interpolation pipeline between multiple anchor images. Builds guide videos, queues generation tasks, stitches outputs. Final stitched videos are uploaded to Supabase when configured. Includes extensive debugging system with `debug_video_analysis()` function that tracks frame counts, file sizes, and processing steps throughout the entire orchestrator → segments → stitching pipeline. **Now uses shared TravelSegmentProcessor to eliminate code duplication.**
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
4. **Video generation** – Every handler now uses the **HeadlessTaskQueue** system which provides efficient model management, memory optimization, and queue-based processing through **headless_model_management.py** and **headless_wgp.py**.
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

### Centralized LoRA Processing System (NEW)

The system now features a comprehensive LoRA processing pipeline that consolidates all LoRA handling into a single entry point:

* **`source/lora_utils.py`** – Central LoRA processing module with `process_all_loras()` function
* **Consolidated Logic** – Replaces 200+ lines of scattered LoRA code across multiple files
* **Auto-Download Support** – Automatically downloads missing LightI2X and CausVid LoRAs from HuggingFace
* **Format Normalization** – Handles multiple input formats (`activated_loras`, `loras_multipliers`, `additional_loras`)
* **URL Processing** – Supports `additional_loras` dict format with automatic URL downloads

### Special LoRA Flags and Auto-Detection

* **`use_causvid_lora`** – Enables CausVid LoRA (`Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors`) with optimized parameters (9 steps, guidance 1.0, flow-shift 1.0). Step optimization only applies when no explicit steps are set. Auto-downloads from HuggingFace if missing.
* **`use_lighti2x_lora`** – Enables LightI2X LoRA (`Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors`) with optimized parameters (6 steps, guidance 1.0, flow-shift 5.0). Step optimization only applies when no explicit steps are set. Auto-downloads from HuggingFace if missing.

### Smart LoRA Detection

The system auto-detects CausVid/LightI2X LoRAs included in model JSON configs:

* **Built-in LoRAs**: When LoRAs are listed in model config's `loras` array, uses the JSON's parameter settings instead of forcing optimization values
* **Explicit flags**: When `use_causvid_lora=True` is explicitly set (but model doesn't have it built-in), applies standard optimization parameters
* **Best of both**: Allows fine-tuned parameter control in JSON configs while maintaining backward compatibility with explicit LoRA requests

### Additional LoRA Support

* **Multiple Formats**: Supports `activated_loras` (list), `loras_multipliers` (string), and `additional_loras` (dict with URLs)
* **URL Downloads**: `additional_loras` can contain `{"url": multiplier}` pairs for automatic download and processing
* **Directory Structure**: Auto-detects correct LoRA directories based on model type (WAN → `loras/`, Hunyuan → `loras_hunyuan/`, etc.)
* **WGP Compatibility**: All processing outputs standard WGP-compatible parameter formats

All LoRA processing automatically configures optimal generation parameters and handles downloads/activation through the centralized pipeline.

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