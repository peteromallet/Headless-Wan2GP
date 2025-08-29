# WanGP Headless System Architecture

## Overview

The WanGP headless system consists of three main components that work together to provide queue-based generation processing without a UI:

1. **`worker.py`** - Main task polling and orchestration service
2. **`headless_wgp.py`** - WanGP orchestrator wrapper for programmatic access
3. **`headless_model_management.py`** - Queue-based task management with model persistence

## Component Responsibilities

### worker.py - Main Task Worker Service

**Core Responsibilities:**
- Polls database (SQLite/Supabase) for queued tasks every N seconds using configurable poll interval
- Claims tasks atomically and updates status to "in_progress" 
- Routes tasks to appropriate handlers based on task_type
- Manages final task status updates (completed/failed) with output locations
- Handles database migrations and configuration setup
- Provides comprehensive CLI interface for worker configuration
- Manages worker-specific logging and debug modes
- Handles graceful shutdown with proper cleanup

**Task Processing Routes:**
- **Direct Queue Integration**: `vace`, `vace_21`, `vace_22`, `flux`, `t2v`, `t2v_22`, `i2v`, `i2v_22`, `hunyuan`, `ltxv`, `generate_video`, `wan_2_2_t2i`
- **Orchestrator Delegation**: `travel_orchestrator`, `different_perspective_orchestrator` → delegates to `source.sm_functions.*`
- **Travel Segments**: `travel_segment` → `_handle_travel_segment_via_queue()` (eliminates blocking waits)
- **Complex Workflows**: `travel_stitch`, `dp_final_gen`, `magic_edit` → specialized handlers
- **Utility Tasks**: `generate_openpose`, `rife_interpolate_images`, `extract_frame` → `source.specialized_handlers.*`

**Database Operations:**
- Supports both SQLite (with WAL change detection) and Supabase backends
- Claims oldest queued task atomically with worker_id support
- Updates task status with results, error messages, and processing times
- Handles project_id propagation for orchestrator sub-task creation
- Manages Edge Function integration for Supabase operations

**Parameter Conversion:**
- Converts DB task parameters to `GenerationTask` objects via `db_task_to_generation_task()`
- Maps task_type to appropriate model keys with Wan 2.2 defaults
- Handles parameter name variations (`steps` → `num_inference_steps`)
- Applies essential defaults (seed, negative_prompt) while preserving model config precedence
- Auto-enables Wan 2.2 acceleration LoRAs for compatible models

**Integration Points:**
- Primary: `HeadlessTaskQueue` for all generation task processing
- Secondary: `source.sm_functions.*` for orchestrator task delegation
- Tertiary: `source.specialized_handlers.*` for utility task processing
- Uses `source.common_utils`, `source.video_utils`, `source.logging_utils` throughout

### headless_wgp.py - WanGP Orchestrator Wrapper

**Core Responsibilities:**
- Provides clean programmatic interface to wgp.py's generate_video function
- Manages model loading/unloading using WGP's exact native patterns
- Handles 3-tier parameter resolution with proper precedence (user > model config > defaults)
- Supports all WGP model types (T2V, VACE, Flux, Hunyuan, LTXV)
- Bridges headless operation with WGP's UI-oriented design

**Model Management:**
- Replicates WGP's exact model loading pattern from generate_video (lines 4249-4258)
- Uses WGP's native unload_model_if_needed() for proper cleanup
- Implements smart model switching with garbage collection and memory management
- Supports VACE module testing and model family detection without requiring loaded models
- Maintains model state persistence across generations via WGP's internal state
- Dynamic model definition loading for missing model JSON files

**Parameter Resolution (`_resolve_parameters()`):**
- **Tier 1**: System defaults (resolution, video_length, steps, guidance, etc.)
- **Tier 2**: Model JSON configuration via `wgp.get_default_settings(model_type)`
- **Tier 3**: Task explicit parameters (highest priority, never overridden)
- Automatic model config loading with error handling and logging
- Preserves parameter precedence without conflicts

**LoRA Management:**
- Per-model LoRA discovery using WGP's setup_loras() with exact UI call pattern
- Pre-populates WGP UI state for LoRA compatibility in headless mode
- Handles LoRA parameter format conversion (list ↔ string)
- Maintains WGP's expected state structure for seamless integration

**Generation Features:**
- **Core Method**: `generate()` with full parameter support and model-specific routing
- **Convenience Methods**: `generate_t2v()`, `generate_vace()`, `generate_flux()`
- **VACE Support**: Full dual-encoding with video_guide, video_mask, video_guide2, video_mask2
- **Flux Support**: Batch image generation (video_length = number of images)
- **Parameter Validation**: Model-specific requirements (VACE needs video_guide)

**WGP Integration:**
- Direct calls to wgp.generate_video() with VACE-fixed wrapper
- Uses WGP's native model loading, state management, and LoRA systems
- Maintains compatibility with WGP's UI state expectations
- Preserves WGP's callback system for progress tracking
- Returns output paths from WGP's state["gen"]["file_list"]

### headless_model_management.py - Task Queue Manager

**Core Responsibilities:**
- Persistent task queue with priority support (higher priority = processed first)
- Model state persistence (keeps models loaded between tasks for efficiency)
- Multi-threaded task processing with configurable workers (default: 1 for GPU)
- Comprehensive task status tracking and monitoring with statistics
- Automatic model switching with memory management and cleanup
- Background monitoring loop for system health and maintenance

**Queue Operations:**
- Priority-based task scheduling using `queue.PriorityQueue` with (priority, timestamp, task) tuples
- Thread-safe task submission and status tracking with `threading.RLock`
- Task history management with status progression (pending → processing → completed/failed)
- Graceful shutdown with timeout handling and cleanup
- Task timeout monitoring and error recovery
- Queue persistence integration with WGP's save system

**Model Management:**
- Efficient model switching using `WanOrchestrator.load_model()` with switch detection
- Memory optimization and VRAM management via WGP's native systems
- Model capability detection (VACE, Flux, T2V) for appropriate routing
- Smart model reuse to minimize loading overhead between tasks
- Model switching statistics and performance tracking

**Task Processing (`_execute_generation()`):**
- **VACE Detection**: Uses `_model_supports_vace()` for video guide requirement validation
- **Generation Routing**: VACE → `generate_vace()`, Flux → `generate_flux()`, T2V → `generate_t2v()`
- **Parameter Conversion**: `_convert_to_wgp_task()` maps GenerationTask to WanOrchestrator format
- **File Path Validation**: Validates video_guide, video_mask, image_refs paths before processing
- **Single-Image Conversion**: Converts single-frame videos to PNG using OpenCV
- **Error Handling**: Comprehensive logging and graceful failure recovery

**Advanced Features:**
- **LoRA Processing**: Delegates to `source.lora_utils.process_all_loras` for centralized LoRA handling
- **Sampler CFG Presets**: Applies model-specific CFG settings based on sample_solver
- **Parameter Path Validation**: Ensures all file paths exist and converts to absolute paths
- **Statistics Tracking**: tasks_submitted, tasks_completed, tasks_failed, model_switches, total_generation_time
- **Debug Logging**: Extensive parameter debugging and generation flow tracing

**Integration Points:**
- **Primary**: `WanOrchestrator` for all model loading and generation operations
- **LoRA Processing**: `source.lora_utils.process_all_loras` for complete LoRA parameter handling
- **Task Format Conversion**: Bridges GenerationTask format ↔ WanOrchestrator parameter format
- **WGP Integration**: Leverages WGP's state management, queue handling, and model persistence

## Source Module Integration

### source/ Directory Usage

**`source.db_operations`** - Database abstraction layer
- **Used by**: `worker.py` for all database interactions
- **Key Functions**: 
  - `get_oldest_queued_task()` / `get_oldest_queued_task_supabase()` - Task claiming
  - `update_task_status()` / `update_task_status_supabase()` - Status updates
  - `get_task_params()` - Parameter retrieval for orchestrator payloads
  - `init_db()` - SQLite schema initialization
- **Features**: Dual backend support (SQLite with WAL + Supabase with Edge Functions)

**`source.specialized_handlers`** - Utility task processors
- **Used by**: `worker.py` for non-generation tasks
- **Handlers**: 
  - `handle_generate_openpose_task()` - OpenPose skeleton generation using dwpose
  - `handle_rife_interpolate_task()` - Frame interpolation between images
  - `handle_extract_frame_task()` - Single frame extraction from videos
- **Features**: Self-contained processing with proper error handling and output management

**`source.sm_functions.*`** - Orchestrator task handlers
- **Used by**: `worker.py` for complex multi-step workflows
- **Modules**: 
  - `travel_between_images` - Travel video orchestration, segment generation, stitching
  - `different_perspective` - Multi-angle video generation workflows
  - `single_image` - Single image generation (legacy, now routed to queue)
  - `magic_edit` - Video editing and transformation workflows
- **Features**: Handle orchestrator task creation, sub-task management, result aggregation

**`source.common_utils`** - Shared utilities
- **Used by**: All components for consistent operations
- **Key Functions**: 
  - `parse_resolution()`, `snap_resolution_to_model_grid()` - Resolution handling
  - `download_image_if_url()`, `load_pil_images()` - Image processing
  - `ensure_valid_prompt()`, `ensure_valid_negative_prompt()` - Input validation
  - `sm_get_unique_target_path()` - Output path management
  - `prepare_output_path_with_upload()` - Upload handling
- **Features**: Centralized validation, path handling, image processing

**`source.video_utils`** - Video processing utilities
- **Used by**: Travel segment processing, VACE guide preparation
- **Key Functions**: 
  - `prepare_vace_ref_for_segment()` - VACE reference video preparation
  - `create_guide_video_for_travel_segment()` - Travel guide video creation
  - `create_mask_video_from_inactive_indices()` - Mask video generation
  - Video format conversion, frame extraction, temporal processing
- **Features**: VACE-optimized video preparation, travel segment video processing

**`source.lora_utils`** - Centralized LoRA processing
- **Used by**: `headless_model_management.py` for all LoRA operations
- **Key Functions**: 
  - `process_all_loras()` - Complete LoRA processing pipeline
  - Auto-download for CausVid and LightI2X LoRAs from HuggingFace
  - Parameter format normalization (activated_loras, loras_multipliers, additional_loras)
  - URL processing for additional_loras dict format
- **Features**: Handles 200+ lines of consolidated LoRA logic, supports multiple input formats

**`source.travel_segment_processor`** - Travel segment processing
- **Used by**: `worker.py` travel segment queue integration
- **Key Components**: 
  - `TravelSegmentProcessor` - Shared segment processing logic
  - `TravelSegmentContext` - Context management for segment parameters
  - Segment video guide creation, mask generation, VACE preparation
- **Features**: Eliminates code duplication between blocking and queue-based travel processing

**`source.logging_utils`** - Structured logging
- **Used by**: All components for consistent logging
- **Loggers**: 
  - `headless_logger` - Worker operations and task processing
  - `orchestrator_logger` - WanOrchestrator operations
  - `model_logger` - Model loading and switching
  - `generation_logger` - Generation process and parameters
- **Features**: Task ID correlation, debug modes, file logging, structured output

## System Flow

### Task Submission → Completion Flow

1. **Task Creation**: External system submits task to database (SQLite/Supabase) with task_type and parameters
2. **Task Claiming**: `worker.py` polls database, atomically claims oldest queued task, updates status to "in_progress"
3. **Task Routing**: Based on task_type, `worker.py` routes to appropriate handler:
   - **Simple Generation** (`vace`, `t2v`, `flux`, etc.) → Direct to `HeadlessTaskQueue`
   - **Travel Segments** (`travel_segment`) → `_handle_travel_segment_via_queue()` → `HeadlessTaskQueue`
   - **Orchestration** (`travel_orchestrator`) → `source.sm_functions.travel_between_images`
   - **Utility Tasks** (`generate_openpose`) → `source.specialized_handlers.*`
4. **Parameter Processing**: 
   - `worker.py` converts DB parameters to `GenerationTask` via `db_task_to_generation_task()`
   - `HeadlessTaskQueue` processes LoRAs via `source.lora_utils.process_all_loras()`
   - `WanOrchestrator` resolves final parameters with model config precedence
5. **Generation**: Handler processes task using loaded models and utilities
6. **Completion**: Results stored, task status updated to "completed"/"failed", cleanup performed

### Model Management Flow

1. **Switch Detection**: `HeadlessTaskQueue._process_task()` detects when task.model != current_model
2. **Model Switch**: Calls `_switch_model()` → `WanOrchestrator.load_model()`
3. **WGP Loading**: Orchestrator replicates WGP's exact generate_video model loading pattern
4. **Persistence**: Model stays loaded in VRAM for subsequent tasks (major efficiency gain)
5. **Memory Cleanup**: WGP's native unload_model_if_needed() handles proper cleanup when switching

### Parameter Resolution Flow

1. **DB Task Parameters**: Raw JSON parameters from database task
2. **Worker Conversion**: `db_task_to_generation_task()` maps task_type to model, applies essential defaults
3. **Queue Format Conversion**: `_convert_to_wgp_task()` maps GenerationTask to WanOrchestrator format
4. **LoRA Processing**: `source.lora_utils.process_all_loras()` handles all LoRA parameter processing
5. **Path Validation**: File paths (video_guide, video_mask, image_refs) validated and converted to absolute paths
6. **Parameter Resolution**: `WanOrchestrator._resolve_parameters()` applies 3-tier precedence:
   - System defaults → Model JSON config → Task explicit parameters
7. **WGP Generation**: Final parameters passed to wgp.generate_video() with proper state setup

### Travel Segment Processing Flow (Enhanced)

1. **Orchestrator Creation**: `travel_orchestrator` task creates multiple `travel_segment` sub-tasks
2. **Segment Claiming**: `worker.py` claims individual segment tasks
3. **Queue Integration**: `_handle_travel_segment_via_queue()` eliminates blocking waits:
   - Uses `TravelSegmentProcessor` for shared segment logic
   - Submits to `HeadlessTaskQueue` for efficient model reuse
   - Waits for completion with timeout handling
4. **Post-Processing**: `_handle_travel_chaining_after_wgp()` applies saturation, color matching
5. **Stitching**: Final `travel_stitch` task combines all segments into final video

## Component Communication

```
                    ┌─ SQLite/Supabase Database ─┐
                    │   (tasks, status, params)   │
                    └─────────────┬───────────────┘
                                  │ polls/claims
                                  ▼
                     ┌─── worker.py (Main Coordinator) ───┐
                     │  • Task claiming & routing         │
                     │  • Status updates                  │
                     │  • Parameter conversion            │
                     └─┬─────────────┬─────────────────┬──┘
                       │             │                 │
            ┌──────────▼──┐     ┌────▼─────┐     ┌────▼─────────────┐
            │HeadlessTask │     │sm_func   │     │specialized_      │
            │Queue        │     │tions.*   │     │handlers.*        │
            │             │     │          │     │                  │
            └──────┬──────┘     └──────────┘     └──────────────────┘
                   │                 │                    │
                   ▼                 ▼                    ▼
            ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
            │WanOrche-    │    │source/      │    │OpenPose,    │
            │strator      │    │travel_seg   │    │RIFE, etc.   │
            │             │    │processor    │    │             │
            └──────┬──────┘    └─────────────┘    └─────────────┘
                   │
                   ▼
            ┌─────────────┐
            │wgp.py       │
            │generate_    │
            │video()      │
            └─────────────┘
```

**Critical Integration Points:**
- **worker.py**: Central coordinator, handles all task routing and database operations
- **HeadlessTaskQueue**: Model-persistent generation with efficient task processing
- **WanOrchestrator**: Clean bridge between headless system and WGP's UI-oriented architecture
- **source/**: Modular utilities providing specialized capabilities (LoRA, video, travel, logging)
- **Shared State**: All components use common logging, path handling, and parameter validation

**Data Flow Efficiency Gains:**
- **Direct Queue Routing**: Simple tasks bypass legacy handlers for 3x efficiency improvement
- **Model Persistence**: Loaded models reused across tasks (eliminates 30-60s load times)
- **Centralized LoRA Processing**: 200+ lines of LoRA logic consolidated into single pipeline
- **Parameter Precedence**: Clean 3-tier resolution prevents conflicts and ensures predictable behavior
