# Complete Task Types & Output Directory Guide

## Overview

Every task type now saves ALL files (intermediate + final) to its own subdirectory under `outputs/`.

## Task Type Directory Mapping

### WGP Generation Tasks (Post-Processed by worker.py)

These tasks use WGP's generation engine. Files are initially generated to base directory, then automatically moved to task-type subdirectories by `move_wgp_output_to_task_type_dir()` in worker.py.

| Task Type | Output Directory | File Pattern | Handler |
|-----------|-----------------|--------------|---------|
| `vace` | `outputs/vace/` | `{task_id}_output.mp4` | WGP → worker.py post-process |
| `vace_21` | `outputs/vace_21/` | `{task_id}_output.mp4` | WGP → worker.py post-process |
| `vace_22` | `outputs/vace_22/` | `{task_id}_output.mp4` | WGP → worker.py post-process |
| `flux` | `outputs/flux/` | `{task_id}_output.png` | WGP → worker.py post-process |
| `t2v` | `outputs/t2v/` | `{task_id}_output.mp4` | WGP → worker.py post-process |
| `t2v_22` | `outputs/t2v_22/` | `{task_id}_output.mp4` | WGP → worker.py post-process |
| `wan_2_2_t2i` | `outputs/wan_2_2_t2i/` | `{task_id}_output.mp4` | WGP → worker.py post-process |
| `i2v` | `outputs/i2v/` | `{task_id}_output.mp4` | WGP → worker.py post-process |
| `i2v_22` | `outputs/i2v_22/` | `{task_id}_output.mp4` | WGP → worker.py post-process |
| `hunyuan` | `outputs/hunyuan/` | `{task_id}_output.mp4` | WGP → worker.py post-process |
| `ltxv` | `outputs/ltxv/` | `{task_id}_output.mp4` | WGP → worker.py post-process |
| `qwen_image_edit` | `outputs/qwen_image_edit/` | `{task_id}_output.png` | WGP → worker.py post-process |
| `qwen_image_style` | `outputs/qwen_image_style/` | `{task_id}_output.png` | WGP → worker.py post-process |
| `image_inpaint` | `outputs/image_inpaint/` | `{task_id}_output.png` | WGP → worker.py post-process |
| `annotated_image_edit` | `outputs/annotated_image_edit/` | `{task_id}_output.png` | WGP → worker.py post-process |
| `generate_video` | `outputs/generate_video/` | `{task_id}_output.mp4` | WGP → worker.py post-process |
| `inpaint_frames` | `outputs/inpaint_frames/` | `{task_id}_output.mp4` | Enqueues WGP → worker.py post-process |

### Specialized Handlers (Direct task_type Usage)

These tasks use `prepare_output_path_with_upload(..., task_type="...")` directly in their handlers.

| Task Type | Output Directory | File Pattern | Handler |
|-----------|-----------------|--------------|---------|
| `extract_frame` | `outputs/extract_frame/` | `{task_id}_frame_{index}.png` | source/specialized_handlers.py:46 |
| `rife_interpolate_images` | `outputs/rife_interpolate_images/` | `{task_id}_interpolated.mp4` | source/specialized_handlers.py:126 |
| `magic_edit` | `outputs/magic_edit/` | `{task_id}_edited.webp` | source/sm_functions/magic_edit.py:164 |
| `create_visualization` | `outputs/create_visualization/` | `{task_id}_visualization.mp4` | source/sm_functions/create_visualization.py:100 |
| `join_clips_segment` | `outputs/join_clips_segment/` | `{task_id}_joined.mp4` | source/sm_functions/join_clips.py:862 |

### Travel Tasks (Complex Multi-File Outputs)

Travel tasks save MANY files (intermediate + final) all to task-type directories.

#### travel_segment (Individual segment processing)

**Output Directory:** `outputs/travel_segment/`

**Files Created:**
- `{task_id}_seg{N}_guide_{timestamp}_{uuid}.mp4` - Guide video (source/travel_segment_processor.py:102)
- `{task_id}_seg{N}_mask_{timestamp}_{uuid}.mp4` - Mask video (source/travel_segment_processor.py:300)
- `{task_id}_seg{N}_saturated.mp4` - Saturation adjustment (source/sm_functions/travel_between_images.py:2467)
- `{task_id}_seg{N}_brightened.mp4` - Brightness adjustment (source/sm_functions/travel_between_images.py:2492)
- `{task_id}_seg{N}_colormatched.mp4` - Color matching (source/sm_functions/travel_between_images.py:2524)
- `{task_id}_seg{N}_banner.mp4` - Banner overlay (source/sm_functions/travel_between_images.py:2560)
- `{task_id}_output.mp4` - WGP-generated segment video (WGP → worker.py post-process)

**Handler:** source/sm_functions/travel_between_images.py + source/travel_segment_processor.py

#### travel_stitch (Final stitched output)

**Output Directory:** `outputs/travel_stitch/`

**Files Created:**
- `{task_id}_single_video.mp4` - When only 1 segment (source/sm_functions/travel_between_images.py:3008)
- `{task_id}_stitched_intermediate.mp4` - Multi-segment stitch (source/sm_functions/travel_between_images.py:3036)
- `{task_id}_output_{timestamp}_{uuid}.mp4` - Final output (source/sm_functions/travel_between_images.py:3495)

**Handler:** source/sm_functions/travel_between_images.py

### Orchestrators (No Direct File Output)

These coordinate sub-tasks and return messages only.

| Task Type | Output | Handler |
|-----------|--------|---------|
| `travel_orchestrator` | Messages only (coordinates travel_segment + travel_stitch) | source/sm_functions/travel_between_images.py |
| `join_clips_orchestrator` | Messages only (coordinates join_clips_segment) | source/sm_functions/join_clips_orchestrator.py |
| `edit_video_orchestrator` | Messages only (coordinates edit operations) | source/sm_functions/edit_video_orchestrator.py |

### Individual Travel Segment (Special Case)

| Task Type | Output Directory | Notes | Handler |
|-----------|-----------------|-------|---------|
| `individual_travel_segment` | `outputs/travel_segment/` | Same as travel_segment | source/sm_functions/travel_between_images.py |

## Complete Directory Structure

After running all task types, the directory structure will be:

```
outputs/
├── vace/                       # VACE generation outputs
├── vace_21/                    # VACE 2.1 outputs
├── vace_22/                    # VACE 2.2 outputs
├── flux/                       # Flux image generation
├── t2v/                        # Text-to-video outputs
├── t2v_22/                     # T2V 2.2 outputs
├── wan_2_2_t2i/               # WAN 2.2 text-to-image
├── i2v/                        # Image-to-video outputs
├── i2v_22/                     # I2V 2.2 outputs
├── hunyuan/                    # Hunyuan outputs
├── ltxv/                       # LTXV outputs
├── qwen_image_edit/           # Qwen image editing
├── qwen_image_style/          # Qwen style transfer
├── image_inpaint/             # Image inpainting
├── annotated_image_edit/      # Annotated editing
├── generate_video/            # Generic video generation
├── inpaint_frames/            # Frame inpainting
├── extract_frame/             # Frame extraction
├── rife_interpolate_images/   # RIFE interpolation
├── magic_edit/                # Magic edit outputs
├── create_visualization/      # Visualization videos
├── join_clips_segment/        # Joined clips
├── travel_segment/            # Travel segment files (many!)
└── travel_stitch/             # Travel final outputs
```

## How It Works

### For WGP Tasks (Post-Processing Approach)

1. **Generation:** WGP generates file to base directory
   ```
   outputs/{task_id}_output.mp4
   ```

2. **Detection:** Worker detects it's a WGP task type

3. **Post-Processing:** `move_wgp_output_to_task_type_dir()` runs
   ```python
   # Check if file is in base directory
   if output_file.parent == main_output_dir_base:
       # Move to task-type directory
       new_path = outputs/vace/{task_id}_output.mp4
       shutil.move(output_file, new_path)
   ```

4. **DB Update:** New path is stored in database

**Location in Code:** worker.py lines 52-128, 167-175

### For Specialized Handlers (Direct Approach)

Handlers call `prepare_output_path_with_upload()` with `task_type` parameter:

```python
final_path, initial_db_location = prepare_output_path_with_upload(
    task_id=task_id,
    filename="output.mp4",
    main_output_dir_base=main_output_dir_base,
    task_type="magic_edit",  # Creates outputs/magic_edit/
    dprint=dprint
)
```

File is created directly in the correct directory.

**Location in Code:** Various handler files (see table above)

### For Travel Tasks (Complex Multi-File)

Multiple `prepare_output_path()` calls throughout processing:

```python
# Guide video
guide_path, _ = prepare_output_path(
    task_id=task_id,
    filename=f"{task_id}_seg00_guide.mp4",
    main_output_dir_base=main_output_dir_base,
    task_type="travel_segment"  # All intermediate files here
)

# Final output
final_path, _ = prepare_output_path_with_upload(
    task_id=stitch_task_id,
    filename=f"{stitch_task_id}_output.mp4",
    main_output_dir_base=main_output_dir_base,
    task_type="travel_stitch"  # Final stitched video here
)
```

**Location in Code:**
- travel_segment_processor.py lines 102-112, 300-309
- travel_between_images.py lines 2467-2566, 3008-3041, 3495

## Temporary Files (Not Organized)

Some tasks create temporary working directories that are NOT organized by task_type:

- `outputs/join_clips/{task_id}/` - Temporary files for join_clips (deleted after completion)
- `outputs/inpaint_frames/{task_id}/` - Temporary guide/mask videos (deleted after completion)

These are intentionally NOT organized because they're deleted automatically. Only the **final outputs** are kept and organized.

## Backward Compatibility

The `task_type` parameter is **optional** in `prepare_output_path()`:

```python
# NEW: With task_type (organized)
path, _ = prepare_output_path(..., task_type="vace")  # → outputs/vace/

# OLD: Without task_type (base directory)
path, _ = prepare_output_path(...)  # → outputs/
```

This ensures existing code continues to work.

## Key Files Modified

1. **worker.py** - WGP post-processing (lines 43-44, 52-128, 167-175)
2. **source/common_utils.py** - Core infrastructure
3. **source/specialized_handlers.py** - extract_frame, rife_interpolate_images
4. **source/sm_functions/magic_edit.py** - magic_edit handler
5. **source/sm_functions/create_visualization.py** - visualization handler
6. **source/sm_functions/join_clips.py** - join_clips handler
7. **source/sm_functions/travel_between_images.py** - travel tasks (7 locations)
8. **source/travel_segment_processor.py** - guide/mask video creation

## Total Task Types: 28

- **16 WGP tasks** (post-processed)
- **5 specialized handlers** (direct task_type)
- **3 travel tasks** (multi-file complex)
- **3 orchestrators** (no files)
- **1 deprecated** (single_image)
