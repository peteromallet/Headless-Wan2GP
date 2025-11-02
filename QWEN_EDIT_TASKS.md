# Qwen Image Edit Tasks Documentation

Complete documentation for all Qwen-based image editing tasks in the API orchestrator.

---

## Overview

The system supports 4 Qwen-based image editing task types:

1. **`qwen_image_edit`** - Basic image editing with optional LoRAs
2. **`qwen_image_style`** - Style transfer between images
3. **`image_inpaint`** - Inpainting with green mask overlay
4. **`annotated_image_edit`** - Scene annotation editing with specialized LoRAs

All use the Wavespeed AI `qwen-image/edit-lora` endpoint.

---

## Supported Task Types

### Supported: 
`'qwen_image_edit'`, `'qwen_image_style'`, `'wan_2_2_t2i'`, `'wan_2_2_i2v'`, `'animate_character'`, `'image-upscale'`, `'image_inpaint'`, `'annotated_image_edit'`

---

## 1. qwen_image_edit

Basic image editing task using Qwen model with optional LoRAs.

### Task Type
```
"task_type": "qwen_image_edit"
```

### Parameters

```json
{
  "image": "https://example.com/image.jpg",
  "prompt": "make the sky blue",
  "seed": -1,
  "resolution": "1024x768",
  "enable_base64_output": false,
  "enable_sync_mode": false,
  "output_format": "jpeg",
  "loras": [
    {
      "url": "https://huggingface.co/path/to/lora.safetensors",
      "strength": 1.0
    }
  ]
}
```

### Parameter Details

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | string | Yes | - | URL of the image to edit |
| `prompt` | string | Yes | - | Text description of the edit |
| `seed` | integer | No | -1 | Random seed (-1 for random) |
| `resolution` | string | No | - | Output resolution (e.g., "1024x768"). Max 1200px on widest side |
| `enable_base64_output` | boolean | No | false | Return base64 encoded output |
| `enable_sync_mode` | boolean | No | false | Synchronous processing |
| `output_format` | string | No | "jpeg" | Output format |
| `loras` | array | No | [] | Array of LoRA configurations |

### LoRA Format

Supports both formats:
```json
// Format 1: url/strength
{
  "url": "https://huggingface.co/path/to/lora.safetensors",
  "strength": 1.0
}

// Format 2: path/scale
{
  "path": "https://huggingface.co/path/to/lora.safetensors",
  "scale": 1.0
}
```

### Code Location
File: `api_orchestrator/main.py`
Lines: 145-214

---

## 2. qwen_image_style

Style transfer task that transfers style from a reference image.

### Task Type
```
"task_type": "qwen_image_style"
```

### Parameters

```json
{
  "style_reference_image": "https://example.com/style.jpg",
  "subject_reference_image": "https://example.com/subject.jpg",
  "prompt": "a beautiful landscape",
  "style_reference_strength": 1.0,
  "subject_strength": 0.8,
  "subject_description": "person",
  "in_this_scene": false,
  "seed": -1,
  "resolution": "1024x768",
  "enable_base64_output": false,
  "enable_sync_mode": false,
  "output_format": "jpeg",
  "loras": []
}
```

### Parameter Details

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `style_reference_image` | string | Yes* | - | URL of style reference image |
| `subject_reference_image` | string | Yes* | - | URL of subject reference image |
| `prompt` | string | Yes | - | Base prompt for the generation |
| `style_reference_strength` | float | No | 0.0 | Strength of style transfer (0.0-2.0) |
| `subject_strength` | float | No | 0.0 | Strength of subject transfer (0.0-2.0) |
| `subject_description` | string | No | "" | Description of subject (e.g., "person", "car") |
| `in_this_scene` | boolean | No | false | Whether to keep subject in original scene |
| `seed` | integer | No | -1 | Random seed |
| `resolution` | string | No | - | Output resolution. Max 1200px |
| `style_lora_path` | string | No | (default) | Custom style LoRA path |
| `loras` | array | No | [] | Additional LoRAs |

*Note: `style_reference_image` and `subject_reference_image` should typically be the same image.

### Automatic Prompt Modification

The system automatically modifies the prompt based on parameters:

```python
# If style_reference_strength > 0:
"In the style of this image, {prompt}"

# If subject_strength > 0 and in_this_scene = false:
"Make an image of this {subject_description}: {prompt}"

# If subject_strength > 0 and in_this_scene = true:
"Make an image of this {subject_description} in this scene: {prompt}"

# Combined example:
"In the style of this image, make an image of this person: a beautiful portrait"
```

### Default LoRAs

**Style LoRA (when style_reference_strength > 0):**
```
https://huggingface.co/peteromallet/ad_motion_loras/resolve/main/style_transfer_qwen_edit_2_000011250.safetensors
```

**Subject LoRA (when subject_strength > 0):**
```
https://huggingface.co/peteromallet/mystery_models/resolve/main/in_subject_qwen_edit_2_000006750.safetensors
```

### Code Location
File: `api_orchestrator/main.py`
Lines: 216-337

---

## 3. image_inpaint

Inpainting task with green mask overlay for targeted editing.

### Task Type
```
"task_type": "image_inpaint"
```

### Parameters

```json
{
  "image_url": "https://example.com/image.jpg",
  "mask_url": "https://example.com/mask.png",
  "prompt": "a red car",
  "seed": -1,
  "resolution": "1024x768",
  "enable_base64_output": false,
  "enable_sync_mode": false,
  "output_format": "jpeg",
  "loras": []
}
```

### Parameter Details

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image_url` (or `image`) | string | Yes | - | URL of the image to inpaint |
| `mask_url` | string | Yes | - | URL of the mask (white = areas to edit) |
| `prompt` | string | Yes | - | Description of what to paint in masked area |
| `seed` | integer | No | -1 | Random seed |
| `resolution` | string | No | - | Output resolution. Max 1200px |
| `loras` | array | No | [] | Additional LoRAs |

### How It Works

1. **Downloads image and mask**
2. **Creates composite image:**
   - Original image where mask is black (0)
   - Pure green (#00FF00) overlay where mask is white (255)
3. **Uploads composite to Supabase** with prefix `inpaint_composite_{task_id}.jpg`
4. **Calls Wavespeed API** with composite and inpainting LoRA
5. **Returns final result**

### Default Inpainting LoRA

```
https://huggingface.co/ostris/qwen_image_edit_inpainting/resolve/main/qwen_image_edit_inpainting.safetensors
```
Scale: 1.0

### Mask Requirements

- **Format:** Any image format (PNG, JPG, etc.)
- **Colors:** White (255) = areas to edit, Black (0) = areas to preserve
- **Size:** Will be automatically resized to match input image
- **Post-processing:** Binary threshold applied after resize for crisp edges

### Code Location
File: `api_orchestrator/main.py`
Lines: 660-757

---

## 4. annotated_image_edit

Scene annotation editing with specialized LoRAs for adding annotations/markers.

### Task Type
```
"task_type": "annotated_image_edit"
```

### Parameters

```json
{
  "image_url": "https://example.com/image.jpg",
  "mask_url": "https://example.com/mask.png",
  "prompt": "add red arrows pointing to the door",
  "seed": -1,
  "resolution": "1024x768",
  "enable_base64_output": false,
  "enable_sync_mode": false,
  "output_format": "jpeg",
  "loras": []
}
```

### Parameter Details

Same as `image_inpaint`, but uses different LoRAs optimized for scene annotations.

### Default Annotation LoRA

```
https://huggingface.co/peteromallet/random_junk/resolve/main/in_scene_pure_squares_flipped_450_lr_000006700.safetensors
```
Scale: 1.0

### Previous LoRAs (commented out)

```python
# Previously used annotation LoRAs:
# - in_scene_arrows_000001500.safetensors
# - in_scene_different_perspective_000019000.safetensors
```

### Use Cases

- Adding arrows/markers to images
- Scene annotations
- Visual emphasis on specific areas
- Instructional image overlays

### Code Location
File: `api_orchestrator/main.py`
Lines: 759-865

---

## Helper Functions

### create_masked_composite_image()

Creates a composite image with green overlay for inpainting tasks.

**Location:** `api_orchestrator/main.py`, Lines 41-128

**Signature:**
```python
async def create_masked_composite_image(
    client: httpx.AsyncClient,
    task_id: str,
    image_url: str,
    mask_url: str,
    filename_prefix: str = "composite"
) -> str
```

**Process:**
1. Downloads original image and mask
2. Resizes image to max 1200px (maintains aspect ratio)
3. Resizes mask to match image size
4. Applies binary threshold to mask (prevents graininess)
5. Creates pure green (#00FF00) overlay
6. Composites: green where mask is white, original where black
7. Saves as JPEG with 95% quality
8. Uploads to Supabase storage
9. Returns public URL of composite

**Returns:** Public URL of uploaded composite image

---

## How Tasks Are Triggered

Tasks are created by inserting records into the `tasks` table in Supabase.

### Task Record Structure

```json
{
  "id": "uuid-task-id",
  "task_type": "qwen_image_edit",
  "status": "Queued",
  "params": {
    "image": "https://example.com/image.jpg",
    "prompt": "make the sky blue",
    "seed": -1
  },
  "user_id": "user-uuid",
  "created_at": "2025-11-02T...",
  "run_type": "cloud"
}
```

### Task Flow

1. **Task Created** → `status = 'Queued'`
2. **Orchestrator Claims** → `status = 'In Progress'`, assigns `worker_id`
3. **Processing** → API orchestrator `process_api_task()` handles the task type
4. **Complete** → `status = 'Completed'`, `storage_path` set
5. **Failed** → `status = 'Failed'`, `error_message` set

### Processing Code

**Main entry point:** `api_orchestrator/main.py::process_api_task()`

```python
async def process_api_task(task_payload: Dict[str, Any], client: httpx.AsyncClient):
    """Process a task based on its type."""
    task_type = task_payload.get("task_type")
    params = task_payload.get("params", {})
    
    # Routes to appropriate handler based on task_type
    if task_type == "qwen_image_edit":
        # ... qwen_image_edit code
    elif task_type == "qwen_image_style":
        # ... qwen_image_style code
    elif task_type == "image_inpaint":
        # ... image_inpaint code
    elif task_type == "annotated_image_edit":
        # ... annotated_image_edit code
    # ... other task types
```

---

## Resolution Limits

All Qwen edit tasks have resolution limits:

- **Max dimension:** 1200px on widest side
- **Maintains aspect ratio** when capping
- **Example:** Input `2048x1536` → Output `1200x900`

### Code:
```python
max_dimension = 1200
if width > max_dimension or height > max_dimension:
    ratio = min(max_dimension / width, max_dimension / height)
    width = int(width * ratio)
    height = int(height * ratio)
```

---

## Common Patterns

### Adding Multiple LoRAs

```json
{
  "loras": [
    {
      "url": "https://huggingface.co/model1.safetensors",
      "strength": 1.0
    },
    {
      "url": "https://huggingface.co/model2.safetensors",
      "strength": 0.7
    }
  ]
}
```

### Using Custom Seeds

```json
{
  "seed": 42  // For reproducible results
}
```

### High Quality Output

```json
{
  "output_format": "png",  // Instead of default jpeg
  "resolution": "1200x1200"  // Max resolution
}
```

---

## API Endpoint

All Qwen tasks call the same Wavespeed API endpoint:

```
POST https://api.wavespeed.ai/v1/wavespeed-ai/qwen-image/edit-lora
```

**Authentication:** Uses `WAVESPEED_API_KEY` from environment variables

---

## Error Handling

Common errors and their causes:

| Error | Cause | Solution |
|-------|-------|----------|
| "image_url parameter is required" | Missing image URL | Add `image_url` or `image` param |
| "mask_url parameter is required" | Missing mask URL | Add `mask_url` param (inpaint only) |
| "Image processing failed" | Invalid image URL or format | Check URL is accessible, valid image |
| "Unsupported task type" | Invalid `task_type` | Use one of the supported types |
| Resolution capping messages | Input too large | Informational - automatic resize to 1200px |

---

## Testing

Use the debug tool to investigate task execution:

```bash
# Check task status
python scripts/debug.py task <task_id>

# View task logs
python scripts/debug.py task <task_id> --logs-only

# Check recent tasks
python scripts/debug.py tasks --limit 20
```

---

## Example Task Creation

```python
from supabase import create_client
import json

supabase = create_client(url, key)

task = {
    "task_type": "image_inpaint",
    "status": "Queued",
    "run_type": "cloud",
    "user_id": user_id,
    "params": json.dumps({
        "image_url": "https://example.com/photo.jpg",
        "mask_url": "https://example.com/mask.png",
        "prompt": "a red sports car",
        "seed": 42,
        "resolution": "1024x768"
    })
}

result = supabase.table('tasks').insert(task).execute()
task_id = result.data[0]['id']
```

---

## File Structure

```
api_orchestrator/
├── main.py                      # Main orchestrator logic
│   ├── create_masked_composite_image()  (lines 41-128)
│   ├── process_api_task()               (lines 130-937)
│   │   ├── qwen_image_edit              (lines 145-214)
│   │   ├── qwen_image_style             (lines 216-337)
│   │   ├── image_inpaint                (lines 660-757)
│   │   └── annotated_image_edit         (lines 759-865)
│   └── spawn_task()                     (lines 1031-1062)
├── storage_utils.py             # Upload helpers
├── wavespeed_utils.py           # Wavespeed API calls
└── video_utils.py               # Video processing
```

---

## Summary

- **4 Qwen-based task types** with different use cases
- **All use same endpoint** (`qwen-image/edit-lora`)
- **Automatic image processing** (resize, composite, upload)
- **Flexible LoRA system** supporting multiple formats
- **Resolution capping** at 1200px max
- **Green mask compositing** for inpaint/annotate tasks
- **Automatic prompt modification** for style transfer

For debugging: `python scripts/debug.py task <task_id>`

