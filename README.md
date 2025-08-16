# Headless-Wan2GP – Minimal Quick Start

This guide shows the two supported ways to run the headless video-generation worker.

---

## 1  Run Locally (SQLite)

The worker keeps its task queue in a local `tasks.db` SQLite file and saves all videos under `./outputs`.

```bash
# 1) Grab the code and enter the folder
git clone https://github.com/peteromallet/Headless-Wan2GP.git
cd Headless-Wan2GP

# 2) Initialize the Wan2GP submodule
git submodule init
git submodule update

# 3) Create a Python ≥3.10 virtual environment (Python 3.12 recommended)
python3.12 -m venv venv
source venv/bin/activate

# 4) Install PyTorch (pick the wheel that matches your hardware)
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 5) Project requirements
pip install -r Wan2GP/requirements.txt
pip install -r requirements.txt

# 6) Create LoRA directories (required for tasks that use LoRAs)
mkdir -p loras loras_i2v

# 7) Start the worker – polls the local SQLite DB every 10 s
python headless.py --main-output-dir ./outputs
```

---

## 2  Run with Supabase (PostgreSQL + Storage)

Use Supabase when you need multiple workers or public download URLs for finished videos.

1.  Create a Supabase project and a **public** storage bucket (e.g. `videos`).
2.  In the SQL editor run the helper functions found in `SUPABASE_SETUP.md` to create the `tasks` table and RPC utilities.
3.  Add a `.env` file at the repo root:

    ```env
    DB_TYPE=supabase
    SUPABASE_URL=https://<project-ref>.supabase.co
    SUPABASE_SERVICE_KEY=<service-role-key>
    SUPABASE_VIDEO_BUCKET=videos
    POSTGRES_TABLE_NAME=tasks   # optional (defaults to "tasks")
    ```
4.  Install dependencies as in the local setup, then run:

    ```bash
    python headless.py --main-output-dir ./outputs
    ```

### Auth: service-role key vs. personal token

The worker needs a Supabase token to read/write the `tasks` table and upload the final video file.

| Token type | Who should use it | Permissions in this repo |
|------------|------------------|--------------------------|
| **Service-role key** | Self-hosted backend worker(s) you control | Full access to every project row and storage object (admin).  Recommended for private deployments.  *Never expose this key in a client app.* |
| **Personal access token (PAT)** or user JWT | Scripts/apps running on behalf of a **single Supabase user** | Access is limited to rows whose `project_id` you own.  Provide `project_id` in each API call so the Edge Functions can enforce ownership. |

If you set `SUPABASE_SERVICE_KEY` in the `.env`, the worker will authenticate as an admin.  If you instead pass a PAT via the `Authorization: Bearer <token>` header when calling the Edge Functions, the backend will validate ownership before returning data.

The worker will automatically upload finished videos to the bucket and store the public URL in the database.

---

## Adding Jobs to the Queue

Use the `add_task.py` script to add tasks to the queue:

```bash
python add_task.py --type <task_type> --params <json_params> [--dependant-on <task_id>]
```

### Parameters:

- `--type`: Task type string (required)
- `--params`: JSON string with task payload OR @path-to-json-file (required)
- `--dependant-on`: Optional task_id that this new task depends on

### Character Escaping in JSON

When using JSON strings in command line, special characters need proper escaping:

#### Apostrophes and Quotes
```bash
# ✅ Correct: Escape apostrophes with backslash
python add_task.py --type video_generation --params '{"prompt": "The soccer ball'\''s position changes", "model": "t2v"}'

# ✅ Alternative: Use double quotes for the outer string
python add_task.py --type video_generation --params "{\"prompt\": \"The soccer ball's position changes\", \"model\": \"t2v\"}"

# ✅ Best practice: Use JSON files for complex prompts
echo '{"prompt": "The soccer ball'\''s position changes dynamically", "model": "t2v"}' > task.json
python add_task.py --type video_generation --params @task.json
```

#### Common Escaping Examples
```bash
# Quotes within prompts
python add_task.py --type video_generation --params '{"prompt": "A character saying \"Hello world\"", "model": "t2v"}'

# Backslashes (need double escaping)
python add_task.py --type video_generation --params '{"prompt": "A path like C:\\\\Users\\\\folder", "model": "t2v"}'

# Newlines and special characters
python add_task.py --type video_generation --params '{"prompt": "Line 1\\nLine 2 with special chars: @#$%", "model": "t2v"}'
```

#### JSON File Method (Recommended for Complex Prompts)
For prompts with many special characters, create a JSON file:

**task_params.json:**
```json
{
  "prompt": "The dragon's breath illuminates the knight's armor, creating a \"magical\" atmosphere with various symbols: @#$%^&*()",
  "negative_prompt": "Don't show blurry or low-quality details",
  "model": "t2v",
  "resolution": "1280x720",
  "seed": 42
}
```

**Command:**
```bash
python add_task.py --type video_generation --params @task_params.json
```

### Examples:

1. **Simple task with JSON string:**

```bash
python add_task.py --type single_image --params '{"prompt": "A beautiful landscape", "model": "t2v", "resolution": "512x512", "seed": 12345}'
```

2. **Task with JSON file:**

```bash
python add_task.py --type travel_orchestrator --params @my_task_params.json
```

3. **Task with dependency:**

```bash
python add_task.py --type generate_openpose --params '{"image_path": "input.png"}' --dependant-on task_12345
```

4. **Task with apostrophes in prompt:**

```bash
python add_task.py --type video_generation --params '{"prompt": "The hero'\''s journey begins at dawn", "model": "t2v", "resolution": "1280x720"}'
```

## Task Types and Examples

### 1. Single Image Generation (`single_image`)

Generates a single image from a text prompt.

**Example payload:**
```json
{
  "prompt": "A futuristic city skyline at night",
  "model": "t2v",
  "resolution": "512x512",
  "seed": 12345,
  "negative_prompt": "blurry, low quality",
  "use_causvid_lora": true
}
```

### 2. Travel Orchestrator (`travel_orchestrator`)

Manages complex travel sequences between multiple images.

**Example payload:**
```json
{
  "project_id": "my_travel_project",
  "orchestrator_details": {
    "run_id": "travel_20250814",
    "input_image_paths_resolved": ["image1.png", "image2.png", "image3.png"],
    "parsed_resolution_wh": "512x512",
    "model_name": "vace_14B",
    "use_causvid_lora": true,
    "num_new_segments_to_generate": 2,
    "base_prompts_expanded": ["Futuristic landscape", "Alien world"],
    "negative_prompts_expanded": ["blurry", "low quality"],
    "segment_frames_expanded": [81, 81],
    "frame_overlap_expanded": [12, 12],
    "fps_helpers": 16,
    "fade_in_params_json_str": "{\"low_point\": 0.0, \"high_point\": 1.0, \"curve_type\": \"ease_in_out\", \"duration_factor\": 0.0}",
    "fade_out_params_json_str": "{\"low_point\": 0.0, \"high_point\": 1.0, \"curve_type\": \"ease_in_out\", \"duration_factor\": 0.0}",
    "seed_base": 11111,
    "main_output_dir_for_run": "./outputs",
    "debug_mode_enabled": true,
    "skip_cleanup_enabled": true
  }
}
```

### 3. Generate OpenPose (`generate_openpose`)

Generates OpenPose skeleton images from input images.

**Example payload:**
```json
{
  "image_path": "input.png",
  "output_dir": "openpose_output"
}
```

### 4. Different Perspective (`different_perspective_orchestrator`)

Generates videos from different perspectives of a single image.

**Example payload:**
```json
{
  "project_id": "perspective_project",
  "run_id": "perspective_20250814",
  "input_image_path": "input.png",
  "prompt": "Cinematic view from a different angle",
  "model_name": "vace_14B",
  "resolution": "700x400",
  "fps_helpers": 16,
  "output_video_frames": 30,
  "seed": 11111,
  "use_causvid_lora": true,
  "debug_mode": true,
  "skip_cleanup": true,
  "perspective_type": "pose"
}
```

## Video Generation Examples with Different Models

### Text-to-Video Examples

#### Wan2.1 Text-to-Video 14B (t2v):
```bash
python add_task.py --type video_generation --params '{
  "prompt": "A spaceship traveling through hyperspace",
  "model": "t2v",
  "resolution": "1280x720",
  "seed": 42,
  "video_length": 81,
  "num_inference_steps": 30
}'
```

#### Wan2.1 Text-to-Video 1.3B (t2v_1.3B):
```bash
python add_task.py --type video_generation --params '{
  "prompt": "A futuristic city with flying cars",
  "model": "t2v_1.3B",
  "resolution": "832x480",
  "seed": 123,
  "video_length": 81,
  "num_inference_steps": 30
}'
```

#### With Single LoRA:
```bash
python add_task.py --type video_generation --params '{
  "prompt": "A spaceship traveling through hyperspace",
  "model": "t2v",
  "resolution": "1280x720",
  "seed": 42,
  "video_length": 81,
  "num_inference_steps": 30,
  "use_causvid_lora": true,
  "lora_name": "your_lora_name"
}'
```

#### With Multiple LoRAs:
```bash
python add_task.py --type video_generation --params '{
  "prompt": "A cyberpunk samurai with neon lights",
  "model": "t2v",
  "resolution": "1280x720",
  "seed": 42,
  "video_length": 81,
  "num_inference_steps": 30,
  "activated_loras": ["cyberpunk_style", "samurai_character", "neon_effects"],
  "loras_multipliers": "1.2 0.8 1.0"
}'
```

### Image-to-Video Examples

#### Image-to-Video 480p (i2v):
```bash
python add_task.py --type video_generation --params '{
  "prompt": "Transform this image into a dynamic video",
  "model": "i2v",
  "resolution": "832x480",
  "seed": 101,
  "video_length": 81,
  "num_inference_steps": 30,
  "image_start": "base64_encoded_image_here"
}'
```

#### Image-to-Video 720p (i2v_720p):
```bash
python add_task.py --type video_generation --params '{
  "prompt": "Create a cinematic video from this image",
  "model": "i2v_720p",
  "resolution": "1280x720",
  "seed": 202,
  "video_length": 81,
  "num_inference_steps": 30,
  "image_start": "base64_encoded_image_here"
}'
```

### ControlNet Examples

#### Vace 14B (vace_14B):
```bash
python add_task.py --type video_generation --params '{
  "prompt": "A character in a dynamic pose",
  "model": "vace_14B",
  "resolution": "832x480",
  "seed": 456,
  "video_length": 81,
  "num_inference_steps": 30
}'
```

#### Vace 1.3B (vace_1.3B):
```bash
python add_task.py --type video_generation --params '{
  "prompt": "A person dancing in the rain",
  "model": "vace_1.3B",
  "resolution": "832x480",
  "seed": 789,
  "video_length": 81,
  "num_inference_steps": 30
}'
```

### Advanced Image-to-Video Examples

#### Fun InP 14B (fun_inp) - Supports End Frames:
```bash
python add_task.py --type video_generation --params '{
  "prompt": "Smooth transition between start and end images",
  "model": "fun_inp",
  "resolution": "832x480",
  "seed": 303,
  "video_length": 81,
  "num_inference_steps": 30,
  "image_start": "base64_encoded_start_image",
  "image_end": "base64_encoded_end_image"
}'
```

#### Fun InP 1.3B (fun_inp_1.3B):
```bash
python add_task.py --type video_generation --params '{
  "prompt": "Creative morphing between two images",
  "model": "fun_inp_1.3B",
  "resolution": "832x480",
  "seed": 404,
  "video_length": 81,
  "num_inference_steps": 30,
  "image_start": "base64_encoded_start_image"
}'
```

---

## Complete Model Reference

Based on the Wan2GP codebase analysis, here are all available models and their recommended parameters:

### Text-to-Video Models

#### Standard Models
- **`t2v`** - Wan2.1 Text-to-Video 14B (recommended for high quality)
- **`t2v_1.3B`** - Wan2.1 Text-to-Video 1.3B (faster, lower VRAM)

#### Specialized Text-to-Video Models
- **`sky_df_1.3B`** - SkyReels2 Diffusion Forcing 1.3B (for long videos)
- **`sky_df_14B`** - SkyReels2 Diffusion Forcing 14B (for long videos, higher quality)
- **`sky_df_720p_14B`** - SkyReels2 Diffusion Forcing 720p 14B (for long 720p videos)
- **`ltxv_13B`** - LTX Video 13B (fast, up to 260 frames)
- **`ltxv_13B_distilled`** - LTX Video Distilled 13B (very fast)
- **`hunyuan`** - Hunyuan Video 720p 13B (best quality text-to-video)
- **`moviigen`** - MoviiGen 1080p 14B (cinematic quality, 21:9 ratio)

### Image-to-Video Models

#### Standard Image-to-Video
- **`i2v`** - Wan2.1 Image-to-Video 480p 14B
- **`i2v_720p`** - Wan2.1 Image-to-Video 720p 14B (higher resolution)
- **`hunyuan_i2v`** - Hunyuan Video Image-to-Video 720p 13B

#### Advanced Image-to-Video
- **`fun_inp`** - Fun InP Image-to-Video 14B (supports end frames)
- **`fun_inp_1.3B`** - Fun InP Image-to-Video 1.3B (faster)
- **`flf2v_720p`** - First Last Frame 2 Video 720p 14B (official start/end frame model)
- **`fantasy`** - Fantasy Speaking 720p 14B (with audio input support)

### ControlNet Models

#### Vace ControlNet (for pose, depth, custom control)
- **`vace_1.3B`** - Vace ControlNet 1.3B
- **`vace_14B`** - Vace ControlNet 14B (recommended)

#### Specialized Control Models
- **`phantom_1.3B`** - Phantom 1.3B (object/person transfer)
- **`phantom_14B`** - Phantom 14B (object/person transfer, higher quality)
- **`hunyuan_custom`** - Hunyuan Custom 720p 13B (person identity preservation)
- **`hunyuan_avatar`** - Hunyuan Avatar 720p 13B (audio-driven animation)

### Utility Models
- **`recam_1.3B`** - ReCamMaster 1.3B (camera movement replay)

### Resolution Guidelines by Model

| Model Type | Recommended Resolutions | Max Resolution |
|------------|------------------------|----------------|
| `t2v_1.3B`, `fun_inp_1.3B` | `832x480`, `480x832` | 848x480 equivalent |
| `t2v`, `i2v`, `fun_inp` | `832x480`, `1280x720` | No strict limit |
| `i2v_720p`, `flf2v_720p` | `1280x720`, `720x1280` | 720p optimized |
| `sky_df_720p_14B` | `1280x720`, `960x544` | 720p optimized |
| `hunyuan*` | `1280x720` | 720p optimized |
| `ltxv*` | `832x480`, `1280x720` | Flexible |
| `moviigen` | `1920x832` (21:9) | 1080p cinematic |

### Frame Length Guidelines

| Model Type | Default Frames | Max Frames | FPS |
|------------|----------------|------------|-----|
| Standard Wan models | 81 | 193 | 16 |
| `sky_df*` (Diffusion Forcing) | 97 | 737 | 24 |
| `ltxv*` | 97 | 737 | 30 |
| `hunyuan*` | 97 | 337-401 | 24-25 |
| `fantasy` | 81 | 233 | 23 |
| `recam_1.3B` | 81 | 193 (locked) | 16 |

### Example with All Common Parameters

```bash
python add_task.py --type single_image --params '{
  "prompt": "A majestic dragon flying over mountains at sunset",
  "negative_prompt": "blurry, low quality, distorted",
  "model": "t2v",
  "resolution": "1280x720",
  "video_length": 81,
  "num_inference_steps": 30,
  "guidance_scale": 5.0,
  "flow_shift": 5.0,
  "seed": 42,
  "repeat_generation": 1,
  "use_causvid_lora": false
}'
```

---

## Complete Parameter Schema

### Task Types

| Task Type | Description | Use Case |
|-----------|-------------|----------|
| `video_generation` | Generate videos from text/image prompts | Most common video generation tasks |
| `single_image` | Generate single images | Image generation only (no video_length) |
| `travel_orchestrator` | Complex multi-segment video sequences | Advanced orchestrated workflows |
| `generate_openpose` | Extract pose information from images | Preprocessing for pose-based generation |
| `different_perspective_orchestrator` | Generate videos from different angles | Perspective-based video generation |

### All Supported Parameters

Based on the [`generate_video`](Wan2GP/wgp.py:2694) function analysis:

#### Core Parameters
```json
{
  "prompt": "string - Text description of desired video content",
  "negative_prompt": "string - What to avoid in generation (optional)",
  "model": "string - Model type (see model list below)",
  "resolution": "string - WIDTHxHEIGHT format (e.g., '1280x720')",
  "video_length": "integer - Number of frames (model-dependent limits)",
  "num_inference_steps": "integer - Denoising steps (1-100, typically 20-50)",
  "seed": "integer - Random seed (-1 for random, 0-999999999)",
  "repeat_generation": "integer - Number of videos to generate (1-25)"
}
```

#### Advanced Parameters
```json
{
  "guidance_scale": "float - CFG scale (1.0-20.0, typically 5.0-7.5)",
  "flow_shift": "float - Flow shift parameter (0.0-25.0, typically 3.0-8.0)",
  "embedded_guidance_scale": "float - For Hunyuan models (1.0-20.0)",
  "audio_guidance_scale": "float - For fantasy model with audio (1.0-20.0)"
}
```

#### Image Input Parameters
```json
{
  "image_start": "string - Base64 encoded start image for i2v models",
  "image_end": "string - Base64 encoded end image (fun_inp models only)",
  "image_refs": "array - Base64 encoded reference images (phantom/hunyuan_custom)",
  "image_prompt_type": "string - 'S' (start only) or 'SE' (start+end)"
}
```

#### Video Input Parameters
```json
{
  "video_source": "string - Path to source video (diffusion_forcing/ltxv/recam)",
  "video_guide": "string - Path to control video (vace models)",
  "video_mask": "string - Path to mask video (vace inpainting)",
  "keep_frames_video_source": "string - Frames to keep from source",
  "keep_frames_video_guide": "string - Frames to keep from guide"
}
```

#### Audio Parameters
```json
{
  "audio_guide": "string - Path to audio file (fantasy/hunyuan_avatar models)"
}
```

#### LoRA Parameters
```json
{
  "use_causvid_lora": "boolean - Enable LoRA usage",
  "lora_name": "string - Specific LoRA name to use",
  "activated_loras": "array - List of LoRA names",
  "loras_multipliers": "string - LoRA strength values"
}
```

#### Advanced Control Parameters
```json
{
  "video_prompt_type": "string - Control type for vace ('I'=image, 'V'=video, 'M'=mask)",
  "model_mode": "integer - Generation mode (0=sync, 5=async for diffusion_forcing)",
  "sliding_window_size": "integer - Window size for long videos",
  "sliding_window_overlap": "integer - Frame overlap between windows",
  "tea_cache_setting": "float - Speed optimization (0=off, 1.5-2.5=faster)",
  "RIFLEx_setting": "integer - Long video support (0=auto, 1=on, 2=off)"
}
```

---

## LoRA Usage Guide

LoRAs (Low-Rank Adaptations) allow you to apply specialized styles, characters, or effects to your video generation. This guide covers everything you need to know about using LoRAs effectively.

### Quick Start

#### Single LoRA Usage
```bash
python add_task.py --type video_generation --params '{
  "prompt": "A fantasy dragon in medieval style",
  "model": "t2v",
  "lora_name": "fantasy_medieval_style"
}'
```

#### Multiple LoRAs with Custom Strengths
```bash
python add_task.py --type video_generation --params '{
  "prompt": "A cyberpunk samurai with neon lights",
  "model": "t2v",
  "activated_loras": ["cyberpunk_style", "samurai_character", "neon_effects"],
  "loras_multipliers": "1.2 0.8 1.0"
}'
```

### LoRA Parameters Reference

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `lora_name` | string | Single LoRA filename (without extension) | `"anime_style"` |
| `activated_loras` | array | Multiple LoRA filenames (without extensions) | `["style1", "char1", "effect1"]` |
| `loras_multipliers` | string | Space-separated strength values (1.0 = default) | `"1.2 0.8 1.0"` |
| `use_causvid_lora` | boolean | Enable/disable LoRA usage | `true` or `false` |

### File Naming and Location

#### ⚠️ Critical: Filename Requirements
- **Always use filenames WITHOUT the `.safetensors` extension** in parameters
- **Avoid dots in LoRA filenames** (known bug in matching system)
- Use underscores instead: `anime_style_v2` instead of `anime.style.v2`

#### Directory Structure
Place your LoRA files in the appropriate model-specific directories:

```
Wan2GP/
├── loras/              # Text-to-Video LoRAs (t2v, sky_df, etc.)
├── loras_i2v/          # Image-to-Video LoRAs (i2v, fun_inp, etc.)
├── loras_hunyuan/      # Hunyuan Video LoRAs
├── loras_hunyuan_i2v/  # Hunyuan Image-to-Video LoRAs
└── loras_ltxv/         # LTX Video LoRAs
```

#### Model-Specific LoRA Locations

| Model Family | LoRA Directory | Supported Extensions |
|--------------|----------------|---------------------|
| `t2v`, `t2v_1.3B`, `sky_df_*`, `moviigen` | `Wan2GP/loras/` | `.safetensors`, `.sft` |
| `i2v`, `i2v_720p`, `fun_inp*`, `flf2v_720p` | `Wan2GP/loras_i2v/` | `.safetensors`, `.sft` |
| `hunyuan`, `hunyuan_custom` | `Wan2GP/loras_hunyuan/` | `.safetensors`, `.sft` |
| `hunyuan_i2v`, `hunyuan_avatar` | `Wan2GP/loras_hunyuan_i2v/` | `.safetensors`, `.sft` |
| `ltxv_13B`, `ltxv_13B_distilled` | `Wan2GP/loras_ltxv/` | `.safetensors`, `.sft` |

### Usage Examples

#### Basic Single LoRA
```bash
python add_task.py --type video_generation --params '{
  "prompt": "A majestic castle in anime style",
  "model": "t2v",
  "resolution": "1280x720",
  "lora_name": "anime_architecture",
  "seed": 12345
}'
```

#### Multiple LoRAs with Different Strengths
```bash
python add_task.py --type video_generation --params '{
  "prompt": "A cyberpunk warrior with glowing weapons in neon city",
  "model": "t2v",
  "resolution": "1280x720",
  "activated_loras": ["cyberpunk_style", "warrior_character", "neon_effects", "weapon_glow"],
  "loras_multipliers": "1.3 1.0 0.8 1.1",
  "seed": 54321
}'
```

#### Image-to-Video with LoRA
```bash
python add_task.py --type video_generation --params '{
  "prompt": "Transform this portrait into anime style with movement",
  "model": "i2v_720p",
  "resolution": "1280x720",
  "image_start": "base64_encoded_image_here",
  "lora_name": "anime_portrait_style",
  "seed": 98765
}'
```

#### Advanced: Dynamic LoRA Strengths
You can vary LoRA strength over time using comma-separated values:
```bash
python add_task.py --type video_generation --params '{
  "prompt": "Gradual style transformation",
  "model": "t2v",
  "activated_loras": ["style_a", "style_b"],
  "loras_multipliers": "1.0,0.8,0.6,0.4 0.0,0.2,0.4,0.6",
  "num_inference_steps": 30
}'
```

### LoRA Strength Guidelines

| Strength Range | Effect | Use Case |
|----------------|--------|----------|
| `0.1 - 0.4` | Subtle influence | Minor style adjustments |
| `0.5 - 0.8` | Moderate effect | Balanced style application |
| `0.9 - 1.2` | Strong influence | Dominant style (recommended) |
| `1.3 - 2.0` | Very strong | Extreme stylization (may cause artifacts) |

### Troubleshooting

#### Common Issues

**LoRA Not Loading:**
- ✅ Check filename doesn't include `.safetensors` extension
- ✅ Verify file is in correct directory for your model
- ✅ Ensure filename doesn't contain dots (use underscores)
- ✅ Check file isn't corrupted: `python check_loras.py`

**Poor Quality Results:**
- Try adjusting LoRA strength (0.8-1.2 range usually works best)
- Ensure LoRA is compatible with your model type
- Check if multiple LoRAs are conflicting

**File Not Found Errors:**
```bash
# Validate all LoRA files
python check_loras.py

# Fix corrupted LoRA files
python check_loras.py --fix
```

#### Debug Commands
```bash
# List all available LoRAs for t2v models
ls -la Wan2GP/loras/

# Check LoRA file integrity
python check_loras.py --verbose

# Test LoRA loading (dry run)
python add_task.py --type video_generation --params '{"model": "t2v", "lora_name": "test_lora", "prompt": "test"}' --dry-run
```

### Best Practices

1. **Naming Convention**: Use descriptive names with underscores: `fantasy_medieval_v2`
2. **Organization**: Group related LoRAs in subdirectories within the main LoRA folder
3. **Testing**: Start with single LoRAs before combining multiple ones
4. **Strength Tuning**: Begin with 1.0 strength and adjust based on results
5. **Compatibility**: Ensure LoRAs are trained for your specific model type
6. **Backup**: Keep backups of working LoRA configurations

### Advanced Features

#### Automatic LoRA Download (Experimental)
```bash
python add_task.py --type video_generation --params '{
  "prompt": "Studio Ghibli style animation",
  "model": "t2v",
  "additional_loras": {
    "https://huggingface.co/user/ghibli-lora/resolve/main/ghibli_style.safetensors": 1.0
  }
}'
```

**Note**: This feature automatically downloads and caches LoRAs from URLs. Use with trusted sources only.

---

## Task Management Operations

### Monitoring Tasks and Queue Status

#### View All Tasks with Details (SQLite)
```bash
# View all tasks with key information
sqlite3 tasks.db "SELECT id, task_type, status, created_at, updated_at FROM tasks ORDER BY created_at DESC;"

# View detailed task information including prompts and parameters
sqlite3 tasks.db "SELECT id, task_type, status, params, output_location, created_at FROM tasks ORDER BY created_at DESC;"

# View only queued tasks
sqlite3 tasks.db "SELECT id, task_type, status, created_at FROM tasks WHERE status = 'Queued' ORDER BY created_at ASC;"

# View currently processing tasks
sqlite3 tasks.db "SELECT id, task_type, status, generation_started_at FROM tasks WHERE status = 'In Progress';"

# View completed tasks with outputs
sqlite3 tasks.db "SELECT id, task_type, output_location, generation_processed_at FROM tasks WHERE status = 'Complete' ORDER BY generation_processed_at DESC;"
```

#### Extract Specific Task Details (SQLite)
```bash
# Get full task details including prompt and LoRA information
sqlite3 tasks.db "SELECT id, task_type, params FROM tasks WHERE id = 'your-task-id';"

# Get task status and timing information
sqlite3 tasks.db "SELECT id, status, created_at, generation_started_at, generation_processed_at FROM tasks WHERE id = 'your-task-id';"

# View task parameters in a more readable format (requires jq)
sqlite3 -json tasks.db "SELECT id, task_type, params FROM tasks WHERE id = 'your-task-id';" | jq '.[0].params | fromjson'
```

#### Monitor Queue Statistics (SQLite)
```bash
# Get queue summary
sqlite3 tasks.db "SELECT status, COUNT(*) as count FROM tasks GROUP BY status;"

# Get task type distribution
sqlite3 tasks.db "SELECT task_type, COUNT(*) as count FROM tasks GROUP BY task_type ORDER BY count DESC;"

# Get recent task activity (last 24 hours)
sqlite3 tasks.db "SELECT id, task_type, status, created_at FROM tasks WHERE datetime(created_at) > datetime('now', '-1 day') ORDER BY created_at DESC;"

# Get average processing time for completed tasks
sqlite3 tasks.db "SELECT task_type, AVG((julianday(generation_processed_at) - julianday(generation_started_at)) * 24 * 60) as avg_minutes FROM tasks WHERE status = 'Complete' AND generation_started_at IS NOT NULL AND generation_processed_at IS NOT NULL GROUP BY task_type;"
```

#### Advanced Task Queries (SQLite)
```bash
# Find tasks using specific LoRAs
sqlite3 tasks.db "SELECT id, task_type, created_at FROM tasks WHERE params LIKE '%lora_name%' OR params LIKE '%activated_loras%';"

# Find tasks with specific models
sqlite3 tasks.db "SELECT id, task_type, created_at FROM tasks WHERE params LIKE '%\"model\":\"t2v\"%';"

# Find failed tasks with error details
sqlite3 tasks.db "SELECT id, task_type, output_location as error_message, created_at FROM tasks WHERE status = 'Failed' ORDER BY created_at DESC;"

# Find tasks by prompt keywords
sqlite3 tasks.db "SELECT id, task_type, created_at FROM tasks WHERE params LIKE '%spaceship%' OR params LIKE '%dragon%';"
```

#### Real-time Queue Monitoring (SQLite)
```bash
# Watch queue status (updates every 5 seconds)
watch -n 5 'sqlite3 tasks.db "SELECT status, COUNT(*) FROM tasks GROUP BY status;"'

# Monitor recent task activity
watch -n 10 'sqlite3 tasks.db "SELECT id, task_type, status, datetime(created_at, \"localtime\") as local_time FROM tasks ORDER BY created_at DESC LIMIT 10;"'

# Track processing times
watch -n 30 'sqlite3 tasks.db "SELECT id, task_type, status, ROUND((julianday(\"now\") - julianday(generation_started_at)) * 24 * 60, 1) as minutes_running FROM tasks WHERE status = \"In Progress\";"'
```

#### Supabase Mode
Use the Supabase dashboard or Edge Functions to check task status. You can also query the tasks table directly:

```sql
-- View all tasks
SELECT id, task_type, status, created_at, updated_at FROM tasks ORDER BY created_at DESC;

-- View task details with parameters
SELECT id, task_type, status, params, output_location FROM tasks WHERE id = 'your-task-id';

-- Monitor queue status
SELECT status, COUNT(*) FROM tasks GROUP BY status;
```

### Canceling Tasks and Queue Management

#### Cancel Individual Tasks

**Cancel a specific queued task (SQLite):**
```bash
# Mark a specific task as cancelled
sqlite3 tasks.db "UPDATE tasks SET status = 'Failed', output_location = 'Cancelled by user', updated_at = datetime('now') WHERE id = 'your-task-id' AND status = 'Queued';"

# Verify cancellation
sqlite3 tasks.db "SELECT id, status, output_location FROM tasks WHERE id = 'your-task-id';"
```

**Cancel a specific task (Python script):**
```python
# Create cancel_task.py
from source import db_operations as db_ops

# Cancel a specific task
task_id = "your-task-id"
db_ops.update_task_status(task_id, "Failed", "Cancelled by user")
print(f"Task {task_id} has been cancelled")
```

#### Cancel All Queued Tasks

**Remove all queued tasks (SQLite):**
```bash
# Show queued tasks before cancellation
sqlite3 tasks.db "SELECT COUNT(*) as queued_count FROM tasks WHERE status = 'Queued';"

# Cancel all queued tasks (marks as Failed instead of deleting)
sqlite3 tasks.db "UPDATE tasks SET status = 'Failed', output_location = 'Bulk cancelled by user', updated_at = datetime('now') WHERE status = 'Queued';"

# Alternative: Delete all queued tasks completely
sqlite3 tasks.db "DELETE FROM tasks WHERE status = 'Queued';"

# Verify results
sqlite3 tasks.db "SELECT status, COUNT(*) FROM tasks GROUP BY status;"
```

**Bulk cancel script (Python):**
```bash
# Create bulk_cancel.py and run it
python -c "
import sqlite3
from datetime import datetime

conn = sqlite3.connect('tasks.db')
cursor = conn.cursor()

# Count queued tasks
cursor.execute('SELECT COUNT(*) FROM tasks WHERE status = \"Queued\"')
queued_count = cursor.fetchone()[0]
print(f'Found {queued_count} queued tasks')

if queued_count > 0:
    # Mark all as cancelled
    cursor.execute('UPDATE tasks SET status = \"Failed\", output_location = \"Bulk cancelled\", updated_at = ? WHERE status = \"Queued\"', (datetime.utcnow().isoformat() + 'Z',))
    print(f'Cancelled {cursor.rowcount} tasks')
    
conn.commit()
conn.close()
"
```

#### Cancel Specific Task Types
```bash
# Cancel all video_generation tasks in queue
sqlite3 tasks.db "UPDATE tasks SET status = 'Failed', output_location = 'Cancelled - video_generation tasks' WHERE task_type = 'video_generation' AND status = 'Queued';"

# Cancel all travel_orchestrator tasks
sqlite3 tasks.db "UPDATE tasks SET status = 'Failed', output_location = 'Cancelled - travel tasks' WHERE task_type = 'travel_orchestrator' AND status = 'Queued';"

# Delete failed tasks (cleanup)
sqlite3 tasks.db "DELETE FROM tasks WHERE status = 'Failed';"
```

#### Stop Currently Running Tasks

**⚠️ Important**: The Headless-Wan2GP system does not have built-in graceful task stopping mechanisms. Currently running tasks cannot be stopped cleanly through the database.

**Manual intervention options:**

1. **Restart the worker process:**
```bash
# Stop the headless worker (Ctrl+C or kill process)
# The current task will be marked as failed on restart
# Then restart:
python headless.py --main-output-dir ./outputs
```

2. **Force mark running task as failed:**
```bash
# Find currently running tasks
sqlite3 tasks.db "SELECT id, task_type, generation_started_at FROM tasks WHERE status = 'In Progress';"

# Force mark as failed (will cause worker to skip it)
sqlite3 tasks.db "UPDATE tasks SET status = 'Failed', output_location = 'Manually stopped', updated_at = datetime('now') WHERE status = 'In Progress';"
```

3. **System-level process termination:**
```bash
# Find Python processes
ps aux | grep headless.py

# Terminate specific process (replace PID)
kill -TERM <process_id>

# Force kill if needed (may cause corruption)
kill -KILL <process_id>
```

**Note**: Stopping tasks mid-generation may leave temporary files and incomplete outputs. The system will clean up on the next restart.

#### Advanced Queue Management

**Pause queue processing:**
```bash
# Create a pause file to stop the worker from processing new tasks
touch PAUSE_PROCESSING

# Remove to resume
rm PAUSE_PROCESSING
```

**Priority task management:**
```bash
# Move specific task to front of queue (change created_at)
sqlite3 tasks.db "UPDATE tasks SET created_at = datetime('now', '-1 day') WHERE id = 'priority-task-id';"

# Delay specific task (move to back)
sqlite3 tasks.db "UPDATE tasks SET created_at = datetime('now', '+1 day') WHERE id = 'delay-task-id';"
```

### Monitoring Task Progress

#### Real-time Monitoring
```bash
# Watch task status updates
watch -n 5 'sqlite3 tasks.db "SELECT id, task_type, status, updated_at FROM tasks ORDER BY updated_at DESC LIMIT 10;"'
```

#### Check LoRA File Integrity
```bash
# Validate all LoRA files
python check_loras.py

# Fix corrupted LoRA files
python check_loras.py --fix
```

---

## Complete Model Examples

### Text-to-Video Models

#### Standard Wan2.1 Models

**t2v (14B) - High Quality Text-to-Video:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "A majestic eagle soaring over snow-capped mountains",
  "model": "t2v",
  "resolution": "1280x720",
  "video_length": 81,
  "num_inference_steps": 30,
  "guidance_scale": 5.0,
  "flow_shift": 5.0,
  "seed": 12345
}'
```

**t2v_1.3B - Faster, Lower VRAM:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "A robot walking through a neon-lit cyberpunk street",
  "model": "t2v_1.3B",
  "resolution": "832x480",
  "video_length": 81,
  "num_inference_steps": 25,
  "guidance_scale": 5.0,
  "flow_shift": 5.0,
  "seed": 54321
}'
```

#### Specialized Text-to-Video Models

**sky_df_14B - Long Video Generation:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "Time-lapse of clouds forming and dispersing over a landscape",
  "model": "sky_df_14B",
  "resolution": "960x544",
  "video_length": 200,
  "num_inference_steps": 30,
  "guidance_scale": 6.0,
  "flow_shift": 8.0,
  "sliding_window_size": 97,
  "seed": 11111
}'
```

**sky_df_720p_14B - Long 720p Videos:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "A journey through different seasons in a forest",
  "model": "sky_df_720p_14B",
  "resolution": "1280x720",
  "video_length": 300,
  "num_inference_steps": 30,
  "guidance_scale": 6.0,
  "flow_shift": 8.0,
  "sliding_window_size": 121,
  "seed": 22222
}'
```

**ltxv_13B - Fast Long Videos:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "A detailed cinematic sequence of a spaceship landing on an alien planet with strange flora and fauna",
  "model": "ltxv_13B",
  "resolution": "1280x720",
  "video_length": 200,
  "num_inference_steps": 30,
  "sliding_window_size": 129,
  "seed": 33333
}'
```

**ltxv_13B_distilled - Very Fast:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "A magical forest with glowing mushrooms and floating particles",
  "model": "ltxv_13B_distilled",
  "resolution": "832x480",
  "video_length": 150,
  "num_inference_steps": 20,
  "seed": 44444
}'
```

**hunyuan - Best Quality Text-to-Video:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "A serene Japanese garden with koi fish swimming in a pond",
  "model": "hunyuan",
  "resolution": "1280x720",
  "video_length": 97,
  "num_inference_steps": 30,
  "guidance_scale": 7.0,
  "embedded_guidance_scale": 6.0,
  "flow_shift": 13.0,
  "seed": 55555
}'
```

**moviigen - Cinematic 21:9:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "Epic cinematic shot of a hero walking towards a sunset",
  "model": "moviigen",
  "resolution": "1920x832",
  "video_length": 81,
  "num_inference_steps": 30,
  "guidance_scale": 5.0,
  "flow_shift": 5.0,
  "seed": 66666
}'
```

### Image-to-Video Models

**i2v - Standard Image-to-Video:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "The person in the image starts walking forward",
  "model": "i2v",
  "resolution": "832x480",
  "video_length": 81,
  "num_inference_steps": 30,
  "image_start": "base64_encoded_image_here",
  "image_prompt_type": "S",
  "seed": 77777
}'
```

**i2v_720p - High Resolution Image-to-Video:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "The landscape comes alive with gentle movement",
  "model": "i2v_720p",
  "resolution": "1280x720",
  "video_length": 81,
  "num_inference_steps": 30,
  "image_start": "base64_encoded_image_here",
  "guidance_scale": 5.0,
  "flow_shift": 7.0,
  "seed": 88888
}'
```

**hunyuan_i2v - Hunyuan Image-to-Video:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "Subtle animation bringing the scene to life",
  "model": "hunyuan_i2v",
  "resolution": "1280x720",
  "video_length": 97,
  "num_inference_steps": 30,
  "image_start": "base64_encoded_image_here",
  "guidance_scale": 7.0,
  "embedded_guidance_scale": 6.0,
  "seed": 99999
}'
```

**fun_inp - Image-to-Video with End Frame:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "Smooth transformation from start to end state",
  "model": "fun_inp",
  "resolution": "832x480",
  "video_length": 81,
  "num_inference_steps": 30,
  "image_start": "base64_encoded_start_image",
  "image_end": "base64_encoded_end_image",
  "image_prompt_type": "SE",
  "seed": 111111
}'
```

**fun_inp_1.3B - Faster End Frame Support:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "Creative morphing between two different scenes",
  "model": "fun_inp_1.3B",
  "resolution": "832x480",
  "video_length": 81,
  "num_inference_steps": 25,
  "image_start": "base64_encoded_start_image",
  "image_end": "base64_encoded_end_image",
  "seed": 222222
}'
```

**flf2v_720p - Official Start/End Frame Model:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "Professional transition between keyframes",
  "model": "flf2v_720p",
  "resolution": "1280x720",
  "video_length": 81,
  "num_inference_steps": 30,
  "image_start": "base64_encoded_start_image",
  "image_end": "base64_encoded_end_image",
  "seed": 333333
}'
```

**fantasy - Audio-Driven Animation:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "Character speaking with natural lip sync",
  "model": "fantasy",
  "resolution": "1280x720",
  "video_length": 81,
  "num_inference_steps": 30,
  "image_start": "base64_encoded_character_image",
  "audio_guide": "/path/to/audio.wav",
  "audio_guidance_scale": 5.0,
  "seed": 444444
}'
```

### ControlNet Models

**vace_14B - Advanced Control:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "A dancer performing with precise movements",
  "model": "vace_14B",
  "resolution": "832x480",
  "video_length": 81,
  "num_inference_steps": 30,
  "video_prompt_type": "PV",
  "video_guide": "/path/to/pose_video.mp4",
  "image_refs": ["base64_encoded_reference_image"],
  "seed": 555555
}'
```

**vace_1.3B - Faster Control:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "Person walking with controlled movement",
  "model": "vace_1.3B",
  "resolution": "832x480",
  "video_length": 81,
  "num_inference_steps": 25,
  "video_prompt_type": "I",
  "image_refs": ["base64_encoded_reference_image"],
  "seed": 666666
}'
```

**phantom_14B - Object/Person Transfer:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "The person appears in a new magical environment",
  "model": "phantom_14B",
  "resolution": "1280x720",
  "video_length": 81,
  "num_inference_steps": 30,
  "image_refs": ["base64_encoded_person_image"],
  "guidance_scale": 7.5,
  "flow_shift": 5.0,
  "remove_background_images_ref": 1,
  "seed": 777777
}'
```

**phantom_1.3B - Faster Object Transfer:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "Object floating in a dreamy landscape",
  "model": "phantom_1.3B",
  "resolution": "832x480",
  "video_length": 81,
  "num_inference_steps": 25,
  "image_refs": ["base64_encoded_object_image"],
  "guidance_scale": 7.5,
  "flow_shift": 5.0,
  "seed": 888888
}'
```

**hunyuan_custom - Identity Preservation:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "The person is walking in a beautiful garden",
  "model": "hunyuan_custom",
  "resolution": "1280x720",
  "video_length": 97,
  "num_inference_steps": 30,
  "image_refs": ["base64_encoded_person_image"],
  "guidance_scale": 7.5,
  "flow_shift": 13.0,
  "seed": 999999
}'
```

**hunyuan_avatar - Audio-Driven Avatar:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "Person speaking naturally with audio sync",
  "model": "hunyuan_avatar",
  "resolution": "1280x720",
  "video_length": 129,
  "num_inference_steps": 30,
  "image_refs": ["base64_encoded_person_image"],
  "audio_guide": "/path/to/speech.wav",
  "guidance_scale": 7.5,
  "flow_shift": 5.0,
  "tea_cache_start_step_perc": 25,
  "seed": 101010
}'
```

### Utility Models

**recam_1.3B - Camera Movement Replay:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "Recreate camera movement with new content",
  "model": "recam_1.3B",
  "resolution": "832x480",
  "video_length": 81,
  "num_inference_steps": 30,
  "video_source": "/path/to/source_video.mp4",
  "model_mode": 5,
  "seed": 121212
}'
```

### Advanced Examples with Multiple Parameters

**Long Video with LoRA:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "Epic fantasy battle scene with dragons",
  "negative_prompt": "blurry, low quality, distorted",
  "model": "sky_df_14B",
  "resolution": "960x544",
  "video_length": 400,
  "num_inference_steps": 35,
  "guidance_scale": 6.5,
  "flow_shift": 8.0,
  "sliding_window_size": 97,
  "sliding_window_overlap": 17,
  "use_causvid_lora": true,
  "lora_name": "fantasy_style",
  "tea_cache_setting": 1.5,
  "RIFLEx_setting": 1,
  "seed": 131313
}'
```

**Complex Vace Control with Inpainting:**
```bash
python add_task.py --type video_generation --params '{
  "prompt": "Character dancing in a new environment",
  "model": "vace_14B",
  "resolution": "832x480",
  "video_length": 81,
  "num_inference_steps": 30,
  "video_prompt_type": "MV",
  "video_guide": "/path/to/control_video.mp4",
  "video_mask": "/path/to/mask_video.mp4",
  "image_refs": ["base64_encoded_character"],
  "keep_frames_video_guide": "1:40",
  "guidance_scale": 6.0,
  "flow_shift": 5.0,
  "seed": 141414
}'
```