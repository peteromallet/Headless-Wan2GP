# WanGP Headless Processing

This document describes the headless processing feature for WanGP, enabling automated video generation by monitoring a task queue. This queue can be a local SQLite database or a centralized PostgreSQL database (e.g., managed by Supabase). Credit for the original Wan2GP repository to [deepbeepmeep](https://github.com/deepbeepmeep).

## Overview

The `headless.py` script allows users to run WanGP without the Gradio web interface. It continuously polls a task queue for video generation jobs. When a new task is found, it processes it using the `wgp.py` engine.

### Orchestrating multi-step workflows with `steerable_motion.py`

Located *outside* the `Wan2GP/` directory, `steerable_motion.py` is a small
command-line helper that builds higher-level tasks on top of the plain
`headless.py` queue.  Instead of inserting raw JSON rows by hand, you can run
a single command and let it generate all necessary subtasks automatically.

It currently provides two sub-commands:

| Sub-command | Purpose | Typical use-case |
|-------------|---------|------------------|
| `travel_between_images` | Generates a video that smoothly "travels" between a list of anchor images.  For every consecutive pair it builds a pose-interpolated guide video, queues a generation segment, then stitches all segments together into a final clip. | Timelapse-like transitions between concept art frames, comic panels, etc. |
| `different_pose` | Takes a single reference image plus a target prompt and produces a new video of that character in a *different pose*.  Under the hood it extracts OpenPose skeletons of both the source and an internally generated T2I target frame, constructs a guide video, and uses it to drive generation. | Turning a static portrait into an animated motion or gesture. |

All common flags such as `--resolution`, `--seed`, `--debug`, `--use_causvid_lora` are accepted and forwarded to every spawned sub-task.  The script also ensures the local SQLite database exists ("`tasks`" table) before queuing work, so you can run it safely on a fresh checkout.

See `python steerable_motion.py --help` for the full argument list and
examples.

**Key Features:**

*   **Dual Database Backend:**
    *   **SQLite (Default):** Easy to set up, stores tasks in a local `tasks.db` file. Ideal for single-machine use.
    *   **PostgreSQL/Supabase:** Allows for a more robust, centralized task queue when configured via a `.env` file. Suitable for multi-worker setups or when a managed database is preferred.
*   **Automated Output Handling:**
    *   **SQLite Mode:** Videos are saved locally to `your_main_output_dir/{task_output_sub_dir_or_task_id}/{task_id}.mp4`. The local file path is stored in the database.
    *   **Supabase Mode:** Videos are uploaded to a configured Supabase Storage bucket (e.g., as `{task_id}/{original_filename}.mp4`). The public URL is stored in the database.
*   **Persistent Task Queue:** Tasks are not lost if the server restarts.
*   **Configurable Polling:** Set how often the server checks for new tasks.
*   **Debug Mode:** Verbose logging for troubleshooting.
*   **Global `wgp.py` Overrides:** Configure `wgp.py` settings at server startup.

## Quick Start / Basic Setup

For a quick setup and to run the server (defaults to SQLite mode):

```bash
git clone https://github.com/deepbeepmeep/Wan2GP /workspace/Wan2GP && \
cd /workspace/Wan2GP && \
apt-get update && apt-get install -y python3.10-venv && \
python3.10 -m venv venv && \
source venv/bin/activate && \
pip install --no-cache-dir torch==2.6.0 torchvision torchaudio -f https://download.pytorch.org/whl/cu124 && \
pip install --no-cache-dir -r requirements.txt && \
python headless.py
```

**Note:** The command above starts `headless.py` with default settings (SQLite, output to `./outputs`). For advanced configuration, including Supabase/PostgreSQL mode, see the detailed steps below.

<details>
<summary>Detailed Configuration and Supabase Setup</summary>

1.  **Clone the Repository (if not done by Quick Start):**
    ```bash
    git clone https://github.com/deepbeepmeep/Wan2GP /workspace/Wan2GP # Or your fork/version
    cd /workspace/Wan2GP
    ```

2.  **Create a Virtual Environment (if not done by Quick Start):**
    ```bash
    # Ensure python3.10-venv or equivalent is installed
    # apt-get update && apt-get install -y python3.10-venv
    python3.10 -m venv venv
    source venv/bin/activate
    ```

3.  **Install PyTorch (if not done by Quick Start):**
    Ensure you install a version of PyTorch compatible with your CUDA version (if using GPU).
    ```bash
    # Example for CUDA 12.4 (adjust as needed)
    # pip install --no-cache-dir torch==2.6.0 torchvision torchaudio -f https://download.pytorch.org/whl/cu124
    ```

4.  **Install Python Dependencies (if not done by Quick Start):**
    Ensure your `requirements.txt` is up to date. It should include:
    ```txt
    # Core wgp.py dependencies (ensure these match the original requirements.txt)
    # Example: Pillow, requests, gradio (even if monkey-patched)
    # Add all other necessary packages for wgp.py to function.

    # Headless server specific dependencies:
    python-dotenv  # For .env file management    
    supabase        # For Supabase DB (via RPC) and Storage
    ```
    Install them:
    ```bash
    # pip install --no-cache-dir -r requirements.txt
    ```
    *(The Quick Start command handles these installations. This section is for reference or manual setup.)*

5.  **Environment Configuration (`.env` file - for Supabase/PostgreSQL Mode):**
    If you want to use PostgreSQL (e.g., with Supabase) as the task queue and for Supabase Storage, create a `.env` file in the root of the `Wan2GP` directory. `headless.py` will use these variables for its runtime operation:
    ```env
    DB_TYPE=sqlite
    POSTGRES_TABLE_NAME="tasks" # Desired table name for tasks, used in RPC calls

    # For Supabase interactions (DB via RPC and Storage)
    SUPABASE_URL="https://your-project-ref.supabase.co"
    SUPABASE_SERVICE_KEY="your_supabase_service_role_key" # Keep this secret!
    SUPABASE_VIDEO_BUCKET="videos" # Your Supabase storage bucket name
    ```
    *   **Note on Database Connection for SQL Function Setup:** To initially create the required SQL functions in your Supabase PostgreSQL database (see Step 6), you will need your database's connection string (DSN). You can typically find this in your Supabase project settings. This DSN is **not** directly used by `headless.py` from the `.env` file at runtime when `DB_TYPE=postgres` is set along with Supabase URL/Key.
    *   If `DB_TYPE=postgres` is set but `SUPABASE_URL` or `SUPABASE_SERVICE_KEY` are missing, the script will warn and fall back to SQLite.
    *   If these variables are not set, or `DB_TYPE` is not `postgres`, the script defaults to using SQLite.

6.  **Supabase Setup (If using PostgreSQL Mode):**
    *   **SQL Functions (CRITICAL):** You MUST create specific SQL functions in your Supabase PostgreSQL database. The `headless.py` script relies on these functions to interact with the tasks table via RPC (Remote Procedure Calls). Go to your Supabase Dashboard -> SQL Editor -> "New query" and execute the following SQL function definitions one by one (you'll need your database connection details/DSN if using an external tool, or use the Supabase SQL Editor directly):

        **Function 1: `func_initialize_tasks_table`**
        ```sql
        CREATE OR REPLACE FUNCTION func_initialize_tasks_table(p_table_name TEXT)
        RETURNS VOID AS $$
        BEGIN
            EXECUTE format('
                CREATE TABLE IF NOT EXISTS %I (
                    id BIGSERIAL PRIMARY KEY,
                    task_id TEXT UNIQUE NOT NULL,
                    params JSONB NOT NULL,
                    status TEXT NOT NULL DEFAULT ''Queued'',
                    worker_id TEXT NULL,
                    output_location TEXT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );', p_table_name);

            EXECUTE format('
                CREATE INDEX IF NOT EXISTS idx_%s_status_created_at ON %I (status, created_at);
            ', p_table_name, p_table_name);

            EXECUTE format('
                CREATE UNIQUE INDEX IF NOT EXISTS idx_%s_task_id ON %I (task_id);
            ', p_table_name, p_table_name);
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
        ```

        **Function 2: `func_claim_task`**
        ```sql
        CREATE OR REPLACE FUNCTION func_claim_task(p_table_name TEXT, p_worker_id TEXT)
        RETURNS TABLE(task_id_out TEXT, params_out JSONB) AS $$
        DECLARE
            v_task_id TEXT;
            v_params JSONB;
        BEGIN
            EXECUTE format('
                WITH selected_task AS (
                    SELECT id, task_id, params
                    FROM %I  -- Use %I for table/identifier names
                    WHERE status = ''Queued''
                    ORDER BY created_at ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                ), updated_task AS (
                    UPDATE %I  -- Use %I for table/identifier names
                    SET
                        status = ''In Progress'',
                        worker_id = $1, -- Use $1 for the first parameter passed to USING
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = (SELECT st.id FROM selected_task st)
                    RETURNING task_id, params
                )
                SELECT ut.task_id, ut.params FROM updated_task ut LIMIT 1', 
                p_table_name, p_table_name
            )
            INTO v_task_id, v_params
            USING p_worker_id; -- This is for the $1 placeholder

            IF v_task_id IS NOT NULL THEN
                RETURN QUERY SELECT v_task_id, v_params;
            ELSE
                RETURN QUERY SELECT NULL::TEXT, NULL::JSONB WHERE FALSE; -- Return empty set if no task found
            END IF;
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
        ```

        **Function 3: `func_update_task_status`**
        ```sql
        CREATE OR REPLACE FUNCTION func_update_task_status(
            p_table_name TEXT,
            p_task_id TEXT,
            p_status TEXT,
            p_output_location TEXT DEFAULT NULL
        )
        RETURNS VOID AS $$
        BEGIN
            IF p_status = 'Complete' AND p_output_location IS NOT NULL THEN
                EXECUTE format('
                    UPDATE %I
                    SET status = $1, updated_at = CURRENT_TIMESTAMP, output_location = $2
                    WHERE task_id = $3;',
                    p_table_name)
                USING p_status, p_output_location, p_task_id;
            ELSE
                EXECUTE format('
                    UPDATE %I
                    SET status = $1, updated_at = CURRENT_TIMESTAMP
                    WHERE task_id = $2;',
                    p_table_name)
                USING p_status, p_task_id;
            END IF;
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
        ```
        **Ensure these SQL functions are created in your Supabase database before running `headless.py` in PostgreSQL mode.**

    *   **Database Table:** The `func_initialize_tasks_table` SQL function (called by `headless.py` on startup if in Postgres mode) will create the `tasks` table (or the name specified in `POSTGRES_TABLE_NAME`) in your PostgreSQL database if it doesn't exist. The schema includes columns like `task_id`, `params` (JSONB), `status`, `output_location`, etc.
    *   **Storage Bucket:** Create a storage bucket in your Supabase project (e.g., "videos" or as defined in `SUPABASE_VIDEO_BUCKET`). For the public URLs to work directly, ensure this bucket is marked as "Public". The script uses the service role key for uploads, which has full permissions.

7.  **Default Tasks (Optional - for SQLite Mode):**
    If using SQLite and the `tasks` table is empty upon first run, the script will look for a `default_tasks.json` file in the same directory as `headless.py`. If found, it will populate the SQLite database with tasks from this file. The structure is an array of task objects (see "Task Definition" below).
</details>

<details>
<summary>How it Works</summary>

1.  **Task Definition (in the Database):**
    Tasks are stored as rows in a database table (`tasks` by default). Each task has:
    *   `task_id` (TEXT, UNIQUE): Your unique identifier for the task.
    *   `params` (TEXT for SQLite, JSONB for PostgreSQL): A JSON string/object containing all parameters for `wgp.py` (prompt, model, resolution, frames, seed, LoRA settings, paths, etc.).
    *   `status` (TEXT): "Queued", "In Progress", "Complete", "Failed".
    *   `output_location` (TEXT): Stores the local file path (SQLite) or public Supabase URL (PostgreSQL/Supabase) of the generated video upon successful completion.
    *   Timestamps for creation and updates.

    **Example of `params` content (similar to the old `tasks.json` objects):**
    ```json
    {
      "task_id": "causvid_lora_example_001", // Also stored as a separate DB column
      "prompt": "A cyberpunk cityscape at night, rain, neon lights, cinematic, highly detailed",
      "model": "t2v",
      "resolution": "960x544",
      "frames": 81,
      "seed": 2024,
      "use_causvid_lora": true,
      "output_sub_dir": "causvid_cyberpunk_city" // Used for local output structure with SQLite
    }
    ```
    **Note:** When adding tasks to the database, the `task_id` should be provided both as a distinct column value and typically within the `params` JSON for consistency with how `process_single_task` expects it.

2.  **Polling Mechanism:**
    `headless.py` periodically queries the database for tasks with `status = 'Queued'`, ordered by creation time. The interval is configurable.

3.  **Task Processing:**
    *   When a 'Queued' task is found, its status is updated to 'In Progress'.
        *   For PostgreSQL, this uses a `SELECT ... FOR UPDATE SKIP LOCKED` mechanism to ensure atomic task claiming by workers.
    *   The script prepares the state for `wgp.py` using the `params` from the task.
    *   If `use_causvid_lora` is true in the task params, it handles LoRA download and specific settings.
    *   `wgp_mod.generate_video()` is invoked. Videos are initially generated in a temporary directory.
    *   **Output Handling:**
        *   **SQLite Mode:** The video is moved from the temp dir to a structured path: `--main-output-dir / {output_sub_dir_or_task_id} / {task_id}.mp4`.
        *   **Supabase Mode:** The video is uploaded from the temp dir to the configured Supabase bucket (e.g., into a folder named by `task_id` or directly as `{task_id}.mp4`).
    *   The temporary generation directory is cleaned up.

4.  **Task Completion:**
    *   The task's status is updated to 'Complete' or 'Failed'.
    *   If successful, the `output_location` (local path or Supabase URL) is saved to the database.
    *   Progress and status messages are printed to the console.
</details>

## Usage

To run the headless server:

```bash
# Using SQLite (default, ensure no conflicting .env variables for PostgreSQL)
python headless.py --db-file tasks.db --main-output-dir /outputs

# Using PostgreSQL/Supabase (ensure .env is configured)
# The --db-file argument is ignored if .env specifies PostgreSQL
python headless.py --main-output-dir /outputs

# Enable verbose debug logging (works for both modes)
python headless.py --debug # ... other args
```

### Client / Adding Tasks to the Queue:
You will need a separate script or application to add tasks to the database.
*   **SQLite:** Connect to the SQLite DB file and `INSERT` rows into the `tasks` table.
*   **PostgreSQL/Supabase:** Connect to your PostgreSQL database and `INSERT` rows. Ensure `task_id` is unique and `params` is a valid JSON string/object. Set `status` to "Queued" or let it default.

### Command-Line Arguments for `headless.py`:

*   **Server Settings:**
    *   `--db-file` (Used for SQLite mode only): Path to the SQLite database file. Defaults to `./tasks.db`. Ignored if PostgreSQL is configured via `.env`.
    *   `--main-output-dir`: Base directory where outputs for each task will be saved if using SQLite mode (in subdirectories named by `output_sub_dir` from task params, or `task_id`). Defaults to `./outputs`.
    *   `--poll-interval`: How often (in seconds) to check the database for new tasks. Defaults to 10 seconds.
    *   `--debug`: Enable verbose debug logging.

*   **WGP Global Config Overrides (Optional - Applied once at server start):**
    These arguments allow you to override global configurations in `wgp.py`.
    *   `--wgp-attention-mode`: Choices: "auto", "sdpa", "sage", "sage2", "flash", "xformers".
    *   `--wgp-compile`: Choices: "", "transformer".
    *   `--wgp-profile`: Integer value.
    *   `--wgp-vae-config`: Integer value.
    *   `--wgp-boost`: Integer value.
    *   `--wgp-transformer-quantization`: Choices: "int8", "bf16".
    *   `--wgp-transformer-dtype-policy`: Choices: "", "fp16", "bf16".
    *   `--wgp-text-encoder-quantization`: Choices: "int8", "bf16".
    *   `--wgp-vae-precision`: Choices: "16", "32".
    *   `--wgp-mixed-precision`: Choices: "0", "1".
    *   `--wgp-preload-policy`: Set `wgp.py`'s preload_model_policy (e.g., 'P,S' or 'P'. Avoid 'U' to keep models loaded longer).

## Advanced: Video-Guided Generation (VACE)

The `params` field for each task in the database can include `video_prompt_type`, `video_guide_path`, `video_mask_path`, and `keep_frames_video_guide` as described below to control VACE features. The pose, depth or greyscale information is always extracted **from the `video_guide_path` file**; the optional `video_mask_path` is only for in/out-painting control.

**Key JSON fields in task `params` for VACE:**
```json
{
  // ... other parameters ...
  "video_prompt_type": "PV", // String of capital letters described below
  "video_guide_path": "path/to/your/control_video.mp4", // Path to the control video, accessible by the server
  "video_mask_path": "path/to/your/mask_video.mp4",   // (Optional) Path to a mask video, accessible by the server
  "keep_frames_video_guide": "1:10 -1", // (Optional) List of whole frames to preserve from the guide
  "image_refs_paths": ["path/to/ref_image1.png"], // (Optional) For 'I' in video_prompt_type
  // ... other parameters ...
}
```

### 1.  `video_prompt_type` letters
You can concatenate several letters (order does not matter).  The most common combinations are shown in the table below:

| Letter | Meaning in the UI        | What the code does                                                                 |
| :----- | :----------------------- | :--------------------------------------------------------------------------------- |
| `V`    | Use the guide video as is | Feeds the RGB frames directly                                                      |
| `P`    | Pose guidance            | Extracts DW-Pose skeletons from the guide video and feeds those instead of RGB     |
| `D`    | Depth guidance           | Extracts MiDaS depth maps and feeds them                                           |
| `G`    | Greyscale guidance       | Converts the guide video to grey before feeding                                    |
| `C`    | Colour-transfer          | Alias for `G`, lets the model recolourise the B&W guide                            |
| `M`    | External mask video      | Expect `video_mask_path`; white = in/out-paint, black = keep                       |
| `I`    | Image references         | Supply `image_refs_paths`; those images are encoded and prepended to the latent    |
| `O`    | Original pixels on black | When present the pixels under the **black** part of the mask are copied through untouched (classic in-painting) |

Examples:
* `"PV"` – feed **P**ose extracted from the guide **V**ideo.
* `"DMV"` – **D**epth maps + mask **M** + original video **V**.
* `"VMO"` – video + mask + keep pixels under black mask.

### 2.  How the mask works
A mask frame is converted to a single channel in range [0, 1].

| Value     | Effect                                                              |
| :-------- | :------------------------------------------------------------------ |
| 0 (black) | Keep original RGB pixels (or copy the guide when `O` is active)     |
| 1 (white) | Pixels are blanked and the model is free to generate new content    |

If you do **not** supply `video_mask_path`, the pipeline internally builds an all-white mask for every frame that is *not* listed in `keep_frames_video_guide` (see next section).

### 3.  `keep_frames_video_guide`
Optional string that lists *whole frames* to keep.  Syntax:

* single index – positive (1 = first frame) or negative (-1 = last frame)
* range        – `a:b` is inclusive and uses the same indexing rules
* separate items with a space.

Examples for an 81-frame clip:

```
""             // empty ⇒ keep every frame (default)
"1:20"         // keep frames 0-19, generate the rest
"1:20 -1"      // keep frames 0-19 and the last frame
"-10:-1"       // keep the last 10 frames
```

Frames not listed are zeroed and their mask set to white, so the network repaints them completely.

If you only need per-pixel control you can omit `keep_frames_video_guide` and drive everything with the mask video alone.

### 4.  Quick recipes

| Goal                                                 | Fields to set (`params` in database task)                                                                  |
| :--------------------------------------------------- | :--------------------------------------------------------------------------------------------------------- |
| Transfer only motion from a control video            | `"video_prompt_type": "PV"`, `"video_guide_path": "path/to/control.mp4"`                                   |
| Depth-guided generation                            | `"video_prompt_type": "DV"`, `"video_guide_path": "path/to/control.mp4"`                                   |
| Classic in-painting with explicit mask             | `"video_prompt_type": "MV"`, `"video_guide_path": "path/to/guide.mp4"`, `"video_mask_path": "path/to/mask.mp4"` |
| Freeze first 20 & last frame, generate the rest    | `"video_prompt_type": "VM"`, `"keep_frames_video_guide": "1:20 -1"`, `"video_guide_path": "path/to/guide.mp4"` (mask video can be all-white or omitted if guide is sufficient) |

This section should give you enough vocabulary to combine mask videos, depth/pose guidance and frame-freezing without modifying the code.

