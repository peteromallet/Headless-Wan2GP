# WanGP Headless Processing

This document describes the headless processing feature for WanGP, enabling automated video generation by monitoring a task queue. This queue can be a local SQLite database or a centralized PostgreSQL database (e.g., managed by Supabase). Credit for the original Wan2GP repository to [deepbeepmeep](https://github.com/deepbeepmeep).

## Overview

The `headless.py` script is the core worker process that allows users to run WanGP without the Gradio web interface. It continuously polls a task queue for video generation jobs. When a new task is found, it processes it using the `wgp.py` engine or other configured handlers (like ComfyUI).

### Orchestrating multi-step workflows with `steerable_motion.py`

Located *outside* the `Wan2GP/` directory, `steerable_motion.py` is a command-line utility designed to simplify the creation of complex, multi-step video generation workflows by enqueuing a series of coordinated tasks for `headless.py` to process. Instead of manually inserting multiple intricate JSON rows into the database, you can use `steerable_motion.py` to define high-level goals.

It currently provides two main sub-commands:

| Sub-command             | Purpose                                                                                                                                                                                             | Typical use-case                                                                    |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `travel_between_images` | Generates a video that smoothly "travels" between a list of anchor images. It enqueues an orchestrator task, which `headless.py` then uses to manage individual segment generations and final stitching. | Timelapse-like transitions between concept art frames, architectural visualizations.  |
| `different_pose`        | Takes a single reference image plus a target prompt and produces a new video of that character in a *different pose*. Internally, it queues tasks for OpenPose extraction and guided video generation. | Turning a static portrait into an animated motion, gesture, or expression change. |

All common flags such as `--resolution`, `--seed`, `--debug`, `--use_causvid_lora`, `--execution_engine` (to choose between `wgp` or `comfyui` for generation steps) are accepted by `steerable_motion.py` and forwarded appropriately within the task payloads it creates. The script also ensures the local SQLite database (default: `tasks.db`) and the necessary `tasks` table exist before queuing work.

See `python steerable_motion.py --help` for the full argument list.

**Key Features of the Headless System:**

*   **Dual Database Backend:**
    *   **SQLite (Default):** Easy to set up, stores tasks in a local `tasks.db` file. Ideal for single-machine use.
    *   **PostgreSQL/Supabase:** Allows for a more robust, centralized task queue when configured via a `.env` file. Suitable for multi-worker setups or when a managed database is preferred.
*   **Automated Output Handling:**
    *   **SQLite Mode:** Videos are saved locally. For `steerable_motion.py` tasks, intermediate files go into subdirectories within `--output_dir`, and the final video is placed directly in `--output_dir`.
    *   **Supabase Mode:** Videos are uploaded to a configured Supabase Storage bucket. The public URL is stored in the database.
    *   **Persistent Task Queue:** Tasks are not lost if the server restarts.
    *   **Configurable Polling:** Set how often `headless.py` checks for new tasks.
    *   **Debug Mode:** Verbose logging for troubleshooting (`--debug` flag for both `headless.py` and `steerable_motion.py`).
    *   **Global `wgp.py` Overrides:** Configure `wgp.py` settings at `headless.py` server startup.

## Quick Start / Basic Setup

For a quick setup and to run the server (defaults to SQLite mode):

```bash
git clone https://github.com/peteromallet/Headless-Wan2GP /workspace/Wan2GP && \\
cd /workspace/Wan2GP && \\
apt-get update && apt-get install -y python3.10-venv ffmpeg && \\
python3.10 -m venv venv && \\
source venv/bin/activate && \\
pip install --no-cache-dir torch==2.6.0 torchvision torchaudio -f https://download.pytorch.org/whl/cu124 && \\
pip install --no-cache-dir -r Wan2GP/requirements.txt && \\
pip install --no-cache-dir -r requirements.txt 
python Wan2GP/headless.py
```

Once `headless.py` is running, you can open another terminal to queue tasks using `steerable_motion.py` (see examples below) or by adding tasks directly to the database.

<details>
<summary>Detailed Configuration and Supabase Setup</summary>

1.  **Clone the Repository (if not done by Quick Start):**
    ```bash
    git clone https://github.com/peteromallet/Headless-Wan2GP # Or your fork
    cd Headless-Wan2GP 
    # Note: The main scripts like steerable_motion.py are at the root.
    # Wan2GP-specific code is under the Wan2GP/ subdirectory.
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
    pip install --no-cache-dir torch==2.6.0 torchvision torchaudio -f https://download.pytorch.org/whl/cu124
    ```

4.  **Install Python Dependencies (if not done by Quick Start):**
    There are two main `requirements.txt` files:
    *   `Wan2GP/requirements.txt`: For the core Wan2GP library.
    *   `requirements.txt` (at the root): For `steerable_motion.py` and `headless.py` (includes `supabase`, `python-dotenv`, etc.).
    ```bash
    pip install --no-cache-dir -r Wan2GP/requirements.txt
    pip install --no-cache-dir -r requirements.txt 
    ```

5.  **Environment Configuration (`.env` file):**
    Create a `.env` file in the root directory of the `Headless-Wan2GP` project (i.e., next to `steerable_motion.py` and `headless.py`).
    ```env
    # --- Database Configuration ---
    # DB_TYPE: "sqlite" (default) or "supabase" (for PostgreSQL via Supabase)
    DB_TYPE=sqlite

    # If DB_TYPE=sqlite, you can optionally specify a custom path for the SQLite DB file.
    # If not set, defaults to "tasks.db" in the current working directory.
    # SQLITE_DB_PATH_ENV="path/to/your/custom_tasks.db"

    # If DB_TYPE=supabase:
    # POSTGRES_TABLE_NAME: Desired table name for tasks, used in RPC calls. Default: "tasks"
    POSTGRES_TABLE_NAME="tasks" 
    SUPABASE_URL="https://your-project-ref.supabase.co"
    SUPABASE_SERVICE_KEY="your_supabase_service_role_key" # Keep this secret!
    SUPABASE_VIDEO_BUCKET="videos" # Your Supabase storage bucket name

    # --- ComfyUI Configuration (Optional) ---
    # If using the "comfyui" execution_engine, headless.py needs to know where ComfyUI saves outputs.
    # COMFYUI_OUTPUT_PATH="/path/to/your/ComfyUI/output" 
    ```
    *   `headless.py` and `steerable_motion.py` will load these variables.
    *   If `DB_TYPE=supabase` is set but `SUPABASE_URL` or `SUPABASE_SERVICE_KEY` are missing, scripts will warn and may fall back to SQLite or fail.

6.  **Supabase Setup (If using `DB_TYPE=supabase`):**
    *   **SQL Functions (CRITICAL):** You MUST create specific SQL functions in your Supabase PostgreSQL database. `headless.py` relies on these. Go to your Supabase Dashboard -> SQL Editor -> "New query" and execute the following definitions:

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
                    task_type TEXT NOT NULL, -- Added NOT NULL constraint
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
            
            -- Index for task_type, useful for querying specific task types
            EXECUTE format('
                CREATE INDEX IF NOT EXISTS idx_%s_task_type ON %I (task_type);
            ', p_table_name, p_table_name);

            -- Index for orchestrator_run_id within params, useful for travel tasks
            -- Note: This creates an index on a JSONB path. Ensure your Postgres version supports this efficiently.
            EXECUTE format('
                CREATE INDEX IF NOT EXISTS idx_%s_params_orchestrator_run_id ON %I ((params->>''orchestrator_run_id''));
            ', p_table_name, p_table_name);

        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
        ```

        **Function 2: `func_claim_task`** (Updated to return `task_type_out`)
        ```sql
        CREATE OR REPLACE FUNCTION func_claim_task(p_table_name TEXT, p_worker_id TEXT)
        RETURNS TABLE(task_id_out TEXT, params_out JSONB, task_type_out TEXT) AS $$
        DECLARE
            v_task_id TEXT;
            v_params JSONB;
            v_task_type TEXT;
        BEGIN
            EXECUTE format('
                WITH selected_task AS (
                    SELECT id, task_id, params, task_type -- Include task_type
                    FROM %I
                    WHERE status = ''Queued''
                    ORDER BY created_at ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                ), updated_task AS (
                    UPDATE %I
                    SET
                        status = ''In Progress'',
                        worker_id = $1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = (SELECT st.id FROM selected_task st)
                    RETURNING task_id, params, task_type -- Return task_type
                )
                SELECT ut.task_id, ut.params, ut.task_type FROM updated_task ut LIMIT 1',
                p_table_name, p_table_name
            )
            INTO v_task_id, v_params, v_task_type -- Store task_type
            USING p_worker_id;

            IF v_task_id IS NOT NULL THEN
                RETURN QUERY SELECT v_task_id, v_params, v_task_type;
            ELSE
                RETURN QUERY SELECT NULL::TEXT, NULL::JSONB, NULL::TEXT WHERE FALSE;
            END IF;
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
        ```

        **Function 3: `func_update_task_status`** (No changes needed from previous version if it handled `p_output_location` correctly)
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

        **Function 4: `func_migrate_tasks_for_task_type`** (New, for schema migration)
        ```sql
        CREATE OR REPLACE FUNCTION func_migrate_tasks_for_task_type(p_table_name TEXT)
        RETURNS TEXT AS $$
        DECLARE
            col_exists BOOLEAN;
            migrated_count INTEGER := 0;
            defaulted_count INTEGER := 0;
        BEGIN
            -- Check if task_type column exists
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = current_schema() -- or your specific schema
                AND table_name = p_table_name
                AND column_name = 'task_type'
            ) INTO col_exists;

            IF NOT col_exists THEN
                RAISE NOTICE 'Column task_type does not exist in %. Adding column.', p_table_name;
                EXECUTE format('ALTER TABLE %I ADD COLUMN task_type TEXT', p_table_name);
            ELSE
                RAISE NOTICE 'Column task_type already exists in %.', p_table_name;
            END IF;

            -- Populate task_type from params where task_type is NULL
            RAISE NOTICE 'Attempting to populate task_type from params->>''task_type'' for NULL rows in %s...', p_table_name;
            EXECUTE format('
                WITH updated_rows AS (
                    UPDATE %I
                    SET task_type = params->>''task_type''
                    WHERE task_type IS NULL AND params->>''task_type'' IS NOT NULL
                    RETURNING 1
                )
                SELECT count(*) FROM updated_rows;',
                p_table_name
            ) INTO migrated_count;
            RAISE NOTICE 'Populated task_type for % rows from params.', migrated_count;

            -- Optionally, default remaining NULL task_types for old standard tasks
            RAISE NOTICE 'Attempting to default remaining NULL task_type to ''standard_wgp_task'' in %s...', p_table_name;
            EXECUTE format('
                WITH defaulted_rows AS (
                    UPDATE %I
                    SET task_type = ''standard_wgp_task''
                    WHERE task_type IS NULL
                    RETURNING 1
                )
                SELECT count(*) FROM defaulted_rows;',
                p_table_name
            ) INTO defaulted_count;
            RAISE NOTICE 'Defaulted task_type for % rows.', defaulted_count;
            
            -- Ensure NOT NULL constraint if this function is also for initial setup
            -- However, func_initialize_tasks_table should handle this for new tables.
            -- If altering, ensure data is clean first or handle potential errors.
            -- EXECUTE format('ALTER TABLE %I ALTER COLUMN task_type SET NOT NULL;', p_table_name);
            -- RAISE NOTICE 'Applied NOT NULL constraint to task_type in %.', p_table_name;

            RETURN 'Migration for task_type completed. Migrated from params: ' || migrated_count || ', Defaulted NULLs: ' || defaulted_count;
        EXCEPTION
            WHEN OTHERS THEN
                RAISE WARNING 'Error during func_migrate_tasks_for_task_type for table %: %', p_table_name, SQLERRM;
                RETURN 'Migration for task_type failed. Check database logs.';
        END;
        $$ LANGUAGE plpgsql SECURITY DEFINER;
        ```

        **Function 5: `func_get_completed_travel_segments`** (New, for stitcher task)
        ```sql
        CREATE OR REPLACE FUNCTION func_get_completed_travel_segments(p_run_id TEXT)
        RETURNS TABLE(segment_index_out INTEGER, output_location_out TEXT) AS $$
        BEGIN
            RETURN QUERY
            SELECT
                CAST(params->>'segment_index' AS INTEGER) as segment_idx,
                output_location
            FROM
                tasks -- Assuming your table is named 'tasks', or use p_table_name if dynamic
            WHERE
                params->>'orchestrator_run_id' = p_run_id
                AND task_type = 'travel_segment'
                AND status = 'Complete'
                AND output_location IS NOT NULL
            ORDER BY
                segment_idx ASC;
        END;
        $$ LANGUAGE plpgsql; 
        -- Consider SECURITY DEFINER if needed, depending on row-level security policies.
        ```

    *   **Database Table:** The `func_initialize_tasks_table` will create the tasks table.
    *   **Storage Bucket:** Create a storage bucket in Supabase (e.g., "videos"). Ensure it's public for URLs to work.

7.  **FFmpeg:** Ensure `ffmpeg` is installed and accessible in your system's PATH, as it's used for various video processing utilities.
</details>

<details>
<summary>How it Works (Updated)</summary>

1.  **Task Definition (in the Database):**
    Tasks are stored as rows in the database. Key columns:
    *   `task_id` (TEXT, UNIQUE)
    *   `params` (JSON string/object): Contains all parameters for the task. For `steerable_motion.py` tasks, this will include a nested `orchestrator_details` payload for `travel_orchestrator` tasks, which is then passed to subsequent `travel_segment` and `travel_stitch` tasks.
    *   `task_type` (TEXT, NOT NULL): Specifies the handler in `headless.py` (e.g., "standard_wgp_task", "generate_openpose", "travel_orchestrator", "travel_segment", "travel_stitch", "comfyui_workflow").
    *   `status` (TEXT): "Queued", "In Progress", "Complete", "Failed".
    *   `output_location` (TEXT): Local path (SQLite) or public URL (Supabase) of the final primary artifact.

    **Example of a `travel_orchestrator` task's `params` (simplified):**
    ```json
    {
        // task_id, task_type, etc. are top-level columns in the DB
        "orchestrator_details": {
            "orchestrator_task_id": "sm_travel_orchestrator_XYZ",
            "run_id": "run_ABC",
            "original_task_args": {
                "input_images": ["img1.png", "img2.png"], 
                "base_prompts": ["prompt1"], 
                /* ...other steerable_motion.py args... */
            },
            "original_common_args": {/* ...common args... */},
            "input_image_paths_resolved": ["/abs/path/to/img1.png", "/abs/path/to/img2.png"],
            "num_new_segments_to_generate": 1,
            "base_prompts_expanded": ["prompt1"],
            "segment_frames_expanded": [81],
            "frame_overlap_expanded": [16],
            // ... and many other parameters for headless tasks to consume ...
            "main_output_dir_for_run": "/path/to/steerable_motion_output" 
        }
    }
    ```

2.  **Polling Mechanism:** `headless.py` polls for 'Queued' tasks.

3.  **Task Processing:**
    *   `headless.py` claims a task, updates status to 'In Progress'.
    *   It calls the appropriate handler based on `task_type`.
        *   For `travel_orchestrator`: Enqueues the first `travel_segment` task.
        *   For `travel_segment`: Creates guide videos, runs WGP/ComfyUI generation (as a sub-task), and then enqueues the next segment or the `travel_stitch` task.
        *   For `travel_stitch`: Collects all segment videos, stitches them (with crossfades), optionally upscales (as a sub-task), and saves the final video.
    *   Outputs are handled (local save or Supabase upload).

4.  **Task Completion:** Status becomes 'Complete'/'Failed', `output_location` is updated.
</details>

## Usage

### 1. Start the Headless Worker:
```bash
# Ensure your .env file is configured if using Supabase
python Wan2GP/headless.py --main-output-dir ./my_video_outputs --poll-interval 5 
# Add --debug for verbose logs
```
`headless.py` will continuously monitor the task queue.

### 2. Queue Tasks using `steerable_motion.py`:

Once `headless.py` is running, execute the following in another terminal:

**A. Example `travel_between_images` with Sample Images:**

```bash
python steerable_motion.py travel_between_images \\
    --input_images samples/image_1.png samples/image_2.png samples/image_3.png \\
    --base_prompts "Transitioning from red" "Moving to green" \\
    --resolution "320x240" \\
    --segment_frames 30 \\
    --frame_overlap 10 \\
    --model_name "vace_14B" \\
    --seed 789 \\
    --output_dir ./my_video_outputs \\
    --debug
```

**B. Example `different_pose` with a Sample Image:**

```bash
python steerable_motion.py different_pose \\
    --input_image samples/image_1.png \\
    --prompt "A red square, now animated and waving" \\
    --resolution "320x240" \\
    --output_video_frames 30 \\
    --model_name "vace_14B" \\
    --seed 101 \\
    --output_dir ./my_video_outputs \\
    --debug
```

Remember to have `headless.py` running in a separate terminal to process these queued tasks.

### Command-Line Arguments for `headless.py`:
(Content from previous README is largely still valid here, ensure it's present)
*   **Server Settings:**
    *   `--db-file` (Used for SQLite mode only): Path to the SQLite database file. Defaults to `tasks.db` (or value from `SQLITE_DB_PATH_ENV` in `.env`).
    *   `--main-output-dir`: Base directory for outputs (e.g., where `steerable_motion.py` tells tasks to save, and where `headless.py` might save for non-`steerable_motion` tasks if they don't specify a full sub-path). Defaults to `./outputs`.
    *   `--poll-interval`: How often (in seconds) to check the database for new tasks. Defaults to 10 seconds.
    *   `--debug`: Enable verbose debug logging.

*   **WGP Global Config Overrides (Optional - Applied once at server start):**
    (List of `--wgp-*` arguments from previous README is still valid)

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

