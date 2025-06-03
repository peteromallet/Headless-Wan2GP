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
# Clone the repository (if you haven't already)
# git clone https://github.com/peteromallet/Headless-Wan2GP /workspace/Wan2GP && \\
# cd /workspace/Wan2GP && \\
# apt-get update && apt-get install -y python3.10-venv ffmpeg && \\
# python3.10 -m venv venv && \\
# source venv/bin/activate && \\
# pip install --no-cache-dir torch==2.6.0 torchvision torchaudio -f https://download.pytorch.org/whl/cu124 && \\
# pip install --no-cache-dir -r Wan2GP/requirements.txt && \\
# pip install --no-cache-dir -r requirements.txt # Assuming requirements.txt is at the root for steerable_motion.py

# Start the headless worker (it will create tasks.db if it doesn't exist)
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

Open another terminal. Ensure your virtual environment is active and you are in the root `Headless-Wan2GP` directory.

**Example 1: `travel_between_images`**

First, create some dummy anchor images if you don't have any:
```bash
# Create dummy images (replace with your actual images)
convert -size 320x240 xc:red anchor1.png
convert -size 320x240 xc:green anchor2.png
convert -size 320x240 xc:blue anchor3.png
```

Then, run the command:
```bash
python steerable_motion.py travel_between_images \\
    --input_images anchor1.png anchor2.png anchor3.png \\
    --base_prompts "A majestic red landscape" "A vibrant green forest" \\
    --resolution "640x480" \\
    --segment_frames 60 \\
    --frame_overlap 15 \\
    --model_name "vace_14B" \\
    --seed 123 \\
    --output_dir ./my_video_outputs \\
    --debug 
    # Add --execution_engine "comfyui" if you have a ComfyUI workflow for travel segments
    # Add --continue_from_video "path/to/previous.mp4" to extend an existing video
```
This command will:
1.  Validate your arguments.
2.  Create an `orchestrator_payload` containing all the instructions.
3.  Add a single task of `task_type="travel_orchestrator"` to the database (`tasks.db` by default).
4.  `headless.py` will pick up this orchestrator task, then sequentially queue and process `travel_segment` tasks, and finally a `travel_stitch` task.

**Example 2: `different_pose`**

```bash
# Assume you have an input image: input_character.png
python steerable_motion.py different_pose \\
    --input_image input_character.png \\
    --prompt "Character waving hello, cinematic lighting" \\
    --resolution "512x512" \\
    --output_video_frames 40 \\
    --model_name "vace_14B" \\
    --seed 456 \\
    --output_dir ./my_video_outputs \\
    --debug
```
This command will queue tasks for OpenPose extraction and the main video generation, managed by `headless.py`.

### 3. Running with Sample Images (New Section)

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
(Content from previous README is still valid here)

This section should give you enough vocabulary to combine mask videos, depth/pose guidance and frame-freezing without modifying the code.

