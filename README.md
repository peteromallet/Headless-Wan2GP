# Headless-Wan2GP – Minimal Quick Start

This guide shows the two supported ways to run the headless video-generation worker.

---

## 1  Run Locally (SQLite)

The worker keeps its task queue in a local `tasks.db` SQLite file and saves all videos under `./outputs`.

```bash
# 1) Grab the code and enter the folder
git clone https://github.com/<your-name>/Headless-Wan2GP.git
cd Headless-Wan2GP

# 2) Create a Python ≥3.10 virtual environment
python3 -m venv venv
source venv/bin/activate

# 3) Install PyTorch (pick the wheel that matches your hardware)
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4) Project requirements
pip install -r Wan2GP/requirements.txt
pip install -r requirements.txt

# 5) Start the worker – polls the local SQLite DB every 10 s
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

For optional flags and advanced usage, run `python headless.py --help`. Thank you for trying Headless-Wan2GP!