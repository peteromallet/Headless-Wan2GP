# Supabase Setup for Headless Worker

This guide explains how to use `headless.py` with Supabase instead of SQLite.

## Prerequisites

1. A Supabase project with the `tasks` table configured
2. Required RPC functions installed (see below)
3. Either a user JWT token or service-role key

## Quick Start

### 1. Test Your Setup

First, verify your Supabase connection works:

```bash
python test_supabase_headless.py \
  --supabase-url https://your-project.supabase.co \
  --supabase-access-token your-jwt-token
```

### 2. Run Headless Worker

```bash
# With user token (processes only your tasks)
python headless.py \
  --db-type supabase \
  --supabase-url https://your-project.supabase.co \
  --supabase-access-token eyJhbGciOi...your-jwt-token

# With service-role key (processes all tasks)
python headless.py \
  --db-type supabase \
  --supabase-url https://your-project.supabase.co \
  --supabase-access-token your-service-role-key
```

## Required Database Setup

### RPC Function Update

The `func_claim_task` RPC function needs to be updated to return `project_id_out`. Add this to your function:

```sql
-- In your func_claim_task function, ensure it returns:
SELECT 
  v_task_id AS task_id_out,
  v_params AS params_out,
  v_task_type AS task_type_out,
  v_project_id AS project_id_out  -- ADD THIS LINE
FROM ...
```

## Token Types

| Token Type | What it processes | Use case |
|------------|-------------------|----------|
| User JWT | Only tasks where `user_id` matches the token | Individual users |
| Service-role key | All tasks (bypasses RLS) | Central server |

## Troubleshooting

### "project_id_out is missing"
Update your `func_claim_task` RPC function to return the project_id field.

### "Failed to connect to Supabase"
- Check your URL format (should be `https://xyz.supabase.co`)
- Verify your token is valid
- Ensure the tasks table exists

### Tasks not being picked up
- Verify tasks have status = 'Queued'
- Check RLS policies if using user token
- Run the test script to debug

## File Storage

Currently, the implementation stores files locally. To use Supabase Storage:

1. Create a storage bucket (e.g., "videos")
2. Update `SUPABASE_VIDEO_BUCKET` in your .env
3. Files will be uploaded automatically (feature in development)

## Environment Variables

You can also set these in your `.env` file:

```env
POSTGRES_TABLE_NAME=tasks
SUPABASE_VIDEO_BUCKET=videos
```

The CLI arguments take precedence over environment variables. 

## Supabase Storage Setup

Your Supabase project needs a storage bucket to store generated videos and images.

1. Go to your Supabase project → Storage
2. Create a bucket named `image_uploads` (or whatever you prefer)
3. Set the bucket to public if you want direct URL access
4. Update `SUPABASE_VIDEO_BUCKET` in your .env

```bash
# .env example
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
SUPABASE_VIDEO_BUCKET=image_uploads
``` 