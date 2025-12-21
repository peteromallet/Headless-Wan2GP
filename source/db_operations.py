# source/sm_functions/db_operations.py
import os
import sys
import json
import time
import traceback
import datetime
import urllib.parse
import httpx  # For calling Supabase Edge Function
from pathlib import Path
import base64 # Added for JWT decoding

try:
    from supabase import create_client, Client as SupabaseClient
except ImportError:
    SupabaseClient = None

# -----------------------------------------------------------------------------
# Global DB Configuration (will be set by worker.py)
# -----------------------------------------------------------------------------
PG_TABLE_NAME = "tasks"
SUPABASE_URL = None
SUPABASE_SERVICE_KEY = None
SUPABASE_VIDEO_BUCKET = "image_uploads"
SUPABASE_CLIENT: SupabaseClient | None = None
SUPABASE_EDGE_COMPLETE_TASK_URL: str | None = None  # Optional override for edge function
SUPABASE_ACCESS_TOKEN: str | None = None # Will be set by worker.py
SUPABASE_EDGE_CREATE_TASK_URL: str | None = None # Will be set by worker.py
SUPABASE_EDGE_CLAIM_TASK_URL: str | None = None # Will be set by worker.py

# -----------------------------------------------------------------------------
# Status Constants
# -----------------------------------------------------------------------------
STATUS_QUEUED = "Queued"
STATUS_IN_PROGRESS = "In Progress"
STATUS_COMPLETE = "Complete"
STATUS_FAILED = "Failed"

# -----------------------------------------------------------------------------
# Debug / Verbose Logging Helpers
# -----------------------------------------------------------------------------
debug_mode = False

def dprint(msg: str):
    """Print a debug message if debug_mode is enabled."""
    if debug_mode:
        print(f"[DEBUG {datetime.datetime.now().isoformat()}] {msg}")

# -----------------------------------------------------------------------------
# Internal Helpers for Supabase
# -----------------------------------------------------------------------------

def _get_user_id_from_jwt(jwt_str: str) -> str | None:
    """Decodes a JWT and extracts the 'sub' (user ID) claim without validation."""
    if not jwt_str:
        return None
    try:
        # JWT is composed of header.payload.signature
        _, payload_b64, _ = jwt_str.split('.')
        # The payload is base64 encoded. It needs to be padded to be decoded correctly.
        payload_b64 += '=' * (-len(payload_b64) % 4)
        payload_json = base64.b64decode(payload_b64).decode('utf-8')
        payload = json.loads(payload_json)
        user_id = payload.get('sub')
        dprint(f"JWT Decode: Extracted user ID (sub): {user_id}")
        return user_id
    except Exception as e:
        dprint(f"[ERROR] Could not decode JWT to get user ID: {e}")
        return None

def _is_jwt_token(token_str: str) -> bool:
    """
    Checks if a token string looks like a JWT (has 3 parts separated by dots).
    """
    if not token_str:
        return False
    parts = token_str.split('.')
    return len(parts) == 3

def _mark_task_failed_via_edge_function(task_id_str: str, error_message: str):
    """Mark a task as failed using the update-task-status Edge Function"""
    try:
        edge_url = (
            os.getenv("SUPABASE_EDGE_UPDATE_TASK_URL")
            or (f"{SUPABASE_URL.rstrip('/')}/functions/v1/update-task-status" if SUPABASE_URL else None)
        )

        if not edge_url:
            print(f"[ERROR] No update-task-status edge function URL available for marking task {task_id_str} as failed")
            return

        headers = {"Content-Type": "application/json"}
        if SUPABASE_ACCESS_TOKEN:
            headers["Authorization"] = f"Bearer {SUPABASE_ACCESS_TOKEN}"

        payload = {
            "task_id": task_id_str,
            "status": STATUS_FAILED,
            "output_location": error_message
        }

        resp = httpx.post(edge_url, json=payload, headers=headers, timeout=30)

        if resp.status_code == 200:
            dprint(f"[DEBUG] Successfully marked task {task_id_str} as Failed via Edge Function")
        else:
            print(f"[ERROR] Failed to mark task {task_id_str} as Failed: {resp.status_code} - {resp.text}")

    except Exception as e:
        print(f"[ERROR] Exception marking task {task_id_str} as Failed: {e}")

# -----------------------------------------------------------------------------
# Public Database Functions
# -----------------------------------------------------------------------------

def init_db():
    """Initializes the Supabase database connection."""
    return init_db_supabase()

def get_oldest_queued_task():
    """Gets the oldest queued task from Supabase."""
    return get_oldest_queued_task_supabase()

def update_task_status(task_id: str, status: str, output_location: str | None = None):
    """Updates a task's status in Supabase."""
    dprint(f"[UPDATE_TASK_STATUS_DEBUG] Called with:")
    dprint(f"[UPDATE_TASK_STATUS_DEBUG]   task_id: '{task_id}'")
    dprint(f"[UPDATE_TASK_STATUS_DEBUG]   status: '{status}'")
    dprint(f"[UPDATE_TASK_STATUS_DEBUG]   output_location: '{output_location}'")

    try:
        dprint(f"[UPDATE_TASK_STATUS_DEBUG] Dispatching to update_task_status_supabase")
        result = update_task_status_supabase(task_id, status, output_location)
        dprint(f"[UPDATE_TASK_STATUS_DEBUG] update_task_status_supabase completed successfully")
        return result
    except Exception as e:
        dprint(f"[UPDATE_TASK_STATUS_DEBUG] ❌ Exception in update_task_status: {e}")
        dprint(f"[UPDATE_TASK_STATUS_DEBUG] Exception type: {type(e).__name__}")
        traceback.print_exc()
        raise

def init_db_supabase():
    """Check if the Supabase tasks table exists and is accessible."""
    if not SUPABASE_CLIENT:
        print("[ERROR] Supabase client not initialized. Cannot check database table.")
        sys.exit(1)
    try:
        # Simply check if the tasks table exists by querying it
        result = SUPABASE_CLIENT.table(PG_TABLE_NAME).select("count", count="exact").limit(1).execute()
        print(f"Supabase: Table '{PG_TABLE_NAME}' exists and accessible (count: {result.count})")
        return True
    except Exception as e:
        print(f"[ERROR] Supabase table check failed: {e}")
        # Don't exit - the table might exist but have different permissions
        # Let the actual operations try and fail gracefully
        return False

def check_task_counts_supabase(run_type: str = "gpu") -> dict | None:
    """Check task counts via Supabase Edge Function before attempting to claim tasks."""
    if not SUPABASE_CLIENT or not SUPABASE_ACCESS_TOKEN:
        dprint("[ERROR] Supabase client or access token not initialized. Cannot check task counts.")
        return None
    
    # Build task-counts edge function URL using same pattern as other functions
    edge_url = (
        os.getenv('SUPABASE_EDGE_TASK_COUNTS_URL')
        or (f"{SUPABASE_URL.rstrip('/')}/functions/v1/task-counts" if SUPABASE_URL else None)
    )
    
    if not edge_url:
        dprint("ERROR: No task-counts edge function URL available")
        return None
    
    try:
        # Use same authentication pattern as other edge functions - SUPABASE_ACCESS_TOKEN
        # This can be a service key, PAT, or JWT - the edge function determines the type
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {SUPABASE_ACCESS_TOKEN}'
        }
        
        payload = {
            "run_type": run_type,
            "include_active": True
        }
        
        dprint(f"DEBUG check_task_counts_supabase: Calling task-counts at {edge_url}")
        resp = httpx.post(edge_url, json=payload, headers=headers, timeout=10)
        dprint(f"Task-counts response status: {resp.status_code}")
        
        if resp.status_code == 200:
            counts_data = resp.json()
            # Always log a concise summary so we can observe behavior without enabling debug
            try:
                totals = counts_data.get('totals', {})
                dprint(f"[TASK_COUNTS] totals={totals} run_type={payload.get('run_type')}")
            except Exception:
                # Fall back to raw text if JSON structure unexpected
                dprint(f"[TASK_COUNTS] raw_response={resp.text[:500]}")
            dprint(f"Task-counts result: {counts_data.get('totals', {})}")
            return counts_data
        else:
            dprint(f"[TASK_COUNTS] error status={resp.status_code} body={resp.text[:500]}")
            dprint(f"Task-counts returned {resp.status_code}: {resp.text}")
            return None
            
    except Exception as e_counts:
        dprint(f"Task-counts call failed: {e_counts}")
        return None

def check_my_assigned_tasks(worker_id: str) -> dict | None:
    """
    Check if this worker has any tasks already assigned to it (In Progress).
    This handles the case where a claim succeeded but the response was lost.
    
    Returns task data dict if found, None otherwise.
    """
    if not SUPABASE_CLIENT or not worker_id:
        return None
    
    try:
        # Query for tasks assigned to this worker that are In Progress
        result = SUPABASE_CLIENT.table('tasks') \
            .select('id, params, task_type, project_id') \
            .eq('worker_id', worker_id) \
            .eq('status', STATUS_IN_PROGRESS) \
            .limit(1) \
            .execute()
        
        if result.data and len(result.data) > 0:
            task = result.data[0]
            task_id = task.get('id')
            task_type = task.get('task_type', 'unknown')
            params = task.get('params', {})
            
            print(f"[RECOVERY] 🔄 Found assigned task {task_id} (type={task_type}) - recovering")
            dprint(f"[RECOVERY_DEBUG] Task was assigned to us but we didn't process it - likely lost HTTP response")
            
            # Return in the same format as claim-next-task Edge Function
            return {
                'task_id': task_id,
                'params': params,
                'task_type': task_type,
                'project_id': task.get('project_id')
            }
        
        return None
        
    except Exception as e:
        # Don't let recovery check failures block normal operation
        dprint(f"[RECOVERY] Failed to check assigned tasks: {e}")
        return None


def get_oldest_queued_task_supabase(worker_id: str = None):
    """Fetches the oldest task via Supabase Edge Function. First checks task counts to avoid unnecessary claim attempts."""
    if not SUPABASE_CLIENT:
        print("[ERROR] Supabase client not initialized. Cannot get task.")
        return None
    
    # Use provided worker_id or use the specific GPU worker ID
    if not worker_id:
        worker_id = "gpu-20250723_221138-afa8403b"
        dprint(f"DEBUG: No worker_id provided, using default GPU worker: {worker_id}")
    else:
        dprint(f"DEBUG: Using provided worker_id: {worker_id}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1: Check for tasks already assigned to this worker (RECOVERY)
    # This handles the case where a previous claim succeeded in the DB but
    # the HTTP response was lost before we could process it.
    # ═══════════════════════════════════════════════════════════════════════════
    assigned_task = check_my_assigned_tasks(worker_id)
    if assigned_task:
        return assigned_task
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2: No assigned tasks - check for new tasks to claim
    # ═══════════════════════════════════════════════════════════════════════════
    
    # OPTIMIZATION: Check task counts first to avoid unnecessary claim attempts
    dprint("Checking task counts before attempting to claim...")
    task_counts = check_task_counts_supabase("gpu")
    
    if task_counts is None:
        dprint("WARNING: Could not check task counts, proceeding with direct claim attempt")
    else:
        totals = task_counts.get('totals', {})
        # Gate claim by queued_only to avoid claiming when only active tasks exist
        available_tasks = totals.get('queued_only', 0)
        eligible_queued = totals.get('eligible_queued', 0)
        active_only = totals.get('active_only', 0)
        
        dprint(f"[CLAIM_DEBUG] Task counts: queued_only={available_tasks}, eligible_queued={eligible_queued}, active_only={active_only}")
        
        # Log warning if counts are inconsistent
        if eligible_queued > 0 and available_tasks == 0:
            print(f"[WARN] ⚠️  Task count inconsistency detected: eligible_queued={eligible_queued} but queued_only={available_tasks}")
            print(f"[WARN] This suggests tasks exist but aren't visible as 'Queued' status - possible replication lag or status corruption")
            # Proceed with claim attempt despite queued_only=0 since eligible_queued>0
            dprint(f"[CLAIM_DEBUG] Proceeding with claim attempt despite queued_only=0 because eligible_queued={eligible_queued}")
        elif available_tasks <= 0:
            dprint("No queued tasks according to task-counts, skipping claim attempt")
            return None
        else:
            dprint(f"Found {available_tasks} queued tasks, proceeding with claim")
    
    # Use Edge Function exclusively
    edge_url = (
        SUPABASE_EDGE_CLAIM_TASK_URL 
        or os.getenv('SUPABASE_EDGE_CLAIM_TASK_URL')
        or (f"{SUPABASE_URL.rstrip('/')}/functions/v1/claim-next-task" if SUPABASE_URL else None)
    )
    
    if edge_url and SUPABASE_ACCESS_TOKEN:
        try:
            dprint(f"DEBUG get_oldest_queued_task_supabase: Calling Edge Function at {edge_url}")
            dprint(f"DEBUG: Using worker_id: {worker_id}")
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {SUPABASE_ACCESS_TOKEN}'
            }
            
            # Pass worker_id and run_type in the request body for edge function to use
            payload = {"worker_id": worker_id, "run_type": "gpu"}
            
            resp = httpx.post(edge_url, json=payload, headers=headers, timeout=15)
            dprint(f"Edge Function response status: {resp.status_code}")
            
            if resp.status_code == 200:
                task_data = resp.json()
                task_id = task_data.get('task_id', 'unknown')
                task_type = task_data.get('task_type', 'unknown')
                params = task_data.get('params', {})
                segment_index = params.get('segment_index') if isinstance(params, dict) else None
                
                print(f"[CLAIM] ✅ Claimed task {task_id} (type={task_type}, segment_index={segment_index})")
                dprint(f"[CLAIM_DEBUG] Full task data: {task_data}")
                return task_data  # Already in the expected format
            elif resp.status_code == 204:
                dprint("Edge Function: No queued tasks available")
                return None
            else:
                dprint(f"Edge Function returned {resp.status_code}: {resp.text}")
                return None
        except Exception as e_edge:
            # Log visibly - this is a critical failure that can cause orphaned tasks
            print(f"[CLAIM] ❌ Edge Function call failed: {e_edge}")
            dprint(f"[CLAIM_DEBUG] Exception type: {type(e_edge).__name__}")
            dprint(f"[CLAIM_DEBUG] Full traceback: {traceback.format_exc()}")
            return None
    else:
        print("[CLAIM] ❌ No edge function URL or access token available for task claiming")
        return None

def update_task_status_supabase(task_id_str, status_str, output_location_val=None, thumbnail_url_val=None):
    """Updates a task's status via Supabase Edge Functions.

    Args:
        task_id_str: Task ID
        status_str: Status to set
        output_location_val: Output file location or URL
        thumbnail_url_val: Optional thumbnail URL to pass to edge function
    """
    dprint(f"[DEBUG] update_task_status_supabase called: task_id={task_id_str}, status={status_str}, output_location={output_location_val}, thumbnail={thumbnail_url_val}")
    
    if not SUPABASE_CLIENT:
        print("[ERROR] Supabase client not initialized. Cannot update task status.")
        return

    # --- Use edge functions for ALL status updates ---
    if status_str == STATUS_COMPLETE and output_location_val is not None:
        # Use complete_task edge function for completion with file
        edge_url = (
            SUPABASE_EDGE_COMPLETE_TASK_URL
            or (os.getenv("SUPABASE_EDGE_COMPLETE_TASK_URL") or None)
            or (f"{SUPABASE_URL.rstrip('/')}/functions/v1/complete_task" if SUPABASE_URL else None)
        )
        
        if not edge_url:
            print(f"[ERROR] No complete_task edge function URL available")
            return

        try:
            # Check if output_location_val is a local file path
            output_path = Path(output_location_val)

            if output_path.exists() and output_path.is_file():
                import base64
                import mimetypes

                # Get file size for logging
                file_size = output_path.stat().st_size
                file_size_mb = file_size / (1024 * 1024)
                dprint(f"[DEBUG] File size: {file_size_mb:.2f} MB")

                headers = {"Content-Type": "application/json"}
                if SUPABASE_ACCESS_TOKEN:
                    headers["Authorization"] = f"Bearer {SUPABASE_ACCESS_TOKEN}"

                # Check if this is a video file that needs thumbnail extraction
                video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
                is_video = output_path.suffix.lower() in video_extensions
                
                # Use base64 encoding for files under 2MB (MODE 1), presigned URLs for larger files (MODE 3)
                FILE_SIZE_THRESHOLD_MB = 2.0
                use_base64 = file_size_mb < FILE_SIZE_THRESHOLD_MB
                
                if use_base64:
                    dprint(f"[DEBUG] Using base64 upload for {output_path.name} ({file_size_mb:.2f} MB < {FILE_SIZE_THRESHOLD_MB} MB)")
                    
                    # MODE 1: Base64 upload (for files under 2MB)
                    # Read and encode file
                    with open(output_path, 'rb') as f:
                        file_bytes = f.read()
                        file_data_base64 = base64.b64encode(file_bytes).decode('utf-8')
                    
                    payload = {
                        "task_id": task_id_str,
                        "file_data": file_data_base64,
                        "filename": output_path.name
                    }
                    
                    # Extract and encode first frame for videos
                    if is_video:
                        try:
                            import tempfile
                            from .common_utils import save_frame_from_video
                            import cv2

                            dprint(f"[DEBUG] Extracting first frame from video {output_path.name}")

                            # Create temporary file for first frame
                            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_frame:
                                temp_frame_path = Path(temp_frame.name)

                            # Get video resolution for frame extraction
                            cap = cv2.VideoCapture(str(output_path))
                            if cap.isOpened():
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                cap.release()

                                # Extract first frame (index 0)
                                if save_frame_from_video(output_path, 0, temp_frame_path, (width, height)):
                                    dprint(f"[DEBUG] Extracted first frame, encoding as base64")
                                    
                                    # Read and encode thumbnail
                                    with open(temp_frame_path, 'rb') as thumb_file:
                                        thumb_bytes = thumb_file.read()
                                        first_frame_base64 = base64.b64encode(thumb_bytes).decode('utf-8')
                                    
                                    payload["first_frame_data"] = first_frame_base64
                                    # Use unique filename based on task_id to prevent overwrites
                                    payload["first_frame_filename"] = f"thumb_{task_id_str[:8]}.jpg"
                                    dprint(f"[DEBUG] First frame encoded successfully")
                                else:
                                    dprint(f"[WARNING] Failed to extract first frame from video")
                            else:
                                dprint(f"[WARNING] Could not open video for frame extraction")

                            # Clean up temporary file
                            try:
                                temp_frame_path.unlink()
                            except:
                                pass

                        except Exception as e:
                            dprint(f"[WARNING] Error extracting/encoding thumbnail: {e}")
                            # Continue without thumbnail
                    
                    dprint(f"[DEBUG] Calling complete_task Edge Function with base64 data for task {task_id_str}")
                    resp = httpx.post(edge_url, json=payload, headers=headers, timeout=60)

                    if resp.status_code == 200:
                        dprint(f"[DEBUG] Edge function SUCCESS for task {task_id_str} → status COMPLETE with base64 upload")
                        # Parse response to get storage URL and thumbnail URL
                        try:
                            resp_data = resp.json()
                            storage_url = resp_data.get('public_url')
                            thumbnail_url = resp_data.get('thumbnail_url')  # Also get thumbnail
                            if storage_url:
                                dprint(f"[DEBUG] File uploaded to: {storage_url}")
                                if thumbnail_url:
                                    dprint(f"[DEBUG] Thumbnail available at: {thumbnail_url}")
                                # Return both URLs as a dict
                                return {'public_url': storage_url, 'thumbnail_url': thumbnail_url}
                        except:
                            pass
                        return None
                    else:
                        error_msg = f"complete_task edge function failed: {resp.status_code} - {resp.text}"
                        print(f"[ERROR] {error_msg}")
                        _mark_task_failed_via_edge_function(task_id_str, f"Upload failed: {error_msg}")
                        return None
                    
                else:
                    dprint(f"[DEBUG] Using presigned upload for {output_path.name} ({file_size_mb:.2f} MB >= {FILE_SIZE_THRESHOLD_MB} MB)")
                    
                    # MODE 3: Presigned URL upload (for files 2MB or larger)
                    # Step 1: Get signed upload URLs (request thumbnail URL for videos)
                    generate_url_edge = f"{SUPABASE_URL.rstrip('/')}/functions/v1/generate-upload-url"
                    content_type = mimetypes.guess_type(str(output_path))[0] or 'application/octet-stream'

                    upload_url_resp = httpx.post(
                        generate_url_edge,
                        headers=headers,
                        json={
                            "task_id": task_id_str,
                            "filename": output_path.name,
                            "content_type": content_type,
                            "generate_thumbnail_url": is_video  # Request thumbnail URL for videos
                        },
                        timeout=30
                    )

                    if upload_url_resp.status_code != 200:
                        error_msg = f"Failed to generate upload URL: {upload_url_resp.status_code} - {upload_url_resp.text}"
                        print(f"[ERROR] {error_msg}")
                        _mark_task_failed_via_edge_function(task_id_str, f"Upload failed: {error_msg}")
                        return

                    upload_data = upload_url_resp.json()

                    # Step 2: Extract and upload thumbnail for videos (if thumbnail URL was generated)
                    thumbnail_storage_path = None
                    if is_video and "thumbnail_upload_url" in upload_data:
                        try:
                            import tempfile
                            from .common_utils import save_frame_from_video
                            import cv2

                            dprint(f"[DEBUG] Extracting first frame from video {output_path.name}")

                            # Create temporary file for first frame
                            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_frame:
                                temp_frame_path = Path(temp_frame.name)

                            # Get video resolution for frame extraction
                            cap = cv2.VideoCapture(str(output_path))
                            if cap.isOpened():
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                cap.release()

                                # Extract first frame (index 0)
                                if save_frame_from_video(output_path, 0, temp_frame_path, (width, height)):
                                    dprint(f"[DEBUG] Extracted first frame, uploading via signed URL")
                                    
                                    # Upload thumbnail directly via signed URL
                                    with open(temp_frame_path, 'rb') as thumb_file:
                                        thumb_resp = httpx.put(
                                            upload_data["thumbnail_upload_url"],
                                            headers={"Content-Type": "image/jpeg"},
                                            content=thumb_file,
                                            timeout=60
                                        )
                                    
                                    if thumb_resp.status_code in [200, 201]:
                                        thumbnail_storage_path = upload_data["thumbnail_storage_path"]
                                        dprint(f"[DEBUG] Thumbnail uploaded successfully via signed URL")
                                    else:
                                        dprint(f"[WARNING] Thumbnail upload failed: {thumb_resp.status_code}")
                                else:
                                    dprint(f"[WARNING] Failed to extract first frame from video")
                            else:
                                dprint(f"[WARNING] Could not open video for frame extraction")

                            # Clean up temporary file
                            try:
                                temp_frame_path.unlink()
                            except:
                                pass

                        except Exception as e:
                            dprint(f"[WARNING] Error extracting/uploading thumbnail: {e}")
                            # Continue without thumbnail

                    # Step 3: Upload main file directly to storage using presigned URL
                    dprint(f"[DEBUG] Uploading main file via signed URL")
                    with open(output_path, 'rb') as f:
                        put_resp = httpx.put(
                            upload_data["upload_url"],
                            headers={"Content-Type": content_type},
                            content=f,
                            timeout=600  # 10 minute timeout for large files
                        )

                    if put_resp.status_code not in [200, 201]:
                        error_msg = f"Failed to upload file to storage: {put_resp.status_code} - {put_resp.text}"
                        print(f"[ERROR] {error_msg}")
                        _mark_task_failed_via_edge_function(task_id_str, f"Upload failed: {error_msg}")
                        return

                    dprint(f"[DEBUG] File uploaded successfully via presigned URL")

                    # Step 4: Complete task with storage paths
                    payload = {
                        "task_id": task_id_str,
                        "storage_path": upload_data["storage_path"]
                    }
                    
                    # Add thumbnail storage path if available
                    if thumbnail_storage_path:
                        payload["thumbnail_storage_path"] = thumbnail_storage_path

                    dprint(f"[DEBUG] Calling complete_task Edge Function with storage_path for task {task_id_str}")
                    resp = httpx.post(edge_url, json=payload, headers=headers, timeout=60)

                    if resp.status_code == 200:
                        dprint(f"[DEBUG] Edge function SUCCESS for task {task_id_str} → status COMPLETE with file upload")
                        # Parse response to get storage URL and thumbnail URL
                        try:
                            resp_data = resp.json()
                            storage_url = resp_data.get('public_url')
                            thumbnail_url = resp_data.get('thumbnail_url')  # Also get thumbnail
                            if storage_url:
                                dprint(f"[DEBUG] File uploaded to: {storage_url}")
                                if thumbnail_url:
                                    dprint(f"[DEBUG] Thumbnail available at: {thumbnail_url}")
                                # Return both URLs as a dict
                                return {'public_url': storage_url, 'thumbnail_url': thumbnail_url}
                        except:
                            pass
                        return None
                    else:
                        error_msg = f"complete_task edge function failed: {resp.status_code} - {resp.text}"
                        print(f"[ERROR] {error_msg}")
                        # Use update-task-status edge function to mark as failed
                        _mark_task_failed_via_edge_function(task_id_str, f"Upload failed: {error_msg}")
                        return None
            else:
                # Not a local file, could be a Supabase storage URL or external reference
                # Check if it's a Supabase storage URL - if so, extract storage_path for MODE 3/4
                storage_path = None
                if "/storage/v1/object/public/image_uploads/" in output_location_val:
                    # Extract storage path from URL
                    # Format: https://xxx.supabase.co/storage/v1/object/public/image_uploads/{userId}/{filename}
                    # MODE 3: userId/tasks/{task_id}/filename (pre-signed URL uploads)
                    # MODE 4: userId/filename (orchestrator referencing child task upload)
                    try:
                        path_parts = output_location_val.split("/storage/v1/object/public/image_uploads/", 1)
                        if len(path_parts) == 2:
                            storage_path = path_parts[1]  # e.g., "userId/filename.mp4" or "userId/tasks/{task_id}/filename.mp4"
                            dprint(f"[DEBUG] Extracted storage_path from URL: {storage_path}")

                            # Determine if this is MODE 3 or MODE 4 based on path structure
                            path_components = storage_path.split('/')
                            if len(path_components) >= 4 and path_components[1] == 'tasks':
                                dprint(f"[DEBUG] MODE 3 path detected (pre-signed URL): {storage_path}")
                            else:
                                dprint(f"[DEBUG] MODE 4 path detected (orchestrator reference): {storage_path}")
                    except Exception as e_extract:
                        dprint(f"[DEBUG] Failed to extract storage_path: {e_extract}")

                # Use MODE 3/4 if we have a storage path, otherwise use output_location (legacy)
                if storage_path:
                    payload = {"task_id": task_id_str, "storage_path": storage_path}
                    dprint(f"[DEBUG] Using storage_path for task {task_id_str}")

                    # Extract thumbnail storage path if thumbnail URL provided
                    if thumbnail_url_val and "/storage/v1/object/public/image_uploads/" in thumbnail_url_val:
                        try:
                            thumb_parts = thumbnail_url_val.split("/storage/v1/object/public/image_uploads/", 1)
                            if len(thumb_parts) == 2:
                                thumbnail_storage_path = thumb_parts[1]
                                payload["thumbnail_storage_path"] = thumbnail_storage_path
                                dprint(f"[DEBUG] Including thumbnail_storage_path: {thumbnail_storage_path}")
                        except Exception as e:
                            dprint(f"[DEBUG] Failed to extract thumbnail path: {e}")
                else:
                    payload = {"task_id": task_id_str, "output_location": output_location_val}
                    dprint(f"[DEBUG] Using output_location (legacy) for task {task_id_str}")

                headers = {"Content-Type": "application/json"}
                if SUPABASE_ACCESS_TOKEN:
                    headers["Authorization"] = f"Bearer {SUPABASE_ACCESS_TOKEN}"

                resp = httpx.post(edge_url, json=payload, headers=headers, timeout=30)

                if resp.status_code == 200:
                    dprint(f"[DEBUG] Edge function SUCCESS for task {task_id_str} → status COMPLETE")
                    # Parse response to get storage URL and thumbnail URL
                    try:
                        resp_data = resp.json()
                        storage_url = resp_data.get('public_url')
                        thumbnail_url = resp_data.get('thumbnail_url')  # Also get thumbnail
                        if storage_url:
                            dprint(f"[DEBUG] Storage URL: {storage_url}")
                            if thumbnail_url:
                                dprint(f"[DEBUG] Thumbnail available at: {thumbnail_url}")
                            # Return both URLs as a dict
                            return {'public_url': storage_url, 'thumbnail_url': thumbnail_url}
                        # If no URL in response, return the original output_location_val as string (legacy)
                        return output_location_val
                    except:
                        return output_location_val
                else:
                    error_msg = f"complete_task edge function failed: {resp.status_code} - {resp.text}"
                    print(f"[ERROR] {error_msg}")
                    # Use update-task-status edge function to mark as failed
                    _mark_task_failed_via_edge_function(task_id_str, f"Completion failed: {error_msg}")
                    return None
        except Exception as e_edge:
            print(f"[ERROR] complete_task edge function exception: {e_edge}")
            return None
    else:
        # Use update-task-status edge function for all other status updates
        edge_url = (
            os.getenv("SUPABASE_EDGE_UPDATE_TASK_URL") 
            or (f"{SUPABASE_URL.rstrip('/')}/functions/v1/update-task-status" if SUPABASE_URL else None)
        )
        
        if not edge_url:
            print(f"[ERROR] No update-task-status edge function URL available")
            return
            
        try:
            headers = {"Content-Type": "application/json"}
            if SUPABASE_ACCESS_TOKEN:
                headers["Authorization"] = f"Bearer {SUPABASE_ACCESS_TOKEN}"
            
            payload = {
                "task_id": task_id_str,
                "status": status_str
            }
            
            if output_location_val:
                payload["output_location"] = output_location_val
                
            if thumbnail_url_val:
                payload["thumbnail_url"] = thumbnail_url_val
            
            dprint(f"[DEBUG] Calling update-task-status Edge Function for task {task_id_str} → {status_str}")
            resp = httpx.post(edge_url, json=payload, headers=headers, timeout=30)
            
            if resp.status_code == 200:
                dprint(f"[DEBUG] Edge function SUCCESS for task {task_id_str} → status {status_str}")
                return
            else:
                print(f"[ERROR] update-task-status edge function failed: {resp.status_code} - {resp.text}")
                return
                
        except Exception as e:
            print(f"[ERROR] update-task-status edge function exception: {e}")
            return

def _migrate_supabase_schema():
    """Legacy migration function - no longer used. Edge Function architecture complete."""
    dprint("Supabase Migration: Migration to Edge Functions complete. Schema migrations handled externally.")
    return  # No-op - migrations complete

def _run_db_migrations():
    """Runs database migrations (no-op for Supabase as schema is managed externally)."""
    dprint("DB Migrations: Skipping Supabase migrations (table assumed to exist).")
    return

def add_task_to_db(task_payload: dict, task_type_str: str, dependant_on: str | None = None, db_path: str | None = None) -> str:
    """
    Adds a new task to the Supabase database via Edge Function.

    Args:
        task_payload: Task parameters dictionary
        task_type_str: Type of task being created
        dependant_on: Optional dependency task ID
        db_path: Ignored (kept for API compatibility)

    Returns:
        str: The database row ID (UUID) assigned to the task
    """
    # Generate a new UUID for the database row ID
    import uuid
    actual_db_row_id = str(uuid.uuid4())

    # Sanitize payload and get project_id
    params_for_db = task_payload.copy()
    params_for_db.pop("task_type", None)  # Ensure task_type is not duplicated in params
    project_id = task_payload.get("project_id", "default_project_id")

    # Build Edge URL
    edge_url = (
        SUPABASE_EDGE_CREATE_TASK_URL if "SUPABASE_EDGE_CREATE_TASK_URL" in globals() else None
    ) or os.getenv("SUPABASE_EDGE_CREATE_TASK_URL") or (
        f"{SUPABASE_URL.rstrip('/')}/functions/v1/create-task" if SUPABASE_URL else None
    )

    if not edge_url:
        raise ValueError("Edge Function URL for create-task is not configured")

    headers = {"Content-Type": "application/json"}
    if SUPABASE_ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {SUPABASE_ACCESS_TOKEN}"

    # Defensive check: if a dependency is specified, ensure it exists before enqueueing
    if dependant_on:
        max_retries = 3
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                if SUPABASE_CLIENT:
                    resp_exist = SUPABASE_CLIENT.table(PG_TABLE_NAME).select("id").eq("id", dependant_on).single().execute()
                    if not getattr(resp_exist, "data", None):
                        dprint(f"[ERROR][DEBUG_DEPENDENCY_CHAIN] dependant_on not found: {dependant_on}. Refusing to create task of type {task_type_str} with broken dependency.")
                        raise RuntimeError(f"dependant_on {dependant_on} not found")
                    # Successfully verified dependency
                    break
            except Exception as e_depchk:
                error_str = str(e_depchk)
                # Check if it's a "0 rows" error (race condition) and we have retries left
                if "0 rows" in error_str and attempt < max_retries - 1:
                    dprint(f"[RETRY][DEBUG_DEPENDENCY_CHAIN] Dependency {dependant_on} not visible yet (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                else:
                    # Final attempt failed or different error - log warning but continue
                    dprint(f"[WARN][DEBUG_DEPENDENCY_CHAIN] Could not verify dependant_on {dependant_on} existence prior to enqueue: {e_depchk}")

    payload_edge = {
        "task_id": actual_db_row_id,
        "params": params_for_db,
        "task_type": task_type_str,
        "project_id": project_id,
        "dependant_on": dependant_on,
    }

    dprint(f"Supabase Edge call >>> POST {edge_url} payload={str(payload_edge)[:120]}…")

    # Retry logic for transient network failures
    max_create_retries = 3
    base_timeout = 45  # Increased from 30s
    
    for attempt in range(max_create_retries):
        try:
            # Use longer timeout on retries
            current_timeout = base_timeout + (attempt * 15)  # 45s, 60s, 75s
            resp = httpx.post(edge_url, json=payload_edge, headers=headers, timeout=current_timeout)
            break  # Success - exit retry loop
        except httpx.TimeoutException as e:
            if attempt < max_create_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"[RETRY] ⏳ create-task timeout (attempt {attempt + 1}/{max_create_retries}), retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                error_msg = f"Edge Function create-task timed out after {max_create_retries} attempts: {e}"
                print(f"[ERROR] ❌ {error_msg}")
                raise RuntimeError(error_msg)
        except httpx.RequestError as e:
            if attempt < max_create_retries - 1:
                wait_time = 2 ** attempt
                print(f"[RETRY] ⚠️ create-task network error (attempt {attempt + 1}/{max_create_retries}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
                continue
            else:
                error_msg = f"Edge Function create-task request failed after {max_create_retries} attempts: {e}"
                print(f"[ERROR] ❌ {error_msg}")
                raise RuntimeError(error_msg)

    if resp.status_code == 200:
        print(f"Task {actual_db_row_id} (Type: {task_type_str}) queued via Edge Function.")
        
        # Verify task visibility and status to catch race conditions
        if SUPABASE_CLIENT:
            max_verify_retries = 10  # Increased from 5 to handle longer replication delays
            verify_start_time = time.time()
            
            for attempt in range(max_verify_retries):
                try:
                    # Check status, created_at, and project_id for comprehensive debugging
                    verify_resp = SUPABASE_CLIENT.table(PG_TABLE_NAME).select("status, created_at, project_id, task_type").eq("id", actual_db_row_id).single().execute()
                    if verify_resp.data:
                        status = verify_resp.data.get("status")
                        created_at = verify_resp.data.get("created_at")
                        db_project_id = verify_resp.data.get("project_id")
                        db_task_type = verify_resp.data.get("task_type")
                        elapsed = time.time() - verify_start_time
                        
                        dprint(f"[VERIFY] Task {actual_db_row_id} found after {elapsed:.2f}s: status={status}, project_id={db_project_id}, task_type={db_task_type}, created_at={created_at}")
                        
                        if status == STATUS_QUEUED:
                            print(f"[VERIFY] ✅ Task {actual_db_row_id} verified visible and Queued (took {elapsed:.2f}s)")
                            break
                        elif status == STATUS_IN_PROGRESS:
                            print(f"[VERIFY] ⚠️  Task {actual_db_row_id} already In Progress (claimed in {elapsed:.2f}s - unusually fast)")
                            break
                        else:
                            print(f"[VERIFY] ⚠️  Task {actual_db_row_id} has unexpected status '{status}' after {elapsed:.2f}s")
                            break
                    else:
                        dprint(f"[VERIFY] Attempt {attempt+1}/{max_verify_retries}: Task {actual_db_row_id} not visible yet (no data returned)")
                except Exception as e_ver:
                    error_str = str(e_ver)
                    if "0 rows" in error_str:
                        dprint(f"[VERIFY] Attempt {attempt+1}/{max_verify_retries}: Task {actual_db_row_id} not visible yet (0 rows)")
                    else:
                        dprint(f"[VERIFY] Attempt {attempt+1}/{max_verify_retries}: Error verifying task {actual_db_row_id}: {e_ver}")
                
                if attempt < max_verify_retries - 1:
                    time.sleep(1.0)
            else:
                # Verification failed after all retries
                elapsed = time.time() - verify_start_time
                print(f"[WARN] ⚠️  Task {actual_db_row_id} creation confirmed by Edge Function but NOT VISIBLE in DB after {max_verify_retries} attempts ({elapsed:.2f}s)")
                print(f"[WARN] This is likely due to database replication lag. The task IS CREATED and will be processed when visible.")
                dprint(f"[WARN] Task details: type={task_type_str}, project_id={project_id}, dependant_on={dependant_on}")

        return actual_db_row_id
    else:
        error_msg = f"Edge Function create-task failed: {resp.status_code} - {resp.text}"
        print(f"[ERROR] ❌ {error_msg}")
        raise RuntimeError(error_msg)

def poll_task_status(task_id: str, poll_interval_seconds: int = 10, timeout_seconds: int = 1800, db_path: str | None = None) -> str | None:
    """
    Polls Supabase for task completion and returns the output_location.

    Args:
        task_id: Task ID to poll
        poll_interval_seconds: Seconds between polls
        timeout_seconds: Maximum time to wait
        db_path: Ignored (kept for API compatibility)

    Returns:
        Output location string if successful, None otherwise
    """
    print(f"Polling for completion of task {task_id} (timeout: {timeout_seconds}s)...")
    start_time = time.time()
    last_status_print_time = 0

    while True:
        current_time = time.time()
        if current_time - start_time > timeout_seconds:
            print(f"Error: Timeout polling for task {task_id} after {timeout_seconds} seconds.")
            return None

        status = None
        output_location = None

        if not SUPABASE_CLIENT:
            print("[ERROR] Supabase client not initialized. Cannot poll status.")
            time.sleep(poll_interval_seconds)
            continue
        try:
            # Direct table query for polling status
            resp = SUPABASE_CLIENT.table(PG_TABLE_NAME).select("status, output_location").eq("id", task_id).single().execute()
            if resp.data:
                status = resp.data.get("status")
                output_location = resp.data.get("output_location")
        except Exception as e:
            print(f"Supabase error while polling task {task_id}: {e}. Retrying...")

        if status:
            if current_time - last_status_print_time > poll_interval_seconds * 2:
                print(f"Task {task_id}: Status = {status} (Output: {output_location if output_location else 'N/A'})")
                last_status_print_time = current_time

            if status == STATUS_COMPLETE:
                if output_location:
                    print(f"Task {task_id} completed successfully. Output: {output_location}")
                    return output_location
                else:
                    print(f"Error: Task {task_id} is COMPLETE but output_location is missing. Assuming failure.")
                    return None
            elif status == STATUS_FAILED:
                print(f"Error: Task {task_id} failed. Error details: {output_location}")
                return None
            elif status not in [STATUS_QUEUED, STATUS_IN_PROGRESS]:
                print(f"Warning: Task {task_id} has unknown status '{status}'. Treating as error.")
                return None
        else:
            if current_time - last_status_print_time > poll_interval_seconds * 2:
                print(f"Task {task_id}: Not found in DB yet or status pending...")
                last_status_print_time = current_time

        time.sleep(poll_interval_seconds)

# Helper to query DB for a specific task's output (needed by segment handler)
def get_task_output_location_from_db(task_id_to_find: str) -> str | None:
    """
    Queries Supabase for a specific task's output location.

    Args:
        task_id_to_find: Task ID to look up

    Returns:
        Output location string if task is complete, None otherwise
    """
    dprint(f"Querying DB for output location of task: {task_id_to_find}")
    if not SUPABASE_CLIENT:
        print(f"[ERROR] Supabase client not initialized. Cannot query task output {task_id_to_find}")
        return None

    try:
        response = SUPABASE_CLIENT.table(PG_TABLE_NAME)\
            .select("output_location, status")\
            .eq("id", task_id_to_find)\
            .single()\
            .execute()

        if response.data:
            task_details = response.data
            if task_details.get("status") == STATUS_COMPLETE and task_details.get("output_location"):
                return task_details.get("output_location")
            else:
                dprint(f"Task {task_id_to_find} found but not complete or no output_location. Status: {task_details.get('status')}")
                return None
        else:
            dprint(f"Task {task_id_to_find} not found in Supabase.")
            return None
    except Exception as e:
        print(f"Error querying Supabase for task output {task_id_to_find}: {e}")
        traceback.print_exc()
        return None

def get_task_params(task_id: str) -> str | None:
    """Gets the raw params JSON string for a given task ID from Supabase."""
    if not SUPABASE_CLIENT:
        print(f"[ERROR] Supabase client not initialized. Cannot get task params for {task_id}")
        return None

    try:
        resp = SUPABASE_CLIENT.table(PG_TABLE_NAME).select("params").eq("id", task_id).single().execute()
        if resp.data:
            return resp.data.get("params")
        return None
    except Exception as e:
        dprint(f"Error getting task params for {task_id} from Supabase: {e}")
        return None

def get_task_dependency(task_id: str, max_retries: int = 3, retry_delay: float = 0.5) -> str | None:
    """
    Gets the dependency task ID for a given task ID from Supabase.

    Includes retry logic to handle race conditions where a newly created task
    may not be immediately visible in the database.

    Args:
        task_id: Task ID to get dependency for
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Seconds to wait between retries (default: 0.5)

    Returns:
        Dependency task ID or None if no dependency
    """
    if not SUPABASE_CLIENT:
        print(f"[ERROR] Supabase client not initialized. Cannot get task dependency for {task_id}")
        return None

    import time

    for attempt in range(max_retries):
        try:
            response = SUPABASE_CLIENT.table(PG_TABLE_NAME).select("dependant_on").eq("id", task_id).single().execute()
            if response.data:
                return response.data.get("dependant_on")
            return None
        except Exception as e_supabase_dep:
            error_str = str(e_supabase_dep)

            # Check if it's a "0 rows" error (race condition)
            if "0 rows" in error_str and attempt < max_retries - 1:
                dprint(f"[RETRY] Task {task_id} not visible yet (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue

            # Final attempt failed or different error
            dprint(f"Error fetching dependant_on from Supabase for task {task_id}: {e_supabase_dep}")
            return None

    return None

def get_orchestrator_child_tasks(orchestrator_task_id: str) -> dict:
    """
    Gets all child tasks for a given orchestrator task ID from Supabase.
    Returns dict with 'segments', 'stitch', and 'join_clips_segment' lists.
    """
    if not SUPABASE_CLIENT:
        print(f"[ERROR] Supabase client not initialized. Cannot get orchestrator child tasks for {orchestrator_task_id}")
        return {'segments': [], 'stitch': [], 'join_clips_segment': []}

    try:
        # Query for child tasks referencing this orchestrator
        response = SUPABASE_CLIENT.table(PG_TABLE_NAME)\
            .select("id, task_type, status, params, output_location")\
            .contains("params", {"orchestrator_task_id_ref": orchestrator_task_id})\
            .order("created_at", desc=False)\
            .execute()

        segments = []
        stitch = []
        join_clips_segment = []

        if response.data:
            for task in response.data:
                task_data = {
                    'id': task['id'],
                    'task_type': task['task_type'],
                    'status': task['status'],
                    'params': task.get('params', {}),
                    'task_params': task.get('params', {}),  # Alias for compatibility
                    'output_location': task.get('output_location', '')
                }
                if task['task_type'] == 'travel_segment':
                    segments.append(task_data)
                elif task['task_type'] == 'travel_stitch':
                    stitch.append(task_data)
                elif task['task_type'] == 'join_clips_segment':
                    join_clips_segment.append(task_data)

        return {'segments': segments, 'stitch': stitch, 'join_clips_segment': join_clips_segment}

    except Exception as e:
        dprint(f"Error querying Supabase for orchestrator child tasks {orchestrator_task_id}: {e}")
        traceback.print_exc()
        return {'segments': [], 'stitch': [], 'join_clips_segment': []}

def cleanup_duplicate_child_tasks(orchestrator_task_id: str, expected_segments: int) -> dict:
    """
    Detects and removes duplicate child tasks for an orchestrator.
    Returns summary of cleanup actions.
    """
    child_tasks = get_orchestrator_child_tasks(orchestrator_task_id)
    segments = child_tasks['segments']
    stitch_tasks = child_tasks['stitch']
    
    cleanup_summary = {
        'duplicate_segments_removed': 0,
        'duplicate_stitch_removed': 0, 
        'errors': []
    }
    
    try:
        # Remove duplicate segments (keep the oldest for each segment_index)
        segment_by_index = {}
        for segment in segments:
            segment_idx = segment['params'].get('segment_index', -1)
            if segment_idx in segment_by_index:
                # We have a duplicate - keep the older one (first created)
                existing = segment_by_index[segment_idx]
                duplicate_id = segment['id']
                
                dprint(f"[IDEMPOTENCY] Found duplicate segment {segment_idx}: keeping {existing['id']}, removing {duplicate_id}")
                
                # Remove the duplicate
                if _delete_task_by_id(duplicate_id):
                    cleanup_summary['duplicate_segments_removed'] += 1
                else:
                    cleanup_summary['errors'].append(f"Failed to delete duplicate segment {duplicate_id}")
            else:
                segment_by_index[segment_idx] = segment
        
        # Remove duplicate stitch tasks (should only be 1)
        if len(stitch_tasks) > 1:
            # Keep the oldest stitch task, remove others
            stitch_sorted = sorted(stitch_tasks, key=lambda x: x.get('created_at', ''))
            for stitch in stitch_sorted[1:]:  # Remove all but first
                duplicate_id = stitch['id']
                dprint(f"[IDEMPOTENCY] Found duplicate stitch task: removing {duplicate_id}")
                
                if _delete_task_by_id(duplicate_id):
                    cleanup_summary['duplicate_stitch_removed'] += 1
                else:
                    cleanup_summary['errors'].append(f"Failed to delete duplicate stitch {duplicate_id}")
    
    except Exception as e:
        cleanup_summary['errors'].append(f"Cleanup error: {str(e)}")
        dprint(f"Error during duplicate cleanup: {e}")
        traceback.print_exc()
    
    return cleanup_summary

def _delete_task_by_id(task_id: str) -> bool:
    """Helper to delete a task by ID from Supabase. Returns True if successful."""
    if not SUPABASE_CLIENT:
        print(f"[ERROR] Supabase client not initialized. Cannot delete task {task_id}")
        return False

    try:
        response = SUPABASE_CLIENT.table(PG_TABLE_NAME).delete().eq("id", task_id).execute()
        return len(response.data) > 0 if response.data else False
    except Exception as e:
        dprint(f"Error deleting Supabase task {task_id}: {e}")
        return False

def get_predecessor_output_via_edge_function(task_id: str) -> tuple[str | None, str | None]:
    """
    Gets both the predecessor task ID and its output location using Supabase Edge Function.

    Args:
        task_id: Task ID to get predecessor for

    Returns:
        (predecessor_id, output_location) or (None, None) if no dependency or error
    """
    if not SUPABASE_URL or not SUPABASE_ACCESS_TOKEN:
        print("[ERROR] Supabase configuration incomplete. Falling back to direct queries.")
        predecessor_id = get_task_dependency(task_id)
        if predecessor_id:
            output_location = get_task_output_location_from_db(predecessor_id)
            return predecessor_id, output_location
        return None, None

    edge_url = f"{SUPABASE_URL.rstrip('/')}/functions/v1/get-predecessor-output"

    try:
        dprint(f"Calling Edge Function: {edge_url} for task {task_id}")
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {SUPABASE_ACCESS_TOKEN}'
        }

        resp = httpx.post(edge_url, json={"task_id": task_id}, headers=headers, timeout=15)
        dprint(f"Edge Function response status: {resp.status_code}")

        if resp.status_code == 200:
            result = resp.json()
            dprint(f"Edge Function result: {result}")

            if result is None:
                # No dependency
                return None, None

            predecessor_id = result.get("predecessor_id")
            output_location = result.get("output_location")
            return predecessor_id, output_location

        elif resp.status_code == 404:
            dprint(f"Edge Function: Task {task_id} not found")
            return None, None
        else:
            dprint(f"Edge Function returned {resp.status_code}: {resp.text}. Falling back to direct queries.")
            # Fall back to separate calls
            predecessor_id = get_task_dependency(task_id)
            if predecessor_id:
                output_location = get_task_output_location_from_db(predecessor_id)
                return predecessor_id, output_location
            return None, None

    except Exception as e_edge:
        dprint(f"Edge Function call failed: {e_edge}. Falling back to direct queries.")
        # Fall back to separate calls
        predecessor_id = get_task_dependency(task_id)
        if predecessor_id:
            output_location = get_task_output_location_from_db(predecessor_id)
            return predecessor_id, output_location
        return None, None


def get_completed_segment_outputs_for_stitch(run_id: str, project_id: str | None = None) -> list:
    """Gets completed travel_segment outputs for a given run_id for stitching from Supabase."""
    if not SUPABASE_URL or not SUPABASE_ACCESS_TOKEN:
        print("[ERROR] Supabase configuration incomplete. Cannot get completed segments.")
        return []

    edge_url = f"{SUPABASE_URL.rstrip('/')}/functions/v1/get-completed-segments"
    try:
        dprint(f"Calling Edge Function: {edge_url} for run_id {run_id}, project_id {project_id}")
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {SUPABASE_ACCESS_TOKEN}'
        }
        payload = {"run_id": run_id}
        if project_id:
            payload["project_id"] = project_id

        resp = httpx.post(edge_url, json=payload, headers=headers, timeout=15)
        if resp.status_code == 200:
            results = resp.json()
            sorted_results = sorted(results, key=lambda x: x['segment_index'])
            return [(r['segment_index'], r['output_location']) for r in sorted_results]
        else:
            dprint(f"Edge Function returned {resp.status_code}: {resp.text}. Falling back to direct query.")
    except Exception as e:
        dprint(f"Edge Function failed: {e}. Falling back to direct query.")

    # Fallback to direct query
    if not SUPABASE_CLIENT:
        print("[ERROR] Supabase client not initialized. Cannot fall back to direct query.")
        return []

    try:
        # First, let's debug by getting ALL completed tasks to see what's there
        debug_resp = SUPABASE_CLIENT.table(PG_TABLE_NAME).select("id, task_type, status, params, output_location")\
            .eq("status", STATUS_COMPLETE).execute()

        dprint(f"[DEBUG_STITCH] Looking for run_id: '{run_id}' (type: {type(run_id)})")
        dprint(f"[DEBUG_STITCH] Total completed tasks in DB: {len(debug_resp.data) if debug_resp.data else 0}")

        travel_segment_count = 0
        matching_run_id_count = 0

        if debug_resp.data:
            for task in debug_resp.data:
                task_type = task.get("task_type", "")
                if task_type == "travel_segment":
                    travel_segment_count += 1
                    params_raw = task.get("params", {})
                    try:
                        params_obj = params_raw if isinstance(params_raw, dict) else json.loads(params_raw)
                        task_run_id = params_obj.get("orchestrator_run_id")
                        dprint(f"[DEBUG_STITCH] Found travel_segment task {task.get('id')}: orchestrator_run_id='{task_run_id}' (type: {type(task_run_id)}), segment_index={params_obj.get('segment_index')}, output_location={task.get('output_location', 'None')}")

                        if str(task_run_id) == str(run_id):
                            matching_run_id_count += 1
                            dprint(f"[DEBUG_STITCH] MATCH FOUND! Task {task.get('id')} matches run_id {run_id}")
                    except Exception as e_debug:
                        dprint(f"[DEBUG_STITCH] Error parsing params for task {task.get('id')}: {e_debug}")

        dprint(f"[DEBUG_STITCH] Travel_segment tasks found: {travel_segment_count}")
        dprint(f"[DEBUG_STITCH] Tasks matching run_id '{run_id}': {matching_run_id_count}")

        # Now do the actual query
        sel_resp = SUPABASE_CLIENT.table(PG_TABLE_NAME).select("params, output_location")\
            .eq("task_type", "travel_segment").eq("status", STATUS_COMPLETE).execute()

        results = []
        if sel_resp.data:
            for i, row in enumerate(sel_resp.data):
                params_raw = row.get("params")
                if params_raw is None:
                    continue
                try:
                    params_obj = params_raw if isinstance(params_raw, dict) else json.loads(params_raw)
                except Exception as e:
                    continue

                row_run_id = params_obj.get("orchestrator_run_id")

                # Use string comparison to handle type mismatches
                if str(row_run_id) == str(run_id):
                    seg_idx = params_obj.get("segment_index")
                    output_loc = row.get("output_location")
                    results.append((seg_idx, output_loc))
                    dprint(f"[DEBUG_STITCH] Added to results: segment_index={seg_idx}, output_location={output_loc}")

        sorted_results = sorted(results, key=lambda x: x[0] if x[0] is not None else 0)
        dprint(f"[DEBUG_STITCH] Final sorted results: {sorted_results}")
        return sorted_results
    except Exception as e_sel:
        dprint(f"Stitch Supabase: Direct select failed: {e_sel}")
        traceback.print_exc()
        return []

def get_initial_task_counts() -> tuple[int, int] | None:
    """
    Gets the total and queued task counts (no longer supported - returns None).
    This function is kept for API compatibility.
    """
    return None

def get_abs_path_from_db_path(db_path: str, dprint) -> Path | None:
    """
    Helper to resolve a path from the DB to a usable absolute path.
    Assumes paths from Supabase are already absolute or valid URLs.
    """
    if not db_path:
        return None

    # Path from DB is assumed to be absolute (Supabase) or a URL
    resolved_path = Path(db_path).resolve()

    if resolved_path and resolved_path.exists():
        return resolved_path
    else:
        dprint(f"Warning: Resolved path '{resolved_path}' from DB path '{db_path}' does not exist.")
        return None

def mark_task_failed_supabase(task_id_str, error_message):
    """Marks a task as Failed with an error message using direct database update."""
    dprint(f"Marking task {task_id_str} as Failed with message: {error_message}")
    if not SUPABASE_CLIENT:
        print("[ERROR] Supabase client not initialized. Cannot mark task failed.")
        return
    
    # Use the standard update function which now uses direct database updates for non-COMPLETE statuses
    update_task_status_supabase(task_id_str, STATUS_FAILED, error_message) 
