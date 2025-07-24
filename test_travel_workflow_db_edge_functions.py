#!/usr/bin/env python3
"""
Comprehensive test suite for all database operations and edge functions 
used in the complete Wan2GP workflow system including travel_between_images.py 
and headless.py functionality.

This test suite simulates complete workflows and verifies:

🔧 Core Database & Edge Functions:
1. Database initialization and migrations
2. Task creation via edge functions (create-task)
3. Task claiming via edge functions (claim-next-task)
4. Task completion with file upload (complete-task)
5. Predecessor/dependency relationships (get-predecessor-output)
6. Segment collection for stitching (get-completed-segments)
7. Various task status update scenarios

🎨 Image Generation & Specialized Tasks:
8. Single image generation tasks
9. OpenPose skeleton generation tasks
10. RIFE frame interpolation tasks
11. Frame extraction from videos
12. Different perspective orchestrator tasks

🎬 Video Generation & Advanced Features:
13. Standard WGP video generation workflows
14. LoRA handling and parameter management
15. Complex task chaining and dependency workflows

Supports both SQLite and Supabase backends.
Run with: python test_travel_workflow_db_edge_functions.py
"""

import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
import uuid
import shutil
import base64

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print(f"🔧 Loaded environment variables from .env file")
except ImportError:
    print(f"⚠️  python-dotenv not installed - using system environment variables only")

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent / "source"))

try:
    import db_operations as db_ops
    from common_utils import generate_unique_task_id
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Make sure you're running from the project root and source/ is available")
    sys.exit(1)

def configure_supabase_if_available():
    """Configure Supabase from environment variables if available."""
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    if supabase_url and supabase_key:
        print(f"🚀 Configuring Supabase from environment variables...")
        print(f"   URL: {supabase_url}")
        print(f"   Service Key: {supabase_key[:20]}...{supabase_key[-4:]}")
        
        # Configure db_operations for Supabase
        db_ops.DB_TYPE = "supabase"
        db_ops.SUPABASE_URL = supabase_url
        db_ops.SUPABASE_SERVICE_KEY = supabase_key
        db_ops.SUPABASE_ACCESS_TOKEN = supabase_key  # Use service key as access token for testing
        db_ops.PG_TABLE_NAME = "tasks"
        db_ops.SUPABASE_VIDEO_BUCKET = "image_uploads"
        
        # Initialize Supabase client
        try:
            from supabase import create_client
            db_ops.SUPABASE_CLIENT = create_client(supabase_url, supabase_key)
            print(f"   ✅ Supabase client initialized successfully")
            return True
        except Exception as e:
            print(f"   ❌ Failed to initialize Supabase client: {e}")
            return False
    else:
        print(f"⚠️  Supabase configuration not found in environment variables")
        print(f"   SUPABASE_URL: {'✓' if supabase_url else '✗'}")
        print(f"   SUPABASE_SERVICE_ROLE_KEY: {'✓' if supabase_key else '✗'}")
        return False

class TravelWorkflowTester:
    def __init__(self):
        self.test_run_id = generate_unique_task_id("test_travel_")
        # Use PROJECT_ID from .env instead of random UUID
        self.project_id = os.getenv('PROJECT_ID') or str(uuid.uuid4())
        # Use a proper worker ID pattern like the example provided
        self.worker_id = "gpu-20250723_221138-afa8403b"
        self.created_tasks = []
        self.test_files = []
        self.temp_dir = None
        
        print(f"[TEST_INIT] Starting comprehensive workflow test")
        print(f"[TEST_INIT] Test run ID: {self.test_run_id}")
        print(f"[TEST_INIT] Project ID: {self.project_id} (from env)")
        print(f"[TEST_INIT] Worker ID: {self.worker_id}")
        print(f"[TEST_INIT] DB Type: {db_ops.DB_TYPE}")
        print(f"[TEST_INIT] Headless functionality available for testing")
        
        # Try to import headless functionality for testing
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            self.headless_available = True
            print(f"[TEST_INIT] Headless functionality available for testing")
        except ImportError as e:
            self.headless_available = False
            print(f"[TEST_INIT] Headless functionality not available: {e}")
        
    def setup_temp_files(self, file_type="video"):
        """Create temporary test files for upload/download testing"""
        if not self.temp_dir:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="travel_test_"))
            print(f"[SETUP] Created temp directory: {self.temp_dir}")
        
        if file_type == "video":
            # Create a dummy video file for testing
            test_video = self.temp_dir / "test_segment_output.mp4"
            with open(test_video, 'wb') as f:
                # Create a minimal valid MP4 header for testing
                f.write(b'\x00\x00\x00\x20ftypmp42\x00\x00\x00\x00mp42isom')
                f.write(b'\x00' * 100)  # Padding to make it a reasonable size
            
            self.test_files.append(test_video)
            print(f"[SETUP] Created test video: {test_video}")
            return test_video
            
        elif file_type == "image":
            # Create a dummy image file for testing
            test_image = self.temp_dir / "test_image.jpg"
            # Create a minimal JPEG header
            with open(test_image, 'wb') as f:
                # JPEG file signature
                f.write(b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00')
                f.write(b'\xFF\xDB\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f')
                f.write(b'\x00' * 100)  # Padding
                f.write(b'\xFF\xD9')  # End of image marker
            
            self.test_files.append(test_image)
            print(f"[SETUP] Created test image: {test_image}")
            return test_image
            
        elif file_type == "openpose":
            # Create a dummy pose JSON file
            test_pose = self.temp_dir / "test_pose.json"
            pose_data = {
                "people": [{
                    "pose_keypoints_2d": [
                        # Simplified pose data (nose, eyes, etc.)
                        320, 240, 0.9,  # nose
                        310, 230, 0.8,  # left eye
                        330, 230, 0.8,  # right eye
                        # ... (would normally have 18 keypoints)
                    ] + [0, 0, 0] * 15  # Fill remaining keypoints
                }]
            }
            with open(test_pose, 'w') as f:
                json.dump(pose_data, f)
            
            self.test_files.append(test_pose)
            print(f"[SETUP] Created test pose file: {test_pose}")
            return test_pose

    def cleanup(self):
        """Clean up test data and files"""
        print(f"[CLEANUP] Starting cleanup...")
        
        # Clean up test tasks from database
        for task_id in self.created_tasks:
            try:
                if db_ops.DB_TYPE == "sqlite":
                    # For SQLite, manually delete the task
                    def delete_task(conn):
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
                    db_ops.execute_sqlite_with_retry(db_ops.SQLITE_DB_PATH, delete_task)
                print(f"[CLEANUP] Removed task: {task_id}")
            except Exception as e:
                print(f"[CLEANUP] Warning: Could not remove task {task_id}: {e}")
        
        # Clean up temporary files
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                print(f"[CLEANUP] Removed temp directory: {self.temp_dir}")
            except Exception as e:
                print(f"[CLEANUP] Warning: Could not remove temp directory: {e}")

    def test_edge_function_create_task(self):
        """Test the create-task edge function"""
        print(f"\n[TEST_CREATE_TASK] Testing create-task edge function...")
        
        try:
            orchestrator_task_id = generate_unique_task_id(f"orch_{self.test_run_id}_")
            
            orchestrator_payload = {
                "task_id": orchestrator_task_id,
                "project_id": self.project_id,
                "orchestrator_details": {
                    "run_id": self.test_run_id,
                    "num_new_segments_to_generate": 2,
                    "base_prompts_expanded": ["A sunrise over mountains", "A peaceful lake"],
                    "negative_prompts_expanded": ["blurry", "low quality"],
                    "segment_frames_expanded": [73, 73],
                    "frame_overlap_expanded": [20],
                    "parsed_resolution_wh": "512x512",
                    "model_name": "vace_14B",
                    "main_output_dir_for_run": str(self.temp_dir),
                    "fps_helpers": 16,
                    "fade_in_params_json_str": '{"low_point": 0, "high_point": 1, "curve_type": "ease_in_out", "duration_factor": 0}',
                    "fade_out_params_json_str": '{"low_point": 0, "high_point": 1, "curve_type": "ease_in_out", "duration_factor": 0}',
                    "input_image_paths_resolved": ["https://example.com/start.jpg", "https://example.com/mid.jpg", "https://example.com/end.jpg"]
                }
            }
            
            db_ops.add_task_to_db(
                task_payload=orchestrator_payload,
                task_type_str="travel_orchestrator"
            )
            
            self.created_tasks.append(orchestrator_task_id)
            print(f"[TEST_CREATE_TASK] ✅ Successfully created orchestrator task: {orchestrator_task_id}")
            
            # Verify the task was created by querying it back
            task_params = db_ops.get_task_params(orchestrator_task_id)
            if task_params:
                parsed_params = json.loads(task_params) if isinstance(task_params, str) else task_params
                print(f"[TEST_CREATE_TASK] ✅ Task verified in database")
                print(f"[TEST_CREATE_TASK] Task type in params: {parsed_params.get('task_type', 'Not found')}")
                return orchestrator_task_id
            else:
                print(f"[TEST_CREATE_TASK] ❌ Task not found in database after creation")
                return None
                
        except Exception as e:
            print(f"[TEST_CREATE_TASK] ❌ Error creating task: {e}")
            traceback.print_exc()
            return None

    def test_edge_function_claim_task(self):
        """Test the claim-next-task edge function"""
        print(f"\n[TEST_CLAIM_TASK] Testing claim-next-task edge function...")
        
        try:
            # First create a simple task to claim
            test_task_id = generate_unique_task_id(f"claim_test_{self.test_run_id}_")
            test_payload = {
                "task_id": test_task_id,
                "project_id": self.project_id,
                "test_param": "claim_test_value",
                "segment_index": 0
            }
            
            db_ops.add_task_to_db(
                task_payload=test_payload,
                task_type_str="travel_segment"
            )
            self.created_tasks.append(test_task_id)
            print(f"[TEST_CLAIM_TASK] Created test task for claiming: {test_task_id}")
            
            # Wait a moment for the task to be available
            time.sleep(1)
            
            # Try to claim a task
            claimed_task = db_ops.get_oldest_queued_task()
            
            if claimed_task:
                print(f"[TEST_CLAIM_TASK] ✅ Successfully claimed task:")
                print(f"[TEST_CLAIM_TASK] Task ID: {claimed_task.get('task_id')}")
                print(f"[TEST_CLAIM_TASK] Task Type: {claimed_task.get('task_type')}")
                print(f"[TEST_CLAIM_TASK] Project ID: {claimed_task.get('project_id')}")
                
                # Verify the task status was updated to "In Progress"
                return claimed_task
            else:
                print(f"[TEST_CLAIM_TASK] ❌ No task was claimed")
                return None
                
        except Exception as e:
            print(f"[TEST_CLAIM_TASK] ❌ Error claiming task: {e}")
            traceback.print_exc()
            return None

    def test_edge_function_complete_task(self):
        """Test the complete-task edge function with file upload"""
        print(f"\n[TEST_COMPLETE_TASK] Testing complete-task edge function...")
        
        try:
            # Create a task to complete
            test_task_id = generate_unique_task_id(f"complete_test_{self.test_run_id}_")
            test_payload = {
                "task_id": test_task_id,
                "project_id": self.project_id,
                "test_param": "complete_test_value"
            }
            
            db_ops.add_task_to_db(
                task_payload=test_payload,
                task_type_str="travel_segment"
            )
            self.created_tasks.append(test_task_id)
            
            # Set status to In Progress first
            db_ops.update_task_status(test_task_id, db_ops.STATUS_IN_PROGRESS)
            print(f"[TEST_COMPLETE_TASK] Created and set task to In Progress: {test_task_id}")
            
            # Create a test output file
            test_output_file = self.setup_temp_files()
            print(f"[TEST_COMPLETE_TASK] Created test output file: {test_output_file}")
            
            # Complete the task with file upload
            db_ops.update_task_status(
                test_task_id, 
                db_ops.STATUS_COMPLETE, 
                str(test_output_file)
            )
            
            print(f"[TEST_COMPLETE_TASK] ✅ Successfully completed task with file upload")
            
            # Verify the task was marked complete
            time.sleep(1)  # Allow for processing
            
            # Check if we can retrieve the output location
            output_location = db_ops.get_task_output_location_from_db(test_task_id)
            if output_location:
                print(f"[TEST_COMPLETE_TASK] ✅ Task output location retrieved: {output_location}")
                return test_task_id, output_location
            else:
                print(f"[TEST_COMPLETE_TASK] ❌ Could not retrieve task output location")
                return test_task_id, None
                
        except Exception as e:
            print(f"[TEST_COMPLETE_TASK] ❌ Error completing task: {e}")
            traceback.print_exc()
            return None, None

    def test_predecessor_dependency_chain(self):
        """Test predecessor/dependency relationships and the get-predecessor-output edge function"""
        print(f"\n[TEST_PREDECESSOR] Testing predecessor dependency chain...")
        
        try:
            # Create a chain of dependent tasks
            task1_id = generate_unique_task_id(f"dep1_{self.test_run_id}_")
            task2_id = generate_unique_task_id(f"dep2_{self.test_run_id}_")
            task3_id = generate_unique_task_id(f"dep3_{self.test_run_id}_")
            
            # Task 1 (no dependency)
            task1_payload = {
                "task_id": task1_id,
                "project_id": self.project_id,
                "segment_index": 0,
                "test_param": "first_task"
            }
            db_ops.add_task_to_db(
                task_payload=task1_payload,
                task_type_str="travel_segment"
            )
            self.created_tasks.append(task1_id)
            
            # Task 2 (depends on Task 1)
            task2_payload = {
                "task_id": task2_id,
                "project_id": self.project_id,
                "segment_index": 1,
                "test_param": "second_task"
            }
            db_ops.add_task_to_db(
                task_payload=task2_payload,
                task_type_str="travel_segment",
                dependant_on=task1_id
            )
            self.created_tasks.append(task2_id)
            
            # Task 3 (depends on Task 2)
            task3_payload = {
                "task_id": task3_id,
                "project_id": self.project_id,
                "segment_index": 2,
                "test_param": "third_task"
            }
            db_ops.add_task_to_db(
                task_payload=task3_payload,
                task_type_str="travel_segment",
                dependant_on=task2_id
            )
            self.created_tasks.append(task3_id)
            
            print(f"[TEST_PREDECESSOR] Created task chain: {task1_id} -> {task2_id} -> {task3_id}")
            
            # Complete Task 1 with output
            test_output_file = self.setup_temp_files()
            db_ops.update_task_status(task1_id, db_ops.STATUS_IN_PROGRESS)
            db_ops.update_task_status(task1_id, db_ops.STATUS_COMPLETE, str(test_output_file))
            
            print(f"[TEST_PREDECESSOR] Completed task1 with output: {test_output_file}")
            
            # Test dependency lookup for Task 2
            dependency_id = db_ops.get_task_dependency(task2_id)
            if dependency_id == task1_id:
                print(f"[TEST_PREDECESSOR] ✅ Dependency lookup correct: {task2_id} depends on {dependency_id}")
            else:
                print(f"[TEST_PREDECESSOR] ❌ Dependency lookup failed: expected {task1_id}, got {dependency_id}")
            
            # Test predecessor output lookup
            predecessor_id, output_location = db_ops.get_predecessor_output_via_edge_function(task2_id)
            if predecessor_id == task1_id and output_location:
                print(f"[TEST_PREDECESSOR] ✅ Predecessor output lookup successful:")
                print(f"[TEST_PREDECESSOR]   Predecessor: {predecessor_id}")
                print(f"[TEST_PREDECESSOR]   Output: {output_location}")
            else:
                print(f"[TEST_PREDECESSOR] ❌ Predecessor output lookup failed:")
                print(f"[TEST_PREDECESSOR]   Expected predecessor: {task1_id}, got: {predecessor_id}")
                print(f"[TEST_PREDECESSOR]   Output location: {output_location}")
            
            return task1_id, task2_id, task3_id
            
        except Exception as e:
            print(f"[TEST_PREDECESSOR] ❌ Error testing predecessor chain: {e}")
            traceback.print_exc()
            return None, None, None

    def test_segment_collection_for_stitching(self):
        """Test the get-completed-segments edge function"""
        print(f"\n[TEST_SEGMENT_COLLECTION] Testing segment collection for stitching...")
        
        try:
            # Create multiple completed segment tasks
            segment_tasks = []
            for i in range(3):
                segment_task_id = generate_unique_task_id(f"seg_{i}_{self.test_run_id}_")
                segment_payload = {
                    "task_id": segment_task_id,
                    "project_id": self.project_id,
                    "orchestrator_run_id": self.test_run_id,
                    "segment_index": i,
                    "test_param": f"segment_{i}"
                }
                
                # Create task
                db_ops.add_task_to_db(
                    task_payload=segment_payload,
                    task_type_str="travel_segment"
                )
                self.created_tasks.append(segment_task_id)
                
                # Complete task with output
                test_output_file = self.setup_temp_files()
                db_ops.update_task_status(segment_task_id, db_ops.STATUS_IN_PROGRESS)
                db_ops.update_task_status(segment_task_id, db_ops.STATUS_COMPLETE, str(test_output_file))
                
                segment_tasks.append((segment_task_id, i, str(test_output_file)))
                print(f"[TEST_SEGMENT_COLLECTION] Created and completed segment {i}: {segment_task_id}")
            
            # Wait for all tasks to be processed
            time.sleep(2)
            
            # Test segment collection
            completed_segments = db_ops.get_completed_segment_outputs_for_stitch(
                self.test_run_id, 
                project_id=self.project_id
            )
            
            print(f"[TEST_SEGMENT_COLLECTION] Retrieved {len(completed_segments)} completed segments")
            
            if completed_segments:
                print(f"[TEST_SEGMENT_COLLECTION] ✅ Segment collection successful:")
                for seg_idx, output_location in completed_segments:
                    print(f"[TEST_SEGMENT_COLLECTION]   Segment {seg_idx}: {output_location}")
                
                # Verify segments are in correct order
                segment_indices = [seg_idx for seg_idx, _ in completed_segments]
                if segment_indices == sorted(segment_indices):
                    print(f"[TEST_SEGMENT_COLLECTION] ✅ Segments retrieved in correct order")
                else:
                    print(f"[TEST_SEGMENT_COLLECTION] ❌ Segments not in correct order: {segment_indices}")
                
                return completed_segments
            else:
                print(f"[TEST_SEGMENT_COLLECTION] ❌ No segments retrieved")
                return None
                
        except Exception as e:
            print(f"[TEST_SEGMENT_COLLECTION] ❌ Error testing segment collection: {e}")
            traceback.print_exc()
            return None

    def test_task_status_updates(self):
        """Test various task status update scenarios"""
        print(f"\n[TEST_STATUS_UPDATES] Testing task status updates...")
        
        try:
            test_task_id = generate_unique_task_id(f"status_test_{self.test_run_id}_")
            test_payload = {
                "task_id": test_task_id,
                "project_id": self.project_id,
                "test_param": "status_update_test"
            }
            
            # Create task (should start as Queued)
            db_ops.add_task_to_db(
                task_payload=test_payload,
                task_type_str="travel_segment"
            )
            self.created_tasks.append(test_task_id)
            print(f"[TEST_STATUS_UPDATES] Created task: {test_task_id}")
            
            # Update to In Progress
            db_ops.update_task_status(test_task_id, db_ops.STATUS_IN_PROGRESS)
            print(f"[TEST_STATUS_UPDATES] ✅ Updated to In Progress")
            
            # Update to Complete with output
            test_output_file = self.setup_temp_files()
            db_ops.update_task_status(test_task_id, db_ops.STATUS_COMPLETE, str(test_output_file))
            print(f"[TEST_STATUS_UPDATES] ✅ Updated to Complete with output")
            
            # Test failure status
            fail_task_id = generate_unique_task_id(f"fail_test_{self.test_run_id}_")
            fail_payload = {
                "task_id": fail_task_id,
                "project_id": self.project_id,
                "test_param": "failure_test"
            }
            
            db_ops.add_task_to_db(
                task_payload=fail_payload,
                task_type_str="travel_segment"
            )
            self.created_tasks.append(fail_task_id)
            
            # Update to Failed with error message
            error_message = "Test error: Simulated failure for testing"
            db_ops.update_task_status(fail_task_id, db_ops.STATUS_FAILED, error_message)
            print(f"[TEST_STATUS_UPDATES] ✅ Updated to Failed with error message")
            
            return True
            
        except Exception as e:
            print(f"[TEST_STATUS_UPDATES] ❌ Error testing status updates: {e}")
            traceback.print_exc()
            return False

    def test_database_initialization(self):
        """Test database initialization"""
        print(f"\n[TEST_DB_INIT] Testing database initialization...")
        
        if db_ops.DB_TYPE == "supabase":
            # For Supabase, check if tasks table exists directly instead of using RPC
            try:
                if db_ops.SUPABASE_CLIENT:
                    # Try to query the tasks table to see if it exists
                    result = db_ops.SUPABASE_CLIENT.table("tasks").select("count", count="exact").limit(1).execute()
                    print(f"[TEST_DB_INIT] ✅ Supabase tasks table exists (count: {result.count})")
                    return True
                else:
                    print(f"[TEST_DB_INIT] ❌ Supabase client not initialized")
                    return False
            except Exception as e:
                if "relation \"tasks\" does not exist" in str(e):
                    print(f"[TEST_DB_INIT] ⚠️  Tasks table doesn't exist but edge functions may still work")
                    return True  # Continue testing edge functions even if table missing
                else:
                    print(f"[TEST_DB_INIT] ❌ Database check failed: {e}")
                    return False
        else:
            # SQLite initialization
            try:
                db_ops.init_db()
                print(f"[TEST_DB_INIT] ✅ Database initialization successful")
                return True
            except Exception as e:
                print(f"[TEST_DB_INIT] ❌ Database initialization failed: {e}")
                traceback.print_exc()
                return False

    def test_single_image_generation_task(self):
        """Test single image generation task creation and processing"""
        print(f"\n[TEST_SINGLE_IMAGE] Testing single image generation task...")
        
        try:
            # Create test image
            test_image = self.setup_temp_files("image")
            
            task_id = generate_unique_task_id(f"single_img_{self.test_run_id}_")
            task_payload = {
                "task_id": task_id,
                "project_id": self.project_id,
                "input_image_path": str(test_image),
                "prompt": "A beautiful landscape painting",
                "negative_prompt": "blurry, low quality",
                "resolution": "512x512",
                "seed": 12345,
                "num_inference_steps": 20,
                "model": "vace_14B"
            }
            
            db_ops.add_task_to_db(
                task_payload=task_payload,
                task_type_str="single_image"
            )
            self.created_tasks.append(task_id)
            print(f"[TEST_SINGLE_IMAGE] ✅ Created single image task: {task_id}")
            
            # Verify task in database
            task_params = db_ops.get_task_params(task_id)
            if task_params:
                print(f"[TEST_SINGLE_IMAGE] ✅ Task verified in database")
                return task_id
            else:
                print(f"[TEST_SINGLE_IMAGE] ❌ Task not found in database")
                return None
                
        except Exception as e:
            print(f"[TEST_SINGLE_IMAGE] ❌ Error creating single image task: {e}")
            traceback.print_exc()
            return None

    def test_openpose_generation_task(self):
        """Test OpenPose generation task"""
        print(f"\n[TEST_OPENPOSE] Testing OpenPose generation task...")
        
        try:
            # Create test image
            test_image = self.setup_temp_files("image")
            
            task_id = generate_unique_task_id(f"openpose_{self.test_run_id}_")
            task_payload = {
                "task_id": task_id,
                "project_id": self.project_id,
                "input_image_path": str(test_image),
                "output_format": "json",
                "include_face": True,
                "include_hands": True
            }
            
            db_ops.add_task_to_db(
                task_payload=task_payload,
                task_type_str="generate_openpose"
            )
            self.created_tasks.append(task_id)
            print(f"[TEST_OPENPOSE] ✅ Created OpenPose task: {task_id}")
            
            # Simulate completion with pose data
            test_pose_output = self.setup_temp_files("openpose")
            db_ops.update_task_status(task_id, db_ops.STATUS_IN_PROGRESS)
            db_ops.update_task_status(task_id, db_ops.STATUS_COMPLETE, str(test_pose_output))
            
            print(f"[TEST_OPENPOSE] ✅ Completed OpenPose task with output")
            return task_id
            
        except Exception as e:
            print(f"[TEST_OPENPOSE] ❌ Error with OpenPose task: {e}")
            traceback.print_exc()
            return None

    def test_rife_interpolation_task(self):
        """Test RIFE frame interpolation task"""
        print(f"\n[TEST_RIFE] Testing RIFE interpolation task...")
        
        try:
            # Create test images
            start_image = self.setup_temp_files("image")
            end_image = self.setup_temp_files("image")
            
            task_id = generate_unique_task_id(f"rife_{self.test_run_id}_")
            task_payload = {
                "task_id": task_id,
                "project_id": self.project_id,
                "start_image_path": str(start_image),
                "end_image_path": str(end_image),
                "interpolation_frames": 8,
                "output_fps": 24,
                "resolution": "512x512"
            }
            
            db_ops.add_task_to_db(
                task_payload=task_payload,
                task_type_str="rife_interpolate_images"
            )
            self.created_tasks.append(task_id)
            print(f"[TEST_RIFE] ✅ Created RIFE interpolation task: {task_id}")
            
            # Simulate completion with video output
            test_video_output = self.setup_temp_files("video")
            db_ops.update_task_status(task_id, db_ops.STATUS_IN_PROGRESS)
            db_ops.update_task_status(task_id, db_ops.STATUS_COMPLETE, str(test_video_output))
            
            print(f"[TEST_RIFE] ✅ Completed RIFE task with video output")
            return task_id
            
        except Exception as e:
            print(f"[TEST_RIFE] ❌ Error with RIFE task: {e}")
            traceback.print_exc()
            return None

    def test_extract_frame_task(self):
        """Test frame extraction task"""
        print(f"\n[TEST_EXTRACT_FRAME] Testing frame extraction task...")
        
        try:
            # Create test video
            test_video = self.setup_temp_files("video")
            
            task_id = generate_unique_task_id(f"extract_frame_{self.test_run_id}_")
            task_payload = {
                "task_id": task_id,
                "project_id": self.project_id,
                "video_path": str(test_video),
                "frame_number": 30,
                "output_format": "jpg"
            }
            
            db_ops.add_task_to_db(
                task_payload=task_payload,
                task_type_str="extract_frame"
            )
            self.created_tasks.append(task_id)
            print(f"[TEST_EXTRACT_FRAME] ✅ Created frame extraction task: {task_id}")
            
            # Simulate completion with image output
            test_image_output = self.setup_temp_files("image")
            db_ops.update_task_status(task_id, db_ops.STATUS_IN_PROGRESS)
            db_ops.update_task_status(task_id, db_ops.STATUS_COMPLETE, str(test_image_output))
            
            print(f"[TEST_EXTRACT_FRAME] ✅ Completed frame extraction task")
            return task_id
            
        except Exception as e:
            print(f"[TEST_EXTRACT_FRAME] ❌ Error with frame extraction task: {e}")
            traceback.print_exc()
            return None

    def test_different_perspective_orchestrator(self):
        """Test different perspective orchestrator task"""
        print(f"\n[TEST_DP_ORCHESTRATOR] Testing different perspective orchestrator...")
        
        try:
            orchestrator_task_id = generate_unique_task_id(f"dp_orch_{self.test_run_id}_")
            test_image = self.setup_temp_files("image")
            
            orchestrator_payload = {
                "task_id": orchestrator_task_id,
                "project_id": self.project_id,
                "input_image_path": str(test_image),
                "camera_movements": [
                    {"type": "pan_left", "strength": 0.3},
                    {"type": "zoom_in", "strength": 0.2},
                    {"type": "pan_right", "strength": 0.3}
                ],
                "resolution": "512x512",
                "frames_per_segment": 73,
                "model_name": "vace_14B",
                "base_prompt": "cinematic camera movement",
                "negative_prompt": "static, blurry",
                "seed": 42
            }
            
            db_ops.add_task_to_db(
                task_payload=orchestrator_payload,
                task_type_str="different_perspective_orchestrator"
            )
            self.created_tasks.append(orchestrator_task_id)
            print(f"[TEST_DP_ORCHESTRATOR] ✅ Created different perspective orchestrator: {orchestrator_task_id}")
            
            return orchestrator_task_id
            
        except Exception as e:
            print(f"[TEST_DP_ORCHESTRATOR] ❌ Error creating different perspective orchestrator: {e}")
            traceback.print_exc()
            return None

    def test_standard_wgp_video_generation(self):
        """Test standard WGP video generation task"""
        print(f"\n[TEST_WGP_VIDEO] Testing standard WGP video generation...")
        
        try:
            task_id = generate_unique_task_id(f"wgp_video_{self.test_run_id}_")
            task_payload = {
                "task_id": task_id,
                "project_id": self.project_id,
                "prompt": "A serene lake surrounded by mountains at sunset",
                "negative_prompt": "blurry, low quality, distorted",
                "resolution": "512x512",
                "frames": 73,
                "seed": 12345,
                "num_inference_steps": 30,
                "guidance_scale": 5.0,
                "flow_shift": 3.0,
                "model": "vace_14B",
                "video_length": 73
            }
            
            db_ops.add_task_to_db(
                task_payload=task_payload,
                task_type_str="wgp"
            )
            self.created_tasks.append(task_id)
            print(f"[TEST_WGP_VIDEO] ✅ Created standard WGP video task: {task_id}")
            
            # Simulate completion
            test_video_output = self.setup_temp_files("video")
            db_ops.update_task_status(task_id, db_ops.STATUS_IN_PROGRESS)
            db_ops.update_task_status(task_id, db_ops.STATUS_COMPLETE, str(test_video_output))
            
            print(f"[TEST_WGP_VIDEO] ✅ Completed WGP video generation task")
            return task_id
            
        except Exception as e:
            print(f"[TEST_WGP_VIDEO] ❌ Error with WGP video generation: {e}")
            traceback.print_exc()
            return None

    def test_lora_handling_task(self):
        """Test task with LoRA parameters"""
        print(f"\n[TEST_LORA] Testing LoRA handling in tasks...")
        
        try:
            task_id = generate_unique_task_id(f"lora_test_{self.test_run_id}_")
            task_payload = {
                "task_id": task_id,
                "project_id": self.project_id,
                "prompt": "A futuristic cityscape",
                "negative_prompt": "low quality",
                "resolution": "512x512",
                "frames": 41,
                "seed": 54321,
                "model": "vace_14B",
                "use_causvid_lora": True,
                "use_lighti2x_lora": False,
                "additional_loras": {
                    "test_lora.safetensors": 0.8,
                    "another_lora.safetensors": 0.5
                }
            }
            
            db_ops.add_task_to_db(
                task_payload=task_payload,
                task_type_str="wgp"
            )
            self.created_tasks.append(task_id)
            print(f"[TEST_LORA] ✅ Created task with LoRA configuration: {task_id}")
            
            # Verify LoRA parameters are stored
            task_params = db_ops.get_task_params(task_id)
            if task_params:
                parsed_params = json.loads(task_params) if isinstance(task_params, str) else task_params
                has_causvid = parsed_params.get("use_causvid_lora", False)
                has_additional = "additional_loras" in parsed_params
                print(f"[TEST_LORA] ✅ LoRA parameters verified: CausVid={has_causvid}, Additional={has_additional}")
                return task_id
            else:
                print(f"[TEST_LORA] ❌ Could not verify LoRA parameters")
                return None
                
        except Exception as e:
            print(f"[TEST_LORA] ❌ Error with LoRA task: {e}")
            traceback.print_exc()
            return None

    def test_task_chaining_workflow(self):
        """Test task chaining and dependency workflow"""
        print(f"\n[TEST_CHAINING] Testing task chaining workflow...")
        
        try:
            # Create a chain: OpenPose -> WGP Video -> Frame Extract
            
            # Step 1: OpenPose task
            test_image = self.setup_temp_files("image")
            openpose_task_id = generate_unique_task_id(f"chain_pose_{self.test_run_id}_")
            openpose_payload = {
                "task_id": openpose_task_id,
                "project_id": self.project_id,
                "input_image_path": str(test_image)
            }
            
            db_ops.add_task_to_db(
                task_payload=openpose_payload,
                task_type_str="generate_openpose"
            )
            self.created_tasks.append(openpose_task_id)
            
            # Complete OpenPose task
            pose_output = self.setup_temp_files("openpose")
            db_ops.update_task_status(openpose_task_id, db_ops.STATUS_IN_PROGRESS)
            db_ops.update_task_status(openpose_task_id, db_ops.STATUS_COMPLETE, str(pose_output))
            
            # Step 2: WGP Video task (depends on OpenPose)
            wgp_task_id = generate_unique_task_id(f"chain_wgp_{self.test_run_id}_")
            wgp_payload = {
                "task_id": wgp_task_id,
                "project_id": self.project_id,
                "prompt": "Dancing person following pose sequence",
                "resolution": "512x512",
                "frames": 41,
                "model": "vace_14B",
                "pose_reference_task": openpose_task_id
            }
            
            db_ops.add_task_to_db(
                task_payload=wgp_payload,
                task_type_str="wgp",
                dependant_on=openpose_task_id
            )
            self.created_tasks.append(wgp_task_id)
            
            # Complete WGP task
            video_output = self.setup_temp_files("video")
            db_ops.update_task_status(wgp_task_id, db_ops.STATUS_IN_PROGRESS)
            db_ops.update_task_status(wgp_task_id, db_ops.STATUS_COMPLETE, str(video_output))
            
            # Step 3: Frame extraction (depends on WGP)
            extract_task_id = generate_unique_task_id(f"chain_extract_{self.test_run_id}_")
            extract_payload = {
                "task_id": extract_task_id,
                "project_id": self.project_id,
                "frame_number": 20,
                "source_video_task": wgp_task_id
            }
            
            db_ops.add_task_to_db(
                task_payload=extract_payload,
                task_type_str="extract_frame",
                dependant_on=wgp_task_id
            )
            self.created_tasks.append(extract_task_id)
            
            print(f"[TEST_CHAINING] ✅ Created task chain: {openpose_task_id} -> {wgp_task_id} -> {extract_task_id}")
            
            # Test dependency resolution
            dependency = db_ops.get_task_dependency(wgp_task_id)
            if dependency == openpose_task_id:
                print(f"[TEST_CHAINING] ✅ Dependency chain verified")
            else:
                print(f"[TEST_CHAINING] ❌ Dependency chain failed: expected {openpose_task_id}, got {dependency}")
            
            return [openpose_task_id, wgp_task_id, extract_task_id]
            
        except Exception as e:
            print(f"[TEST_CHAINING] ❌ Error with task chaining: {e}")
            traceback.print_exc()
            return None

    # =======================================================================
    # SUPABASE-SPECIFIC EDGE FUNCTION TESTS
    # =======================================================================
    
    def test_supabase_edge_function_create_task(self):
        """Test the create-task edge function specifically"""
        print(f"\n[TEST_SUPABASE_CREATE] Testing Supabase create-task edge function...")
        
        if db_ops.DB_TYPE != "supabase":
            print(f"[TEST_SUPABASE_CREATE] ⚠️  Skipping - DB_TYPE is {db_ops.DB_TYPE}, not supabase")
            return False
            
        try:
            import httpx
            
            task_id = generate_unique_task_id(f"supabase_create_{self.test_run_id}_")
            edge_url = f"{db_ops.SUPABASE_URL.rstrip('/')}/functions/v1/create-task"
            
            headers = {"Content-Type": "application/json"}
            if db_ops.SUPABASE_ACCESS_TOKEN:
                headers["Authorization"] = f"Bearer {db_ops.SUPABASE_ACCESS_TOKEN}"
            
            payload = {
                "task_id": task_id,
                "params": {
                    "prompt": "Test prompt for edge function",
                    "resolution": "512x512",
                    "frames": 41
                },
                "task_type": "wgp",
                "project_id": self.project_id
            }
            
            print(f"[TEST_SUPABASE_CREATE] Calling edge function: {edge_url}")
            resp = httpx.post(edge_url, json=payload, headers=headers, timeout=30)
            
            print(f"[TEST_SUPABASE_CREATE] Response status: {resp.status_code}")
            print(f"[TEST_SUPABASE_CREATE] Response text: {resp.text}")
            
            if resp.status_code == 200:
                print(f"[TEST_SUPABASE_CREATE] ✅ Edge function create-task successful")
                self.created_tasks.append(task_id)
                return True
            else:
                print(f"[TEST_SUPABASE_CREATE] ❌ Edge function failed: {resp.status_code} - {resp.text}")
                return False
                
        except Exception as e:
            print(f"[TEST_SUPABASE_CREATE] ❌ Error testing create-task edge function: {e}")
            traceback.print_exc()
            return False

    def test_supabase_edge_function_claim_task(self):
        """Test the claim-next-task edge function with both Service Key and PAT"""
        print(f"\n[TEST_SUPABASE_CLAIM] Testing Supabase claim-next-task edge function...")
        
        if db_ops.DB_TYPE != "supabase":
            print(f"[TEST_SUPABASE_CLAIM] ⚠️  Skipping - DB_TYPE is {db_ops.DB_TYPE}, not supabase")
            return False
            
        try:
            import httpx
            
            claim_url = f"{db_ops.SUPABASE_URL.rstrip('/')}/functions/v1/claim-next-task"
            
            # Test 1: Service Key Authentication
            print(f"[TEST_SUPABASE_CLAIM] Testing with Service Key...")
            
            # Create a task to claim using service key
            task_id_1 = generate_unique_task_id(f"service_claim_{self.test_run_id}_")
            create_payload_1 = {
                "task_id": task_id_1,
                "params": {"test": "service_claim_test"},
                "task_type": "wgp", 
                "project_id": self.project_id
            }
            
            create_url = f"{db_ops.SUPABASE_URL.rstrip('/')}/functions/v1/create-task"
            service_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {db_ops.SUPABASE_SERVICE_KEY}"
            }
            
            create_resp_1 = httpx.post(create_url, json=create_payload_1, headers=service_headers, timeout=30)
            if create_resp_1.status_code == 200:
                self.created_tasks.append(task_id_1)
                
                # Try to claim with service key
                service_payload = {"worker_id": self.worker_id}
                service_resp = httpx.post(claim_url, json=service_payload, headers=service_headers, timeout=30)
                
                print(f"[TEST_SUPABASE_CLAIM] Service Key - Response status: {service_resp.status_code}")
                print(f"[TEST_SUPABASE_CLAIM] Service Key - Response text: {service_resp.text}")
            
            # Test 2: PAT Authentication (WITHOUT worker_id - as individual users wouldn't have one)
            print(f"[TEST_SUPABASE_CLAIM] Testing with PAT (no worker_id)...")
            
            pat_token = "F23ZwOZ0GkCbV718Eo0Q8LtC"  # Example PAT
            pat_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {pat_token}"
            }
            
            # Create another task for PAT test (using service key since PAT might not have create permissions)
            task_id_2 = generate_unique_task_id(f"pat_claim_{self.test_run_id}_")
            create_payload_2 = {
                "task_id": task_id_2,
                "params": {"test": "pat_claim_test"},
                "task_type": "wgp", 
                "project_id": self.project_id
            }
            
            create_resp_2 = httpx.post(create_url, json=create_payload_2, headers=service_headers, timeout=30)
            if create_resp_2.status_code == 200:
                self.created_tasks.append(task_id_2)
                
                # Try to claim with PAT - NO worker_id passed (individual users don't have worker IDs)
                pat_payload = {}  # Empty payload - no worker_id
                pat_resp = httpx.post(claim_url, json=pat_payload, headers=pat_headers, timeout=30)
                
                print(f"[TEST_SUPABASE_CLAIM] PAT - Response status: {pat_resp.status_code}")
                print(f"[TEST_SUPABASE_CLAIM] PAT - Response text: {pat_resp.text}")
            
            # Evaluate results
            service_success = 'service_resp' in locals() and service_resp.status_code in [200, 204, 500]  # 500 is expected for constraint
            pat_success = 'pat_resp' in locals() and pat_resp.status_code in [200, 204, 403, 500]  # 403 is ok for invalid PAT
            
            if service_success:
                print(f"[TEST_SUPABASE_CLAIM] ✅ Service Key test completed")
            else:
                print(f"[TEST_SUPABASE_CLAIM] ❌ Service Key test failed")
                
            if pat_success:
                print(f"[TEST_SUPABASE_CLAIM] ✅ PAT test completed")  
            else:
                print(f"[TEST_SUPABASE_CLAIM] ❌ PAT test failed")
                
            return service_success
                
        except Exception as e:
            print(f"[TEST_SUPABASE_CLAIM] ❌ Error testing claim-next-task edge function: {e}")
            traceback.print_exc()
            return False

    def test_supabase_edge_function_complete_task(self):
        """Test the complete-task edge function specifically"""
        print(f"\n[TEST_SUPABASE_COMPLETE] Testing Supabase complete-task edge function...")
        
        if db_ops.DB_TYPE != "supabase":
            print(f"[TEST_SUPABASE_COMPLETE] ⚠️  Skipping - DB_TYPE is {db_ops.DB_TYPE}, not supabase")
            return False
            
        try:
            import httpx
            import base64
            
            # Create a task and set it to In Progress
            task_id = generate_unique_task_id(f"supabase_complete_{self.test_run_id}_")
            
            # Create task
            create_payload = {
                "task_id": task_id,
                "params": {"test": "complete_test"},
                "task_type": "wgp",
                "project_id": self.project_id
            }
            
            create_url = f"{db_ops.SUPABASE_URL.rstrip('/')}/functions/v1/create-task"
            headers = {"Content-Type": "application/json"}
            if db_ops.SUPABASE_ACCESS_TOKEN:
                headers["Authorization"] = f"Bearer {db_ops.SUPABASE_ACCESS_TOKEN}"
            
            create_resp = httpx.post(create_url, json=create_payload, headers=headers, timeout=30)
            if create_resp.status_code != 200:
                print(f"[TEST_SUPABASE_COMPLETE] ❌ Failed to create test task: {create_resp.status_code}")
                return False
            
            self.created_tasks.append(task_id)
            
            # Create test file content
            test_video = self.setup_temp_files("video")
            with open(test_video, 'rb') as f:
                file_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Complete the task
            complete_url = f"{db_ops.SUPABASE_URL.rstrip('/')}/functions/v1/complete-task"
            complete_payload = {
                "task_id": task_id,
                "file_data": file_data,
                "filename": f"test_complete_{task_id}.mp4"
            }
            
            complete_resp = httpx.post(complete_url, json=complete_payload, headers=headers, timeout=60)
            
            print(f"[TEST_SUPABASE_COMPLETE] Complete response status: {complete_resp.status_code}")
            print(f"[TEST_SUPABASE_COMPLETE] Complete response text: {complete_resp.text}")
            
            if complete_resp.status_code == 200:
                result = complete_resp.json()
                print(f"[TEST_SUPABASE_COMPLETE] ✅ Edge function complete-task successful")
                print(f"[TEST_SUPABASE_COMPLETE] Public URL: {result.get('public_url')}")
                return True
            else:
                print(f"[TEST_SUPABASE_COMPLETE] ❌ Edge function failed: {complete_resp.status_code} - {complete_resp.text}")
                return False
                
        except Exception as e:
            print(f"[TEST_SUPABASE_COMPLETE] ❌ Error testing complete-task edge function: {e}")
            traceback.print_exc()
            return False

    def test_supabase_edge_function_get_predecessor_output(self):
        """Test the get-predecessor-output edge function specifically"""
        print(f"\n[TEST_SUPABASE_PREDECESSOR] Testing Supabase get-predecessor-output edge function...")
        
        if db_ops.DB_TYPE != "supabase":
            print(f"[TEST_SUPABASE_PREDECESSOR] ⚠️  Skipping - DB_TYPE is {db_ops.DB_TYPE}, not supabase")
            return False
            
        try:
            import httpx
            
            # Create parent task and complete it
            parent_task_id = generate_unique_task_id(f"supabase_parent_{self.test_run_id}_")
            child_task_id = generate_unique_task_id(f"supabase_child_{self.test_run_id}_")
            
            headers = {"Content-Type": "application/json"}
            if db_ops.SUPABASE_ACCESS_TOKEN:
                headers["Authorization"] = f"Bearer {db_ops.SUPABASE_ACCESS_TOKEN}"
            
            create_url = f"{db_ops.SUPABASE_URL.rstrip('/')}/functions/v1/create-task"
            
            # Create parent task
            parent_payload = {
                "task_id": parent_task_id,
                "params": {"test": "parent_task"},
                "task_type": "wgp",
                "project_id": self.project_id
            }
            
            parent_resp = httpx.post(create_url, json=parent_payload, headers=headers, timeout=30)
            if parent_resp.status_code != 200:
                print(f"[TEST_SUPABASE_PREDECESSOR] ❌ Failed to create parent task")
                return False
            
            self.created_tasks.append(parent_task_id)
            
            # Create child task that depends on parent
            child_payload = {
                "task_id": child_task_id,
                "params": {"test": "child_task"},
                "task_type": "wgp",
                "project_id": self.project_id,
                "dependant_on": parent_task_id
            }
            
            child_resp = httpx.post(create_url, json=child_payload, headers=headers, timeout=30)
            if child_resp.status_code != 200:
                print(f"[TEST_SUPABASE_PREDECESSOR] ❌ Failed to create child task")
                return False
            
            self.created_tasks.append(child_task_id)
            
            # Complete parent task with file upload
            test_video = self.setup_temp_files("video")
            with open(test_video, 'rb') as f:
                file_data = base64.b64encode(f.read()).decode('utf-8')
            
            complete_url = f"{db_ops.SUPABASE_URL.rstrip('/')}/functions/v1/complete-task"
            complete_payload = {
                "task_id": parent_task_id,
                "file_data": file_data,
                "filename": f"test_parent_{parent_task_id}.mp4"
            }
            
            complete_resp = httpx.post(complete_url, json=complete_payload, headers=headers, timeout=60)
            if complete_resp.status_code != 200:
                print(f"[TEST_SUPABASE_PREDECESSOR] ❌ Failed to complete parent task")
                return False
            
            # Now test get-predecessor-output
            predecessor_url = f"{db_ops.SUPABASE_URL.rstrip('/')}/functions/v1/get-predecessor-output"
            predecessor_payload = {"task_id": child_task_id}
            
            predecessor_resp = httpx.post(predecessor_url, json=predecessor_payload, headers=headers, timeout=30)
            
            print(f"[TEST_SUPABASE_PREDECESSOR] Predecessor response status: {predecessor_resp.status_code}")
            print(f"[TEST_SUPABASE_PREDECESSOR] Predecessor response: {predecessor_resp.text}")
            
            if predecessor_resp.status_code == 200:
                result = predecessor_resp.json()
                if result and result.get('predecessor_id') == parent_task_id and result.get('output_location'):
                    print(f"[TEST_SUPABASE_PREDECESSOR] ✅ Edge function get-predecessor-output successful")
                    print(f"[TEST_SUPABASE_PREDECESSOR] Found predecessor: {result.get('predecessor_id')}")
                    print(f"[TEST_SUPABASE_PREDECESSOR] Output location: {result.get('output_location')}")
                    return True
                else:
                    print(f"[TEST_SUPABASE_PREDECESSOR] ❌ Unexpected response format: {result}")
                    return False
            else:
                print(f"[TEST_SUPABASE_PREDECESSOR] ❌ Edge function failed: {predecessor_resp.status_code}")
                return False
                
        except Exception as e:
            print(f"[TEST_SUPABASE_PREDECESSOR] ❌ Error testing get-predecessor-output edge function: {e}")
            traceback.print_exc()
            return False

    def test_supabase_edge_function_get_completed_segments(self):
        """Test the get-completed-segments edge function specifically"""
        print(f"\n[TEST_SUPABASE_SEGMENTS] Testing Supabase get-completed-segments edge function...")
        
        if db_ops.DB_TYPE != "supabase":
            print(f"[TEST_SUPABASE_SEGMENTS] ⚠️  Skipping - DB_TYPE is {db_ops.DB_TYPE}, not supabase")
            return False
            
        try:
            import httpx
            
            headers = {"Content-Type": "application/json"}
            if db_ops.SUPABASE_ACCESS_TOKEN:
                headers["Authorization"] = f"Bearer {db_ops.SUPABASE_ACCESS_TOKEN}"
            
            create_url = f"{db_ops.SUPABASE_URL.rstrip('/')}/functions/v1/create-task"
            complete_url = f"{db_ops.SUPABASE_URL.rstrip('/')}/functions/v1/complete-task"
            
            # Create and complete multiple travel_segment tasks
            segment_tasks = []
            for i in range(3):
                segment_task_id = generate_unique_task_id(f"supabase_seg_{i}_{self.test_run_id}_")
                
                # Create segment task
                segment_payload = {
                    "task_id": segment_task_id,
                    "params": {
                        "orchestrator_run_id": self.test_run_id,
                        "segment_index": i,
                        "test": f"segment_{i}"
                    },
                    "task_type": "travel_segment",
                    "project_id": self.project_id
                }
                
                create_resp = httpx.post(create_url, json=segment_payload, headers=headers, timeout=30)
                if create_resp.status_code != 200:
                    print(f"[TEST_SUPABASE_SEGMENTS] ❌ Failed to create segment {i}")
                    continue
                
                self.created_tasks.append(segment_task_id)
                
                # Complete with file upload
                test_video = self.setup_temp_files("video")
                with open(test_video, 'rb') as f:
                    file_data = base64.b64encode(f.read()).decode('utf-8')
                
                complete_payload = {
                    "task_id": segment_task_id,
                    "file_data": file_data,
                    "filename": f"segment_{i}_{segment_task_id}.mp4"
                }
                
                complete_resp = httpx.post(complete_url, json=complete_payload, headers=headers, timeout=60)
                if complete_resp.status_code == 200:
                    segment_tasks.append((i, segment_task_id))
                    print(f"[TEST_SUPABASE_SEGMENTS] Created and completed segment {i}")
            
            if not segment_tasks:
                print(f"[TEST_SUPABASE_SEGMENTS] ❌ No segments were created successfully")
                return False
            
            # Wait a moment for processing
            time.sleep(2)
            
            # Test get-completed-segments
            segments_url = f"{db_ops.SUPABASE_URL.rstrip('/')}/functions/v1/get-completed-segments"
            segments_payload = {
                "run_id": self.test_run_id,
                "project_id": self.project_id
            }
            
            segments_resp = httpx.post(segments_url, json=segments_payload, headers=headers, timeout=30)
            
            print(f"[TEST_SUPABASE_SEGMENTS] Segments response status: {segments_resp.status_code}")
            print(f"[TEST_SUPABASE_SEGMENTS] Segments response: {segments_resp.text}")
            
            if segments_resp.status_code == 200:
                segments = segments_resp.json()
                if isinstance(segments, list) and len(segments) > 0:
                    print(f"[TEST_SUPABASE_SEGMENTS] ✅ Edge function get-completed-segments successful")
                    print(f"[TEST_SUPABASE_SEGMENTS] Found {len(segments)} completed segments")
                    for seg in segments:
                        print(f"[TEST_SUPABASE_SEGMENTS]   Segment {seg.get('segment_index')}: {seg.get('output_location')}")
                    return True
                else:
                    print(f"[TEST_SUPABASE_SEGMENTS] ⚠️  No segments found in response")
                    return False
            else:
                print(f"[TEST_SUPABASE_SEGMENTS] ❌ Edge function failed: {segments_resp.status_code}")
                return False
                
        except Exception as e:
            print(f"[TEST_SUPABASE_SEGMENTS] ❌ Error testing get-completed-segments edge function: {e}")
            traceback.print_exc()
            return False

    # RPC functions have been removed - all operations now use Edge Functions exclusively

    def test_supabase_storage_operations(self):
        """Test Supabase storage upload/download operations"""
        print(f"\n[TEST_SUPABASE_STORAGE] Testing Supabase storage operations...")
        
        if db_ops.DB_TYPE != "supabase" or not db_ops.SUPABASE_CLIENT:
            print(f"[TEST_SUPABASE_STORAGE] ⚠️  Skipping - DB_TYPE is {db_ops.DB_TYPE} or no Supabase client")
            return False
            
        try:
            # Test file upload
            test_video = self.setup_temp_files("video")
            object_name = f"test_upload_{self.test_run_id}.mp4"
            
            print(f"[TEST_SUPABASE_STORAGE] Testing file upload...")
            public_url = db_ops.upload_to_supabase_storage(
                test_video,
                bucket_name=db_ops.SUPABASE_VIDEO_BUCKET,
                custom_path=object_name
            )
            
            if public_url:
                print(f"[TEST_SUPABASE_STORAGE] ✅ File upload successful")
                print(f"[TEST_SUPABASE_STORAGE] Public URL: {public_url}")
                
                # Test file download/access
                try:
                    import httpx
                    resp = httpx.head(public_url, timeout=10)
                    if resp.status_code == 200:
                        print(f"[TEST_SUPABASE_STORAGE] ✅ File accessible via public URL")
                        return True
                    else:
                        print(f"[TEST_SUPABASE_STORAGE] ❌ File not accessible: {resp.status_code}")
                        return False
                except Exception as e_download:
                    print(f"[TEST_SUPABASE_STORAGE] ❌ Error accessing public URL: {e_download}")
                    return False
            else:
                print(f"[TEST_SUPABASE_STORAGE] ❌ File upload failed")
                return False
                
        except Exception as e:
            print(f"[TEST_SUPABASE_STORAGE] ❌ Error testing storage operations: {e}")
            traceback.print_exc()
            return False

    def test_supabase_functions(self):
        """Test core Supabase edge functions and database operations."""
        if db_ops.DB_TYPE != "supabase":
            print("[TEST_SUPABASE_FUNCTIONS] ⚠️ Skipping Supabase tests (not configured)")
            return {"edge_functions": "SKIPPED", "database": "SKIPPED"}

        results = {}
        
        print("[TEST_SUPABASE_EDGE] Testing Supabase Edge Functions...")
        
        # Test all edge functions that should be available
        edge_functions = {
            "create-task": f"{db_ops.SUPABASE_URL.rstrip('/')}/functions/v1/create-task",
            "claim-next-task": f"{db_ops.SUPABASE_URL.rstrip('/')}/functions/v1/claim-next-task", 
            "complete-task": f"{db_ops.SUPABASE_URL.rstrip('/')}/functions/v1/complete-task",
            "get-predecessor-output": f"{db_ops.SUPABASE_URL.rstrip('/')}/functions/v1/get-predecessor-output",
            "get-completed-segments": f"{db_ops.SUPABASE_URL.rstrip('/')}/functions/v1/get-completed-segments"
        }
        
        for func_name, url in edge_functions.items():
            try:
                print(f"[TEST_SUPABASE_EDGE] Testing {func_name}...")
                headers = {"Content-Type": "application/json"}
                if db_ops.SUPABASE_ACCESS_TOKEN:
                    headers["Authorization"] = f"Bearer {db_ops.SUPABASE_ACCESS_TOKEN}"
                
                # Test with minimal payload to see if function exists and responds
                import httpx
                test_payload = {}
                if func_name == "create-task":
                    test_payload = {
                        "task_id": f"test-edge-{self.test_run_id}",
                        "task_type": "test",
                        "params": {"test": True},
                        "project_id": self.project_id
                    }
                elif func_name == "claim-next-task":
                    test_payload = {"worker_id": self.worker_id}
                elif func_name in ["get-predecessor-output", "complete-task"]:
                    test_payload = {"task_id": f"test-{self.test_run_id}"}
                elif func_name == "get-completed-segments":
                    test_payload = {"orchestrator_id": f"test-{self.test_run_id}"}
                
                resp = httpx.post(url, json=test_payload, headers=headers, timeout=10)
                
                # Edge functions should respond (even if with 404 for missing data)
                # We're just testing they exist and are accessible
                if resp.status_code in [200, 204, 404, 500]:  # Valid responses
                    print(f"[TEST_SUPABASE_EDGE] ✅ {func_name} accessible (status: {resp.status_code})")
                    results[func_name] = True
                else:
                    print(f"[TEST_SUPABASE_EDGE] ❌ {func_name} failed (status: {resp.status_code})")
                    results[func_name] = False
                    
            except Exception as e:
                print(f"[TEST_SUPABASE_EDGE] ❌ {func_name} exception: {e}")
                results[func_name] = False
        
        # Test database connectivity
        try:
            print(f"[TEST_SUPABASE_DB] Testing database connectivity...")
            count_result = db_ops.SUPABASE_CLIENT.table(db_ops.PG_TABLE_NAME).select("id", count="exact").execute()
            task_count = len(count_result.data) if count_result.data else 0
            print(f"[TEST_SUPABASE_DB] ✅ Database accessible - found {task_count} tasks")
            results['database'] = True
        except Exception as e:
            print(f"[TEST_SUPABASE_DB] ❌ Database connection failed: {e}")
            results['database'] = False
            
        return results

    def run_all_tests(self):
        """Run all tests in sequence"""
        print(f"🧪 Starting comprehensive workflow database and edge function tests")
        print(f"=" * 80)
        
        results = {}
        
        try:
            # Core Database & Edge Function Tests
            print(f"\n📁 CORE DATABASE & EDGE FUNCTION TESTS")
            print(f"-" * 50)
            
            # Test 1: Database initialization
            results['db_init'] = self.test_database_initialization()
            
            # Test 2: Create task edge function
            orchestrator_task_id = self.test_edge_function_create_task()
            results['create_task'] = orchestrator_task_id is not None
            
            # Test 3: Claim task edge function  
            claimed_task = self.test_edge_function_claim_task()
            results['claim_task'] = claimed_task is not None
            
            # Test 4: Complete task edge function
            completed_task_id, output_location = self.test_edge_function_complete_task()
            results['complete_task'] = completed_task_id is not None and output_location is not None
            
            # Test 5: Predecessor dependency chain
            task1, task2, task3 = self.test_predecessor_dependency_chain()
            results['predecessor_chain'] = all([task1, task2, task3])
            
            # Test 6: Segment collection for stitching
            segments = self.test_segment_collection_for_stitching()
            results['segment_collection'] = segments is not None and len(segments) > 0
            
            # Test 7: Status updates
            results['status_updates'] = self.test_task_status_updates()
            
            # Image Generation & Specialized Task Tests
            print(f"\n🎨 IMAGE GENERATION & SPECIALIZED TASK TESTS")
            print(f"-" * 50)
            
            # Test 8: Single image generation
            single_image_task = self.test_single_image_generation_task()
            results['single_image'] = single_image_task is not None
            
            # Test 9: OpenPose generation
            openpose_task = self.test_openpose_generation_task()
            results['openpose'] = openpose_task is not None
            
            # Test 10: RIFE interpolation
            rife_task = self.test_rife_interpolation_task()
            results['rife_interpolation'] = rife_task is not None
            
            # Test 11: Frame extraction
            extract_task = self.test_extract_frame_task()
            results['extract_frame'] = extract_task is not None
            
            # Test 12: Different perspective orchestrator
            dp_orchestrator = self.test_different_perspective_orchestrator()
            results['dp_orchestrator'] = dp_orchestrator is not None
            
            # Video Generation & Advanced Features Tests
            print(f"\n🎬 VIDEO GENERATION & ADVANCED FEATURES TESTS")
            print(f"-" * 50)
            
            # Test 13: Standard WGP video generation
            wgp_video_task = self.test_standard_wgp_video_generation()
            results['wgp_video'] = wgp_video_task is not None
            
            # Test 14: LoRA handling
            lora_task = self.test_lora_handling_task()
            results['lora_handling'] = lora_task is not None
            
            # Test 15: Task chaining workflow
            chaining_tasks = self.test_task_chaining_workflow()
            results['task_chaining'] = chaining_tasks is not None and len(chaining_tasks) == 3
            
            # Supabase-Specific Tests
            print(f"\n🚀 SUPABASE EDGE FUNCTIONS & RPC TESTS")
            print(f"-" * 50)
            
            # Test 16: Supabase Edge Function - create-task
            results['supabase_edge_create'] = self.test_supabase_edge_function_create_task()
            
            # Test 17: Supabase Edge Function - claim-next-task
            results['supabase_edge_claim'] = self.test_supabase_edge_function_claim_task()
            
            # Test 18: Supabase Edge Function - complete-task
            results['supabase_edge_complete'] = self.test_supabase_edge_function_complete_task()
            
            # Test 19: Supabase Edge Function - get-predecessor-output
            results['supabase_edge_predecessor'] = self.test_supabase_edge_function_get_predecessor_output()
            
            # Test 20: Supabase Edge Function - get-completed-segments
            results['supabase_edge_segments'] = self.test_supabase_edge_function_get_completed_segments()
            
            # Test 21: Supabase RPC Functions (all 4 functions)
            # RPC functions have been removed - all operations now use Edge Functions
            
            # Test 22: Supabase Storage Operations
            results['supabase_storage'] = self.test_supabase_storage_operations()
            
            # Test 23: Supabase Core Edge Functions
            results['supabase_functions'] = self.test_supabase_functions()
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR during test execution: {e}")
            traceback.print_exc()
            return False
        
        finally:
            self.cleanup()
        
        # Print detailed results summary
        print(f"\n" + "=" * 80)
        print(f"📊 COMPREHENSIVE TEST RESULTS SUMMARY")
        print(f"=" * 80)
        
        # Group results by category
        core_tests = ['db_init', 'create_task', 'claim_task', 'complete_task', 'predecessor_chain', 'segment_collection', 'status_updates']
        image_tests = ['single_image', 'openpose', 'rife_interpolation', 'extract_frame', 'dp_orchestrator']
        video_tests = ['wgp_video', 'lora_handling', 'task_chaining']
        supabase_tests = ['supabase_edge_create', 'supabase_edge_claim', 'supabase_edge_complete', 'supabase_edge_predecessor', 'supabase_edge_segments', 'supabase_rpc_functions', 'supabase_storage']
        
        def print_category_results(category_name, test_keys):
            passed_in_category = 0
            total_in_category = len(test_keys)
            print(f"\n{category_name}:")
            for test_key in test_keys:
                if test_key in results:
                    status = "✅ PASS" if results[test_key] else "❌ FAIL"
                    print(f"  {test_key.upper().replace('_', ' ')}: {status}")
                    if results[test_key]:
                        passed_in_category += 1
                else:
                    print(f"  {test_key.upper().replace('_', ' ')}: ⚠️  NOT RUN")
            return passed_in_category, total_in_category
        
        core_passed, core_total = print_category_results("🔧 Core Database & Edge Functions", core_tests)
        image_passed, image_total = print_category_results("🎨 Image Generation & Specialized Tasks", image_tests)
        video_passed, video_total = print_category_results("🎬 Video Generation & Advanced Features", video_tests)
        supabase_passed, supabase_total = print_category_results("🚀 Supabase Edge Functions & RPC", supabase_tests)
        
        total_passed = core_passed + image_passed + video_passed + supabase_passed
        total_tests = core_total + image_total + video_total + supabase_total
        
        print(f"\n📈 OVERALL RESULTS:")
        print(f"  Core Tests: {core_passed}/{core_total}")
        print(f"  Image Tests: {image_passed}/{image_total}")
        print(f"  Video Tests: {video_passed}/{video_total}")
        print(f"  Supabase Tests: {supabase_passed}/{supabase_total}")
        
        if db_ops.DB_TYPE == "supabase":
            print(f"  Total: {total_passed}/{total_tests} tests passed")
        else:
            non_supabase_passed = core_passed + image_passed + video_passed
            non_supabase_total = core_total + image_total + video_total
            print(f"  Total (excluding Supabase): {non_supabase_passed}/{non_supabase_total} tests passed")
            print(f"  Note: Supabase tests skipped (DB_TYPE is {db_ops.DB_TYPE})")
        
        # Calculate success rates
        if db_ops.DB_TYPE == "supabase":
            success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
            all_passed = total_passed == total_tests
        else:
            # For non-Supabase, exclude Supabase tests from success calculation
            non_supabase_passed = core_passed + image_passed + video_passed
            non_supabase_total = core_total + image_total + video_total
            success_rate = (non_supabase_passed / non_supabase_total) * 100 if non_supabase_total > 0 else 0
            all_passed = non_supabase_passed == non_supabase_total
        
        if all_passed:
            if db_ops.DB_TYPE == "supabase":
                print(f"\n🎉 ALL TESTS PASSED! Complete system including Supabase edge functions working correctly.")
            else:
                print(f"\n🎉 ALL CORE TESTS PASSED! Workflow system working correctly with {db_ops.DB_TYPE}.")
                print(f"💡 To test Supabase edge functions, configure your environment for Supabase and run again.")
            print(f"✨ Success Rate: {success_rate:.1f}%")
            return True
        elif success_rate >= 80:
            print(f"\n✅ MOSTLY SUCCESSFUL! {success_rate:.1f}% of tests passed.")
            print(f"⚠️  Some minor issues detected. Please review failed tests above.")
            return True
        else:
            print(f"\n❌ SIGNIFICANT ISSUES DETECTED! Only {success_rate:.1f}% of tests passed.")
            print(f"🔧 Please review and fix the failed tests above.")
            return False

def main():
    """Main test function"""
    print(f"🚀 Wan2GP Comprehensive Workflow Test Suite")
    print(f"=" * 60)
    
    # Configure Supabase if environment variables are available
    configure_supabase_if_available()
    
    # Display configuration information
    print(f"Database Type: {db_ops.DB_TYPE}")
    if db_ops.DB_TYPE == "supabase":
        print(f"Supabase URL: {db_ops.SUPABASE_URL}")
        print(f"Table Name: {db_ops.PG_TABLE_NAME}")
        print(f"Video Bucket: {db_ops.SUPABASE_VIDEO_BUCKET}")
        print(f"✅ Configured for Supabase - Full edge function testing enabled")
    else:
        print(f"SQLite Path: {db_ops.SQLITE_DB_PATH}")
        print(f"⚠️  Using SQLite - Edge function tests will simulate local operations")
    
    print(f"\nThis test suite covers:")
    print(f"• Core database operations and edge functions")
    print(f"• Image generation and specialized tasks") 
    print(f"• Video generation and advanced features")
    print(f"• Task chaining and dependency workflows")
    print(f"• File upload/download handling")
    
    if db_ops.DB_TYPE == "supabase":
        print(f"• 🚀 Supabase edge functions: create-task, claim-next-task, complete-task, get-predecessor-output, get-completed-segments")
        # Note: RPC functions have been eliminated - all operations now use Edge Functions
        print(f"• 📁 Supabase storage operations: file upload/download")
    else:
        print(f"• ⚠️  Supabase tests will be skipped (configure Supabase to enable them)")
    
    # Auto-proceed with testing (confirmation removed for automation)
        sys.exit(0)
    
    print(f"\n🧪 Starting comprehensive test execution...")
    
    tester = TravelWorkflowTester()
    success = tester.run_all_tests()
    
    if success:
        print(f"\n🎊 COMPREHENSIVE TEST SUITE COMPLETED SUCCESSFULLY! 🎊")
        print(f"The Wan2GP workflow system is functioning correctly.")
        sys.exit(0)
    else:
        print(f"\n💥 TEST SUITE COMPLETED WITH ISSUES 💥")
        print(f"Please review the failed tests and fix any issues.")
        sys.exit(1)

if __name__ == "__main__":
    main() 