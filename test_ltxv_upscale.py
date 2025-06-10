#!/usr/bin/env python3
"""
Test script for LTXV upscaling task.
This script demonstrates how to add an LTXV upscaling task to the database queue.
"""

import json
import time
from pathlib import Path

# Add the current directory to Python path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from source import db_operations as db_ops

def test_ltxv_upscale_task():
    """Create a test LTXV upscaling task and add it to the database."""
    
    # Initialize database
    db_ops.DB_TYPE = "sqlite"
    db_ops.SQLITE_DB_PATH = "tasks.db"
    db_ops.init_db()
    
    # Look for a sample video file in common locations
    possible_video_paths = [
        "sample_video.mp4",
        "test.mp4",
        "test_video.mp4", 
        "input.mp4",
        Path("videos") / "sample.mp4",
        Path("test_videos") / "sample.mp4"
    ]
    
    video_source_path = None
    for path in possible_video_paths:
        if Path(path).exists():
            video_source_path = str(Path(path).resolve())
            break
    
    if not video_source_path:
        print("⚠️  No test video found. Please provide a video file path:")
        print("   You can either:")
        print("   1. Place a video file named 'sample_video.mp4' in the current directory")
        print("   2. Modify the script to point to your video file")
        
        # Ask user for video path
        user_input = input("\nEnter path to a video file (or press Enter to skip): ").strip()
        if user_input and Path(user_input).exists():
            video_source_path = str(Path(user_input).resolve())
        else:
            print("❌ No valid video file provided. Creating sample JSON only.")
            create_sample_task_json()
            return
    
    # Example task parameters
    task_params = {
        "task_id": f"ltxv_upscale_test_{int(time.time())}",
        "video_source_path": video_source_path,
        "upscale_factor": 2.0,  # 2x upscaling
        "output_path": None,  # Let the system choose the output path
    }
    
    print(f"Adding LTXV upscaling task to database...")
    print(f"Input video: {video_source_path}")
    print(f"Task Parameters: {json.dumps(task_params, indent=2)}")
    
    # Add task to database
    try:
        db_ops.add_task_to_db(
            task_payload=task_params,
            task_type_str="ltxv_upscale"
        )
        print(f"✅ Successfully added task '{task_params['task_id']}' to database")
        print(f"\nTask added with ID: {task_params['task_id']}")
        print("\nTo process this task, run the headless server:")
        print("python headless.py --debug")
        print("\nThe task will:")
        print("1. Load the input video")
        print("2. Encode it to LTXV latent space")
        print("3. Apply the spatial upsampler (2x resolution)")
        print("4. Decode back to video")
        print("5. Save the upscaled result")
        
    except Exception as e:
        print(f"❌ Failed to add task to database: {e}")
        import traceback
        traceback.print_exc()

def create_sample_task_json():
    """Create a sample task JSON that can be used with the API."""
    
    sample_task = {
        "task_type": "ltxv_upscale",
        "video_source_path": "/path/to/your/input/video.mp4",
        "upscale_factor": 2.0,
        "output_path": None  # Optional: specify custom output path
    }
    
    output_file = "sample_ltxv_upscale_task.json"
    with open(output_file, 'w') as f:
        json.dump(sample_task, f, indent=2)
    
    print(f"✅ Created sample task JSON: {output_file}")
    print("You can submit this to the API or modify the video_source_path and submit it.")

if __name__ == "__main__":
    print("LTXV Upscaling Task Test")
    print("=" * 40)
    
    # Create sample JSON file first
    create_sample_task_json()
    print()
    
    # Actually test by adding a task to the database
    test_ltxv_upscale_task()
    
    print("\nHow to use the LTXV upscaling task:")
    print("1. The task has been added to the database (if a video was found)")
    print("2. Run: python headless.py --debug")
    print("3. Watch the headless server process the LTXV upscaling task")
    print("\nExpected workflow:")
    print("• Input video → LTXV VAE encode → Spatial upsampler → LTXV VAE decode → Output video")
    print("• The output video will be 2x the spatial resolution of the input") 