#!/usr/bin/env python3
"""
Quick Dual-VACE Task Creator
Creates a dual-VACE task with first/last frame references and context frames
"""

import os
import sys
import json
import datetime
import uuid
from pathlib import Path

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent / "source"))

def create_quick_dual_vace_task():
    """Creates a dual-VACE task directly using the database functions"""
    
    # Import database functions
    from db_operations import add_task_to_db, init_db
    
    # Initialize database
    init_db()
    
    # Generate task ID
    task_id = f"dual_vace_{int(datetime.datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"
    
    # Create the task payload
    task_payload = {
        "task_id": task_id,
        "prompt": "high quality video",
        "negative_prompt": "ugly, blurry, distorted, low quality",
        "resolution": "720x720",
        "video_length": 40,
        "num_inference_steps": 9,
        "seed": 42,
        "guidance_scale": 1.0,
        "flow_shift": 5.0,
        "use_causvid_lora": True,
        "apply_reward_lora": True,
        "multi_vace_inputs": [
            {
                # VACE Encoding 1: First and last frame references
                "ref_image_paths": ["tests/ref_frame_0.png", "tests/ref_frame_39.png"],
                "strength": 0.25,
                "start_percent": 0.0,
                "end_percent": 1.0,
                "description": "First and Last Frame References"
            },
            {
                # VACE Encoding 2: Context frames (you'll need to create these)
                "frame_paths": [f"tests/context_frame_{i:04d}.png" for i in range(40)],
                "strength": 1.0,
                "start_percent": 0.0,
                "end_percent": 0.9,
                "description": "Context Frames with Grey Fill"
            }
        ],
        "offload_model": True,
        "VAE_tile_size": 0,
        "RIFLEx_setting": 1,
        "joint_pass": False,
        "project_id": "dual_vace_project"
    }
    
    # Add task to database
    try:
        add_task_to_db(task_payload, "dual_vace_task")
        print(f"\n✅ Dual-VACE task created successfully!")
        print(f"📝 Task ID: {task_id}")
        print(f"📊 Status: Queued")
        print(f"🎯 VACE Encoding 1: First + Last frame references")
        print(f"🎯 VACE Encoding 2: Context frames with grey fill")
        print(f"\n🚀 Run 'python headless.py' to process the task!")
        return task_id
        
    except Exception as e:
        print(f"❌ Failed to add task to database: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Creating Quick Dual-VACE Task...")
    task_id = create_quick_dual_vace_task()
    
    if task_id:
        print(f"\n🎉 SUCCESS! Task {task_id} is ready for processing!")
    else:
        print(f"\n❌ FAILED to create task") 