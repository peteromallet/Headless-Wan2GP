#!/usr/bin/env python3
"""
Quick test to verify parameter override fix.
This creates a simple test task to check if model config parameters are properly respected.
"""

import json
import sys
from pathlib import Path

# Add our project to Python path
sys.path.insert(0, str(Path(__file__).parent))

from source.common_utils import generate_unique_task_id

def create_test_task():
    """Create a test task to verify parameter handling"""
    
    test_task_params = {
        "task_type": "travel_orchestrator",
        "model_name": "vace_14B_fake_cocktail_2_2",
        "prompt": "Test parameter fix - camera rotates around subject",
        "negative_prompt": "blurry, low quality",
        "resolution": "768x576", 
        "segment_frames": [65],
        "frame_overlap": [8],
        "seed_base": 789,
        "fps_helpers": 16,
        "project_id": "test-param-fix-project",
        "debug_mode_enabled": True,
        "orchestrator_class": "travel_between_images",
        "image_refs": [
            {
                "download_url": "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/87b3fed9-6caa-46b8-85e1-24fc6026bf20/1.png",
                "local_path": "samples/1.png",
                "segment_idx_for_naming": 0
            }
        ]
    }
    
    task_payload_str = json.dumps(test_task_params, indent=2)
    print("Test task payload:")
    print(task_payload_str)
    print()
    print("Expected behavior:")
    print("1. Model config should load guidance_scale=1, flow_shift=2, num_inference_steps=10")
    print("2. Wan 2.2 optimizations should apply guidance_scale=1.0, flow_shift=2.0") 
    print("3. Final generation should use guidance_scale=1.0 (NOT 5.0)")
    print("4. CausVid warnings should be reduced or eliminated")
    print()
    print("To test:")
    print("python add_task.py --json-payload '{}'".format(task_payload_str.replace("'", "\\'")))

if __name__ == "__main__":
    create_test_task()
