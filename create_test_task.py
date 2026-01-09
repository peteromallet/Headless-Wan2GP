#!/usr/bin/env python3
"""
Create test tasks by duplicating known-good task configurations.

Usage:
    python create_test_task.py travel_orchestrator
    python create_test_task.py qwen_image_style
    python create_test_task.py --list
    python create_test_task.py --all  # Create one of each type
"""

import os
import sys
import json
import uuid
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Test task templates - these are real task configs that exercise the LoRA flow
TEST_TASKS = {
    "uni3c_basic": {
        "description": "Basic Uni3C test - guide video controls motion structure",
        "task_type": "individual_travel_segment",
        "params": {
            # Model config
            "model_name": "wan_2_2_i2v_lightning_baseline_2_2_2",
            "num_frames": 29,
            "parsed_resolution_wh": "902x508",
            "seed_to_use": 42,
            
            # Prompts
            "base_prompt": "",
            "negative_prompt": "",
            
            # Generation settings
            "flow_shift": 5,
            "sample_solver": "euler",
            "guidance_scale": 1,
            "guidance2_scale": 1,
            "guidance_phases": 2,
            "num_inference_steps": 6,
            "switch_threshold": 826.0999755859375,
            "model_switch_phase": 1,
            "cfg_zero_step": -1,
            "cfg_star_switch": 0,
            
            # Segment info
            "segment_index": 0,
            "is_first_segment": True,
            "is_last_segment": True,
            "amount_of_motion": 0.5,
            "debug_mode_enabled": False,
            
            # LoRAs (Lightning 2-phase)
            "lora_names": [
                "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
                "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
            ],
            "lora_multipliers": ["1.0;0", "0;1.0"],
            "additional_loras": {
                "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors": 1.0,
                "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors": 0
            },
            
            # Input images
            "individual_segment_params": {
                "num_frames": 29,
                "base_prompt": "",
                "negative_prompt": "",
                "seed_to_use": 42,
                "random_seed": False,
                "motion_mode": "basic",
                "advanced_mode": False,
                "amount_of_motion": 0.5,
                "start_image_url": "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/41V0rWGAaFwJ4Y9AOqcVC.jpg",
                "end_image_url": "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/e2699835-35d2-4547-85f5-d59219341e4d-u1_3c8779e7-54b4-436c-bfce-9eee8872e370.jpeg",
                "input_image_paths_resolved": [
                    "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/41V0rWGAaFwJ4Y9AOqcVC.jpg",
                    "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/e2699835-35d2-4547-85f5-d59219341e4d-u1_3c8779e7-54b4-436c-bfce-9eee8872e370.jpeg"
                ],
                "after_first_post_generation_brightness": 0,
                "after_first_post_generation_saturation": 1
            },
            "input_image_paths_resolved": [
                "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/41V0rWGAaFwJ4Y9AOqcVC.jpg",
                "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/e2699835-35d2-4547-85f5-d59219341e4d-u1_3c8779e7-54b4-436c-bfce-9eee8872e370.jpeg"
            ],
            "after_first_post_generation_brightness": 0,
            "after_first_post_generation_saturation": 1,
            
            # Uni3C parameters - THE NEW STUFF
            "use_uni3c": True,
            "uni3c_guide_video": "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/guidance-videos/onboarding/structure_video_optimized.mp4",
            "uni3c_strength": 1.0,
            "uni3c_start_percent": 0.0,
            "uni3c_end_percent": 1.0,
            "uni3c_frame_policy": "fit"
        },
        "project_id": "ea5709f3-4592-4d5b-b9a5-87ed2ecf07c9"
    },
    
    "uni3c_strength_test": {
        "description": "Uni3C with strength=0 (should match non-Uni3C output)",
        "task_type": "individual_travel_segment",
        "params": {
            "model_name": "wan_2_2_i2v_lightning_baseline_2_2_2",
            "num_frames": 29,
            "parsed_resolution_wh": "902x508",
            "seed_to_use": 42,
            "base_prompt": "",
            "negative_prompt": "",
            "flow_shift": 5,
            "sample_solver": "euler",
            "guidance_scale": 1,
            "guidance2_scale": 1,
            "guidance_phases": 2,
            "num_inference_steps": 6,
            "switch_threshold": 826.0999755859375,
            "model_switch_phase": 1,
            "cfg_zero_step": -1,
            "cfg_star_switch": 0,
            "segment_index": 0,
            "is_first_segment": True,
            "is_last_segment": True,
            "amount_of_motion": 0.5,
            "debug_mode_enabled": False,
            "lora_names": [
                "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
                "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
            ],
            "lora_multipliers": ["1.0;0", "0;1.0"],
            "additional_loras": {
                "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors": 1.0,
                "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors": 0
            },
            "individual_segment_params": {
                "num_frames": 29,
                "base_prompt": "",
                "negative_prompt": "",
                "seed_to_use": 42,
                "random_seed": False,
                "motion_mode": "basic",
                "advanced_mode": False,
                "amount_of_motion": 0.5,
                "start_image_url": "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/41V0rWGAaFwJ4Y9AOqcVC.jpg",
                "end_image_url": "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/e2699835-35d2-4547-85f5-d59219341e4d-u1_3c8779e7-54b4-436c-bfce-9eee8872e370.jpeg",
                "input_image_paths_resolved": [
                    "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/41V0rWGAaFwJ4Y9AOqcVC.jpg",
                    "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/e2699835-35d2-4547-85f5-d59219341e4d-u1_3c8779e7-54b4-436c-bfce-9eee8872e370.jpeg"
                ],
                "after_first_post_generation_brightness": 0,
                "after_first_post_generation_saturation": 1
            },
            "input_image_paths_resolved": [
                "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/41V0rWGAaFwJ4Y9AOqcVC.jpg",
                "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/e2699835-35d2-4547-85f5-d59219341e4d-u1_3c8779e7-54b4-436c-bfce-9eee8872e370.jpeg"
            ],
            "after_first_post_generation_brightness": 0,
            "after_first_post_generation_saturation": 1,
            
            # Uni3C with strength=0 (effectively disabled)
            "use_uni3c": True,
            "uni3c_guide_video": "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/guidance-videos/onboarding/structure_video_optimized.mp4",
            "uni3c_strength": 0.0
        },
        "project_id": "ea5709f3-4592-4d5b-b9a5-87ed2ecf07c9"
    },
    
    "uni3c_baseline": {
        "description": "Baseline without Uni3C (for comparison)",
        "task_type": "individual_travel_segment",
        "params": {
            "model_name": "wan_2_2_i2v_lightning_baseline_2_2_2",
            "num_frames": 29,
            "parsed_resolution_wh": "902x508",
            "seed_to_use": 42,
            "base_prompt": "",
            "negative_prompt": "",
            "flow_shift": 5,
            "sample_solver": "euler",
            "guidance_scale": 1,
            "guidance2_scale": 1,
            "guidance_phases": 2,
            "num_inference_steps": 6,
            "switch_threshold": 826.0999755859375,
            "model_switch_phase": 1,
            "cfg_zero_step": -1,
            "cfg_star_switch": 0,
            "segment_index": 0,
            "is_first_segment": True,
            "is_last_segment": True,
            "amount_of_motion": 0.5,
            "debug_mode_enabled": False,
            "lora_names": [
                "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
                "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
            ],
            "lora_multipliers": ["1.0;0", "0;1.0"],
            "additional_loras": {
                "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors": 1.0,
                "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors": 0
            },
            "individual_segment_params": {
                "num_frames": 29,
                "base_prompt": "",
                "negative_prompt": "",
                "seed_to_use": 42,
                "random_seed": False,
                "motion_mode": "basic",
                "advanced_mode": False,
                "amount_of_motion": 0.5,
                "start_image_url": "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/41V0rWGAaFwJ4Y9AOqcVC.jpg",
                "end_image_url": "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/e2699835-35d2-4547-85f5-d59219341e4d-u1_3c8779e7-54b4-436c-bfce-9eee8872e370.jpeg",
                "input_image_paths_resolved": [
                    "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/41V0rWGAaFwJ4Y9AOqcVC.jpg",
                    "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/e2699835-35d2-4547-85f5-d59219341e4d-u1_3c8779e7-54b4-436c-bfce-9eee8872e370.jpeg"
                ],
                "after_first_post_generation_brightness": 0,
                "after_first_post_generation_saturation": 1
            },
            "input_image_paths_resolved": [
                "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/41V0rWGAaFwJ4Y9AOqcVC.jpg",
                "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/8a9fdac5-ed89-482c-aeca-c3dd7922d53c/e2699835-35d2-4547-85f5-d59219341e4d-u1_3c8779e7-54b4-436c-bfce-9eee8872e370.jpeg"
            ],
            "after_first_post_generation_brightness": 0,
            "after_first_post_generation_saturation": 1,
            
            # NO Uni3C - baseline for comparison
            "use_uni3c": False
        },
        "project_id": "ea5709f3-4592-4d5b-b9a5-87ed2ecf07c9"
    },
    
    "travel_orchestrator": {
        "description": "Travel orchestrator with 3-phase Lightning LoRAs (VACE model)",
        "task_type": "travel_orchestrator",
        "params": {
            "tool_type": "travel-between-images",
            "orchestrator_details": {
                "steps": 20,
                "run_id": "",  # Will be generated
                "shot_id": "4be72ce7-c223-481b-95a5-b71f15de84ff",
                "seed_base": 789,
                "model_name": "wan_2_2_vace_lightning_baseline_2_2_2",
                "model_type": "vace",
                "base_prompt": "",
                "motion_mode": "basic",
                "phase_config": {
                    "mode": "vace",
                    "phases": [
                        {
                            "loras": [
                                {"url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors", "multiplier": "0.75"},
                                {"url": "https://huggingface.co/peteromallet/random_junk/resolve/main/14b-i2v.safetensors", "multiplier": "0.50"}
                            ],
                            "phase": 1,
                            "guidance_scale": 3
                        },
                        {
                            "loras": [
                                {"url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors", "multiplier": "1.0"},
                                {"url": "https://huggingface.co/peteromallet/random_junk/resolve/main/14b-i2v.safetensors", "multiplier": "0.50"}
                            ],
                            "phase": 2,
                            "guidance_scale": 1
                        },
                        {
                            "loras": [
                                {"url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/low_noise_model.safetensors", "multiplier": "1.0"},
                                {"url": "https://huggingface.co/peteromallet/random_junk/resolve/main/14b-i2v.safetensors", "multiplier": "0.50"}
                            ],
                            "phase": 3,
                            "guidance_scale": 1
                        }
                    ],
                    "flow_shift": 5,
                    "num_phases": 3,
                    "sample_solver": "euler",
                    "steps_per_phase": [2, 2, 2],
                    "model_switch_phase": 2
                },
                "advanced_mode": False,
                "enhance_prompt": True,
                "generation_mode": "timeline",
                "amount_of_motion": 0.5,
                "dimension_source": "project",
                "show_input_images": False,
                "debug_mode_enabled": False,
                "independent_segments": True,
                "orchestrator_task_id": "",  # Will be generated
                "parsed_resolution_wh": "902x508",
                "base_prompts_expanded": [""],
                "frame_overlap_expanded": [10],
                "main_output_dir_for_run": "./outputs/default_travel_output",
                "segment_frames_expanded": [65],
                "selected_phase_preset_id": "__builtin_default_vace__",
                "enhanced_prompts_expanded": [""],
                "negative_prompts_expanded": [""],
                "input_image_generation_ids": ["3a8f129a-d070-4fc1-a6ac-2f694648b1d9"],
                "input_image_paths_resolved": [
                    "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/702a2ebf-569e-4f7d-a7df-78e7c1847000/uploads/1767704204249-u1aseeew.jpg"
                ],
                "num_new_segments_to_generate": 1,
                "after_first_post_generation_brightness": 0,
                "after_first_post_generation_saturation": 1
            }
        },
        "project_id": "ea5709f3-4592-4d5b-b9a5-87ed2ecf07c9"
    },
    
    "qwen_image_style": {
        "description": "Qwen image style with Lightning LoRA phases",
        "task_type": "qwen_image_style",
        "params": {
            "seed": 1788395169,
            "model": "qwen-image",
            "steps": 10,
            "prompt": "A woman in period costume flailing at a duck near a grape arbor, dappled sunlight filtering through vine leaves onto weathered stone",
            "shot_id": "3e4e9f9e-bd93-430e-bb16-955645be6fe1",
            "task_id": "",  # Will be generated
            "resolution": "1353x762",
            "hires_scale": 1,
            "hires_steps": 8,
            "hires_denoise": 0.5,
            "add_in_position": False,
            "style_reference_image": "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/files/1759856102830-678jkcbp.png",
            "subject_reference_image": "https://wczysqzxlwdndgxitrvc.supabase.co/storage/v1/object/public/image_uploads/files/1759856102830-678jkcbp.png",
            "style_reference_strength": 1.1,
            "lightning_lora_strength_phase_1": 0.85,
            "lightning_lora_strength_phase_2": 0.4
        },
        "project_id": "ea5709f3-4592-4d5b-b9a5-87ed2ecf07c9"
    }
}


def create_task(task_type: str, dry_run: bool = False) -> str:
    """Create a test task in Supabase."""
    
    if task_type not in TEST_TASKS:
        print(f"❌ Unknown task type: {task_type}")
        print(f"   Available: {', '.join(TEST_TASKS.keys())}")
        sys.exit(1)
    
    template = TEST_TASKS[task_type]
    
    # Generate unique IDs
    task_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:17]
    
    # Deep copy and update params
    params = json.loads(json.dumps(template["params"]))

    # Update task-specific fields with generated IDs
    if task_type == "travel_orchestrator":
        params["orchestrator_details"]["run_id"] = timestamp
        params["orchestrator_details"]["orchestrator_task_id"] = f"test_travel_{timestamp[:14]}"
    elif task_type == "qwen_image_style":
        params["task_id"] = f"test_qwen_{timestamp[:14]}"
    
    task_data = {
        "id": task_id,
        "task_type": template["task_type"],
        "params": params,
        "status": "Queued",
        "project_id": template["project_id"],
        "attempts": 0
    }
    
    if dry_run:
        print(f"\n🔍 DRY RUN - Would create {task_type} task:")
        print(f"   ID: {task_id}")
        print(f"   Type: {template['task_type']}")
        print(f"   Description: {template['description']}")
        print(f"\n   Params preview:")
        print(json.dumps(params, indent=2)[:500] + "...")
        return task_id
    
    # Connect to Supabase
    from supabase import create_client
    
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    
    if not url or not key:
        print("❌ SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        print("   Add them to .env file or export them")
        sys.exit(1)
    
    client = create_client(url, key)
    
    # Insert task
    result = client.table("tasks").insert(task_data).execute()
    
    if result.data:
        print(f"\n✅ Created {task_type} task:")
        print(f"   ID: {task_id}")
        print(f"   Type: {template['task_type']}")
        print(f"   Description: {template['description']}")
        print(f"\n   Debug: python debug.py task {task_id}")
        return task_id
    else:
        print(f"❌ Failed to create task")
        sys.exit(1)


def list_tasks():
    """List available test task templates."""
    print("\n📋 Available test task templates:\n")
    for name, template in TEST_TASKS.items():
        print(f"  {name}")
        print(f"    Type: {template['task_type']}")
        print(f"    Description: {template['description']}")
        print()


def create_all_tasks(dry_run: bool = False):
    """Create one task of each type."""
    print(f"\n{'🔍 DRY RUN - ' if dry_run else ''}Creating one task of each type...\n")

    created_ids = []
    for task_type in TEST_TASKS.keys():
        try:
            task_id = create_task(task_type, dry_run=dry_run)
            created_ids.append((task_type, task_id))
            print()  # Add spacing between tasks
        except Exception as e:
            print(f"❌ Failed to create {task_type}: {e}\n")

    if created_ids and not dry_run:
        print("\n" + "=" * 80)
        print(f"✅ Successfully created {len(created_ids)} test tasks:")
        for task_type, task_id in created_ids:
            print(f"   {task_type:<25} → {task_id}")


def main():
    parser = argparse.ArgumentParser(description="Create test tasks for worker testing")
    parser.add_argument("task_type", nargs="?", help="Task type to create (travel_orchestrator, qwen_image_style)")
    parser.add_argument("--list", "-l", action="store_true", help="List available task templates")
    parser.add_argument("--all", "-a", action="store_true", help="Create one task of each type")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be created without creating")

    args = parser.parse_args()

    if args.list:
        list_tasks()
        return

    if args.all:
        create_all_tasks(dry_run=args.dry_run)
        return

    if not args.task_type:
        parser.print_help()
        print("\n")
        list_tasks()
        return

    create_task(args.task_type, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

