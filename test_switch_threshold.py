#!/usr/bin/env python3
"""
Switch Threshold Testing Script

This script creates 7 tasks using vace_14B_fake_cocktail_2_2 model with different 
switch_threshold values (100, 200, 300, 400, 500, 600, 700) to test the dual-model switching behavior.

Usage:
    # Use all defaults (video.mp4, mask.mp4, "camera rotates around him")
    python test_switch_threshold.py
    
    # Use custom files with default prompt
    python test_switch_threshold.py my_video.mp4 my_mask.mp4
    
    # Use custom files and prompt
    python test_switch_threshold.py my_video.mp4 my_mask.mp4 "custom prompt here"
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add our project to Python path
sys.path.insert(0, str(Path(__file__).parent))

from source.common_utils import generate_unique_task_id


def create_switch_threshold_test_tasks(video_path: str, mask_path: str, prompt: str, 
                                     base_output_dir: str = "outputs/switch_threshold_test",
                                     project_id: str = None):
    """
    Create 10 tasks testing different switch_threshold values.
    
    Args:
        video_path: Path to the guide video
        mask_path: Path to the mask video  
        prompt: Generation prompt
        base_output_dir: Base directory for outputs
        project_id: Optional project ID for organization
    """
    
    # Generate a unique test run ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_run_id = f"switch_test_{timestamp}"
    
    if project_id is None:
        project_id = f"switch-threshold-test-{timestamp}"
    
    tasks = []
    
    print(f"🧪 Creating Switch Threshold Test Suite")
    print(f"📁 Test Run ID: {test_run_id}")
    print(f"🎬 Video: {video_path}")
    print(f"🎭 Mask: {mask_path}")
    print(f"💭 Prompt: {prompt}")
    # Define specific threshold values to test
    threshold_values = [100, 200, 300, 400, 500, 600, 700]
    print(f"📊 Testing switch_threshold values: {threshold_values}")
    print()
    
    for i, switch_threshold in enumerate(threshold_values):
        
        task_params = {
            "task_type": "travel_orchestrator",
            "model_name": "vace_14B_fake_cocktail_2_2",
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, distorted",
            "resolution": "768x576",
            "segment_frames": [65],  # Single segment
            "frame_overlap": [0],    # No overlap needed for single segment
            "seed_base": 12345,
            "fps_helpers": 16,
            "project_id": project_id,
            "debug_mode_enabled": True,
            "orchestrator_class": "travel_between_images",
            
            # 🎯 KEY TEST PARAMETER
            "switch_threshold": switch_threshold,
            
            # Custom output naming for easy comparison
            "custom_output_dir": f"{base_output_dir}/{test_run_id}",
            "output_filename_prefix": f"switch_threshold_{switch_threshold}_",
            
            # Video guide and mask
            "image_refs": [
                {
                    "download_url": video_path,
                    "local_path": video_path,
                    "segment_idx_for_naming": 0,
                    "is_video_guide": True
                }
            ],
            "mask_refs": [
                {
                    "download_url": mask_path,
                    "local_path": mask_path,
                    "segment_idx_for_naming": 0
                }
            ]
        }
        
        task_id = generate_unique_task_id(f"switch_test_{switch_threshold}_")
        task_params["task_id"] = task_id
        
        tasks.append({
            "task_id": task_id,
            "switch_threshold": switch_threshold,
            "params": task_params
        })
        
        print(f"📋 Task {i+1:2d}/7: threshold={switch_threshold:3d} → {task_id}")
    
    return tasks, test_run_id


def save_tasks_batch(tasks: list, output_file: str = None):
    """Save all tasks to a JSON file for batch processing."""
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"switch_threshold_test_batch_{timestamp}.json"
    
    batch_data = {
        "test_suite": "switch_threshold_comparison",
        "model": "vace_14B_fake_cocktail_2_2",
        "parameter_tested": "switch_threshold",
        "values_tested": [task["switch_threshold"] for task in tasks],
        "total_tasks": len(tasks),
        "tasks": [task["params"] for task in tasks]
    }
    
    with open(output_file, 'w') as f:
        json.dump(batch_data, f, indent=2)
    
    print(f"💾 Saved {len(tasks)} tasks to: {output_file}")
    return output_file


def generate_add_task_commands(tasks: list):
    """Generate individual add_task.py commands for each task."""
    
    commands = []
    
    print("\n🔧 Individual add_task.py commands:")
    print("=" * 60)
    
    for i, task in enumerate(tasks):
        task_json = json.dumps(task["params"])
        # Escape quotes for shell
        task_json_escaped = task_json.replace('"', '\\"')
        
        command = f'python add_task.py --json-payload "{task_json_escaped}"'
        commands.append(command)
        
        print(f"# Task {i+1}: switch_threshold={task['switch_threshold']}")
        print(command)
        print()
    
    return commands


def generate_comparison_analysis():
    """Generate a comparison analysis template."""
    
    analysis_template = """
# Switch Threshold Analysis Template

## Expected Results:

### switch_threshold = 100-200 (Early Switch)  
- Quick switch to Low model (refinement focused)
- Should have excellent detail but potentially less structure

### switch_threshold = 300-400 (Early-Mid Switch)
- Balanced toward detail refinement
- Should provide good detail with decent structure

### switch_threshold = 500 (Current Default)
- Balanced between structure and detail
- Should provide good overall quality

### switch_threshold = 600-700 (Late Switch)
- High model dominates, Low model for final touches
- Should have excellent structure with good detail

## Analysis Questions:

1. **Quality Sweet Spot**: Which switch_threshold provides the best overall quality?
2. **Detail vs Structure**: How does early vs late switching affect the balance?
3. **Model Efficiency**: Do earlier thresholds generate faster due to more Low model usage?
4. **Prompt Adherence**: Which threshold best follows the prompt instructions?
5. **Consistency**: Which threshold produces the most consistent results?

## Comparison Metrics:

- [ ] Visual quality assessment (1-10 scale)
- [ ] Prompt adherence (1-10 scale)  
- [ ] Detail preservation (1-10 scale)
- [ ] Structural coherence (1-10 scale)
- [ ] Generation time
- [ ] Final video file size

"""
    
    analysis_file = f"switch_threshold_analysis_template.md"
    with open(analysis_file, 'w') as f:
        f.write(analysis_template)
    
    print(f"📊 Analysis template saved to: {analysis_file}")
    return analysis_file


def main():
    parser = argparse.ArgumentParser(description="Generate switch_threshold test tasks for vace_14B_fake_cocktail_2_2")
    parser.add_argument("video", nargs='?', default="video.mp4", help="Path to guide video (default: video.mp4)")
    parser.add_argument("mask", nargs='?', default="mask.mp4", help="Path to mask video (default: mask.mp4)")
    parser.add_argument("prompt", nargs='?', default="camera rotates around him", help="Generation prompt (default: 'camera rotates around him')")
    parser.add_argument("--project-id", help="Project ID for organization")
    parser.add_argument("--output-dir", default="outputs/switch_threshold_test", 
                       help="Base output directory")
    parser.add_argument("--batch-file", help="Custom batch file name")
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.video).exists():
        print(f"❌ Error: Video file not found: {args.video}")
        sys.exit(1)
        
    if not Path(args.mask).exists():
        print(f"❌ Error: Mask file not found: {args.mask}")
        sys.exit(1)
    
    # Create test tasks
    tasks, test_run_id = create_switch_threshold_test_tasks(
        video_path=args.video,
        mask_path=args.mask, 
        prompt=args.prompt,
        base_output_dir=args.output_dir,
        project_id=args.project_id
    )
    
    # Save batch file
    batch_file = save_tasks_batch(tasks, args.batch_file)
    
    # Generate individual commands
    commands = generate_add_task_commands(tasks)
    
    # Generate analysis template
    analysis_file = generate_comparison_analysis()
    
    print("\n🎯 Test Suite Ready!")
    print("=" * 60)
    print(f"📁 Test Run ID: {test_run_id}")
    print(f"📋 Tasks Created: {len(tasks)}")
    print(f"💾 Batch File: {batch_file}")
    print(f"📊 Analysis Template: {analysis_file}")
    print(f"🎬 Expected Outputs: {args.output_dir}/{test_run_id}/")
    print()
    print("🚀 To run all tasks:")
    print(f"   for i in {{0..6}}; do python add_task.py --json-payload \"$(jq -r '.tasks[$i]' {batch_file})\"; done")
    print()
    print("🔍 Monitor progress and compare results to find optimal switch_threshold!")


if __name__ == "__main__":
    main()
