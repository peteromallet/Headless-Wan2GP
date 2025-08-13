#!/usr/bin/env python3
"""Test switch_threshold values with vace_14B_fake_cocktail_2_2 model."""

import json
import sys
import argparse
import time
import os
from pathlib import Path
from datetime import datetime

# Add our project to Python path
sys.path.insert(0, str(Path(__file__).parent))

from source.common_utils import generate_unique_task_id
from headless_wgp import WanOrchestrator


def run_switch_threshold_tests(base_output_dir: str = "outputs/switch_threshold_test"):
    """Run tasks testing different switch_threshold values across multiple video pairs."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_run_id = f"switch_test_{timestamp}"
    threshold_values = [500, 600, 700, 800, 900]
    
    # Define video pairs to test
    video_pairs = [
        {"video": "video_1.mp4", "mask": "mask_1.mp4", "prompt": "camera rotates around him"},
        {"video": "video_2.mp4", "mask": "mask_2.mp4", "prompt": "he runs into the kitchen and starts cooking"},
        {"video": "video_3.mp4", "mask": "mask_3.mp4", "prompt": "it zooms out from his eye to show him reading"}
    ]
    
    # Validate all video files exist
    for pair in video_pairs:
        video_path = Path(pair["video"]).absolute()
        mask_path = Path(pair["mask"]).absolute()
        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}")
            return [], test_run_id, None
        if not mask_path.exists():
            print(f"Error: Mask file not found: {mask_path}")
            return [], test_run_id, None
    
    print(f"Testing {len(video_pairs)} video pairs × {len(threshold_values)} thresholds = {len(video_pairs) * len(threshold_values)} total tasks")
    print(f"Video pairs: {[pair['video'] for pair in video_pairs]}")
    print(f"Threshold values: {threshold_values}")
    
    # Initialize and load model
    wan_root = Path(__file__).parent / "Wan2GP"
    orchestrator = WanOrchestrator(str(wan_root))
    orchestrator.load_model("vace_14B_fake_cocktail_2_2")
    
    output_dir = (Path(base_output_dir).absolute() / test_run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    task_num = 0
    total_tasks = len(video_pairs) * len(threshold_values)
    
    for video_idx, pair in enumerate(video_pairs, 1):
        video_path = Path(pair["video"]).absolute()
        mask_path = Path(pair["mask"]).absolute()
        
        print(f"\n--- Video Pair {video_idx}/{len(video_pairs)}: {pair['video']} + {pair['mask']} ---")
        print(f"Prompt: \"{pair['prompt']}\"")
        
        for threshold_idx, switch_threshold in enumerate(threshold_values, 1):
            task_num += 1
            print(f"\nTask {task_num}/{total_tasks}: video_{video_idx}_threshold_{switch_threshold}")
            
            start_time = time.time()
            
            try:
                output_path = orchestrator.generate_vace(
                    prompt=pair['prompt'],
                    video_guide=str(video_path),
                    video_mask=str(mask_path),
                    video_prompt_type="VM",
                    resolution="768x576",
                    video_length=65,
                    num_inference_steps=10,
                    guidance_scale=3,
                    flow_shift=2,
                    seed=12345,
                    negative_prompt="blurry, low quality, distorted",
                    control_net_weight=1.0,
                    control_net_weight2=1.0,
                    switch_threshold=switch_threshold,
                    guidance2_scale=1
                )
                
                generation_time = time.time() - start_time
                
                if output_path and Path(output_path).exists():
                    output_filename = f"video_{video_idx}_threshold_{switch_threshold}.mp4"
                    final_path = output_dir / output_filename
                    Path(output_path).rename(final_path)
                    
                    print(f"✅ Completed in {generation_time:.1f}s -> {output_filename}")
                    
                    results.append({
                        "video_pair": video_idx,
                        "video_file": pair["video"],
                        "mask_file": pair["mask"],
                        "prompt": pair["prompt"],
                        "switch_threshold": switch_threshold,
                        "output_path": str(final_path),
                        "generation_time": generation_time,
                        "file_size_mb": final_path.stat().st_size / (1024 * 1024),
                        "status": "success"
                    })
                else:
                    print(f"⚠️ No output found")
                    results.append({
                        "video_pair": video_idx,
                        "video_file": pair["video"], 
                        "mask_file": pair["mask"],
                        "prompt": pair["prompt"],
                        "switch_threshold": switch_threshold,
                        "status": "no_output"
                    })
                    
            except Exception as e:
                print(f"❌ Failed: {e}")
                results.append({
                    "video_pair": video_idx,
                    "video_file": pair["video"],
                    "mask_file": pair["mask"],
                    "prompt": pair["prompt"],
                    "switch_threshold": switch_threshold,
                    "status": "failed",
                    "error": str(e)
                })
            
            if task_num < total_tasks:
                time.sleep(2)
    
    orchestrator.unload_model()
    return results, test_run_id, output_dir


def save_results(results: list, test_run_id: str, output_dir: Path):
    """Save test results to JSON."""
    
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "test_run_id": test_run_id,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)
    
    return results_file


def main():
    parser = argparse.ArgumentParser(description="Test switch_threshold values across multiple video pairs")
    parser.add_argument("--output-dir", default="outputs/switch_threshold_test", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        results, test_run_id, output_dir = run_switch_threshold_tests(
            base_output_dir=args.output_dir
        )
        
        if not output_dir:
            print("Failed: Missing video files")
        sys.exit(1)
            
        results_file = save_results(results, test_run_id, output_dir)
        
        successful = [r for r in results if r.get("status") == "success"]
        print(f"\nCompleted: {len(successful)}/{len(results)} successful")
        print(f"Results: {results_file}")
        print(f"Videos: {output_dir}")
        
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()