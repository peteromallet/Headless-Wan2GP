#!/usr/bin/env python3
"""Test different model configurations: vace_14B, vace_14B_fake_cocktail_2_2, and optimised-t2i."""

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


def run_model_comparison_tests(base_output_dir: str = "outputs/model_comparison_test"):
    """Run tasks testing different models with appropriate inputs."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_run_id = f"model_comparison_{timestamp}"
    
    # Define test configurations
    test_configs = [
        {
            "model": "vace_14B",
            "name": "Vace ControlNet 14B",
            "test_type": "vace_video",
            "video": "video.mp4",
            "mask": "mask.mp4", 
            "prompt": "camera rotates around him",
            "params": {
                "video_prompt_type": "VM",
                "resolution": "768x576",
                "video_length": 65,
                "num_inference_steps": 10,
                "guidance_scale": 3,
                "flow_shift": 2,
                "seed": 12345,
                "negative_prompt": "blurry, low quality, distorted",
                "control_net_weight": 1.0,
                "control_net_weight2": 1.0,
                "switch_threshold": 500,
                "guidance2_scale": 1
            }
        },
        {
            "model": "vace_14B_fake_cocktail_2_2", 
            "name": "Fake Vace Cocktail 14B",
            "test_type": "vace_video",
            "video": "video.mp4",
            "mask": "mask.mp4",
            "prompt": "he runs into the kitchen and starts cooking",
            "params": {
                "video_prompt_type": "VM",
                "resolution": "768x576", 
                "video_length": 65,
                "num_inference_steps": 10,
                "guidance_scale": 3,
                "flow_shift": 2,
                "seed": 12345,
                "negative_prompt": "blurry, low quality, distorted",
                "control_net_weight": 1.0,
                "control_net_weight2": 1.0,
                "switch_threshold": 500,
                "guidance2_scale": 1
            }
        },
        {
            "model": "optimised-t2i",
            "name": "Optimised T2I 14B",
            "test_type": "single_image",
            "image": "image.jpg",  # Single image input
            "prompt": "a beautiful landscape with mountains and a lake at sunset",
            "params": {
                "resolution": "768x576",
                "video_length": 65,
                "num_inference_steps": 10,
                "guidance_scale": 1,
                "flow_shift": 2,
                "seed": 12345,
                "negative_prompt": "blurry, low quality, distorted",
                "switch_threshold": 950,
                "guidance2_scale": 1
            }
        }
    ]
    
    # Validate input files exist
    for config in test_configs:
        if config["test_type"] == "vace_video":
            video_path = Path(config["video"]).absolute()
            mask_path = Path(config["mask"]).absolute()
            if not video_path.exists():
                print(f"Error: Video file not found: {video_path}")
                return [], test_run_id, None
            if not mask_path.exists():
                print(f"Error: Mask file not found: {mask_path}")
                return [], test_run_id, None
        elif config["test_type"] == "single_image":
            image_path = Path(config["image"]).absolute()
            if not image_path.exists():
                print(f"Error: Image file not found: {image_path}")
                return [], test_run_id, None
    
    print(f"Testing {len(test_configs)} different model configurations:")
    for i, config in enumerate(test_configs, 1):
        print(f"  {i}. {config['name']} ({config['model']}) - {config['test_type']}")
    
    output_dir = (Path(base_output_dir).absolute() / test_run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize orchestrator
    wan_root = Path(__file__).parent / "Wan2GP"
    orchestrator = WanOrchestrator(str(wan_root))
    
    results = []
    
    for config_idx, config in enumerate(test_configs, 1):
        print(f"\n--- Test {config_idx}/{len(test_configs)}: {config['name']} ---")
        print(f"Model: {config['model']}")
        print(f"Type: {config['test_type']}")
        print(f"Prompt: \"{config['prompt']}\"")
        
        start_time = time.time()
        
        try:
            # Load the specific model
            orchestrator.load_model(config["model"])
            
            if config["test_type"] == "vace_video":
                video_path = Path(config["video"]).absolute()
                mask_path = Path(config["mask"]).absolute()
                
                print(f"Video: {config['video']}")
                print(f"Mask: {config['mask']}")
                
                output_path = orchestrator.generate_vace(
                    prompt=config['prompt'],
                    video_guide=str(video_path),
                    video_mask=str(mask_path),
                    **config['params']
                )
                
            elif config["test_type"] == "single_image":
                image_path = Path(config["image"]).absolute()
                
                print(f"Image: {config['image']}")
                
                output_path = orchestrator.generate_i2v(
                    prompt=config['prompt'],
                    image=str(image_path),
                    resolution=config['params']['resolution'],
                    video_length=config['params']['video_length'],
                    num_inference_steps=config['params']['num_inference_steps'],
                    guidance_scale=config['params']['guidance_scale'],
                    flow_shift=config['params']['flow_shift'],
                    seed=config['params']['seed'],
                    negative_prompt=config['params']['negative_prompt'],
                    switch_threshold=config['params']['switch_threshold'],
                    guidance2_scale=config['params']['guidance2_scale']
                )
            
            generation_time = time.time() - start_time
            
            if output_path and Path(output_path).exists():
                output_filename = f"{config['model']}_test.mp4"
                final_path = output_dir / output_filename
                Path(output_path).rename(final_path)
                
                print(f"✅ Completed in {generation_time:.1f}s -> {output_filename}")
                
                results.append({
                    "model": config["model"],
                    "model_name": config["name"],
                    "test_type": config["test_type"],
                    "prompt": config["prompt"],
                    "input_files": {
                        "video": config.get("video"),
                        "mask": config.get("mask"),
                        "image": config.get("image")
                    },
                    "output_path": str(final_path),
                    "generation_time": generation_time,
                    "file_size_mb": final_path.stat().st_size / (1024 * 1024),
                    "parameters": config["params"],
                    "status": "success"
                })
            else:
                print(f"⚠️ No output found")
                results.append({
                    "model": config["model"],
                    "model_name": config["name"],
                    "test_type": config["test_type"],
                    "prompt": config["prompt"],
                    "status": "no_output"
                })
                
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                "model": config["model"],
                "model_name": config["name"], 
                "test_type": config["test_type"],
                "prompt": config["prompt"],
                "status": "failed",
                "error": str(e)
            })
        
        finally:
            # Unload model between tests
            orchestrator.unload_model()
        
        # Brief pause between tests
        if config_idx < len(test_configs):
            time.sleep(3)
    
    return results, test_run_id, output_dir


def save_results(results: list, test_run_id: str, output_dir: Path):
    """Save test results to JSON."""
    
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "test_run_id": test_run_id,
            "timestamp": datetime.now().isoformat(),
            "description": "Model comparison test: vace_14B vs vace_14B_fake_cocktail_2_2 vs optimised-t2i",
            "results": results
        }, f, indent=2)
    
    return results_file


def print_summary(results: list):
    """Print a summary of test results."""
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON TEST SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]
    no_output = [r for r in results if r.get("status") == "no_output"]
    
    print(f"Total Tests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"No Output: {len(no_output)}")
    
    if successful:
        print(f"\n{'Successful Tests':<30} {'Time (s)':<10} {'Size (MB)':<10}")
        print("-" * 50)
        for result in successful:
            time_str = f"{result['generation_time']:.1f}"
            size_str = f"{result['file_size_mb']:.1f}"
            print(f"{result['model']:<30} {time_str:<10} {size_str:<10}")
    
    if failed:
        print(f"\nFailed Tests:")
        for result in failed:
            print(f"  ❌ {result['model']}: {result.get('error', 'Unknown error')}")


def main():
    parser = argparse.ArgumentParser(description="Test different model configurations")
    parser.add_argument("--output-dir", default="outputs/model_comparison_test", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        results, test_run_id, output_dir = run_model_comparison_tests(
            base_output_dir=args.output_dir
        )
        
        if not output_dir:
            print("Failed: Missing input files")
            sys.exit(1)
            
        results_file = save_results(results, test_run_id, output_dir)
        
        print_summary(results)
        print(f"\nResults saved: {results_file}")
        print(f"Videos saved: {output_dir}")
        
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
