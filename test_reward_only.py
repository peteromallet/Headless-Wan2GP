#!/usr/bin/env python3
"""Test reward LoRA strength only with proper phase filtering"""

import json
import sys
import argparse
import time
import os
import random
from pathlib import Path
from datetime import datetime

# Add our project to Python path
sys.path.insert(0, str(Path(__file__).parent))

from source.common_utils import generate_unique_task_id
from headless_wgp import WanOrchestrator


def generate_reward_lora_tests(video_numbers=[1]):
    """Generate only reward LoRA strength test configurations"""

    # Define video configurations
    all_video_configs = [
        {
            "video": "samples/video1.mp4",
            "mask": "samples/mask1.mp4",
            "resolution": "768x576",
            "video_number": 1
        },
        {
            "video": "samples/video2.mp4",
            "mask": "samples/mask2.mp4",
            "resolution": "496x896",
            "video_number": 2
        }
    ]

    # Filter video configurations
    video_configs = [config for config in all_video_configs if config["video_number"] in video_numbers]

    base_config = {
        "model": "vace_fun_14B_cocktail_lightning",
        "name": "Vace Fun Cocktail Lightning 14B",
        "test_type": "vace_video",
        "video_configs": video_configs,
        "prompt": "zooming out timelapse of a plant growing as the sun fades and the moon moves across the sky and becomes the sun, timlapsiagro",
        "base_params": {
            "video_prompt_type": "VM",
            "video_length": 81,
            "num_inference_steps": 6,
            "guidance_scale": 1,
            "guidance2_scale": 1,
            "flow_shift": 5,
            "seed": 12345,
            "negative_prompt": "blurry, low quality, distorted, static, overexposed",
            "control_net_weight": 1.0,
            "control_net_weight2": 1.0,
            "sample_solver": "unipc"
        }
    }

    test_configs = []
    test_id = 1
    base_switch_threshold = 875

    print("Generating Phase 6: Reward LoRA strength variations...")

    # Generate strength values: 1.0, 0.75, 0.5, 0.25
    reward_strength_values = [round(1.0 - (i * 0.25), 2) for i in range(4)]

    for reward_strength in reward_strength_values:
        for video_config in video_configs:
            prompt = "zooming out timelapse of a plant growing as the sun fades and the moon moves across the sky and becomes the sun, timlapsiagro" if video_config["video_number"] == 1 else "he turns around and runs to the other painting"

            config = {
                **base_config,
                "test_id": test_id,
                "phase": "reward_lora_strength",
                "video": video_config["video"],
                "mask": video_config["mask"],
                "video_number": video_config["video_number"],
                "prompt": prompt,
                "variation": f"reward_strength_{reward_strength}_video{video_config['video_number']}",
                "description": f"Reward LoRA strength: {reward_strength} (video{video_config['video_number']})",
                "params": {
                    **base_config["base_params"],
                    "resolution": video_config["resolution"],
                    "switch_threshold": base_switch_threshold,
                    # Modulate reward LoRAs (positions 2 and 3 in the multipliers)
                    "loras_multipliers": f"3.0;0;{reward_strength};{reward_strength}" + (";1.3" if video_config["video_number"] == 1 else ""),
                    # Add bloom LoRA for video1
                    **({
                        "activated_loras": ["Wan2GP/loras/bloom.safetensors"]
                    } if video_config["video_number"] == 1 else {})
                }
            }
            test_configs.append(config)
            test_id += 1

    print(f"Generated {len(test_configs)} test configurations:")
    print(f"  - Phase 6 (Reward LoRA strength): {len(test_configs)} tests (4 parameters × {len(video_configs)} videos)")
    print(f"  - Total: {len(test_configs)} tests using video{video_configs[0]['video_number']}.mp4 only" if len(video_configs) == 1 else f"  - Total: {len(test_configs)} tests using both videos")

    return test_configs


def run_reward_lora_tests(base_output_dir="tests_output", video_numbers=[1]):
    """Run reward LoRA strength tests only"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_run_id = f"reward_lora_{timestamp}"

    test_configs = generate_reward_lora_tests(video_numbers)

    output_dir = (Path(base_output_dir).absolute() / test_run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize orchestrator
    wan_root = Path(__file__).parent / "Wan2GP"
    orchestrator = WanOrchestrator(str(wan_root))

    results = []

    print(f"\n{'='*70}")
    print(f"REWARD LORA STRENGTH TEST")
    print(f"{'='*70}")
    print(f"Model: vace_fun_14B_cocktail_lightning")
    print(f"Total tests: {len(test_configs)}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")

    # Load model
    try:
        print(f"\nLoading model: vace_fun_14B_cocktail_lightning...")
        orchestrator.load_model("vace_fun_14B_cocktail_lightning")
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return [], test_run_id, output_dir

    try:
        for config in test_configs:
            test_id = config["test_id"]
            total_tests = len(test_configs)

            print(f"\n--- Test {test_id}/{total_tests}: {config['description']} ---")
            print(f"Variation: {config['variation']}")

            start_time = time.time()

            try:
                video_path = Path(config['video']).absolute()
                mask_path = Path(config['mask']).absolute()

                output_path = orchestrator.generate_vace(
                    prompt=config['prompt'],
                    video_guide=str(video_path),
                    video_mask=str(mask_path),
                    **config['params']
                )

                generation_time = time.time() - start_time

                if output_path and Path(output_path).exists():
                    video_num = config.get('video_number', 1)
                    reward_strength_str = config["params"]["loras_multipliers"].split(";")[2]
                    reward_strength_clean = reward_strength_str.replace(".", "_")
                    output_filename = f"video{video_num}_reward_lora_{reward_strength_clean}.mp4"

                    final_path = output_dir / output_filename
                    Path(output_path).rename(final_path)

                    file_size_mb = final_path.stat().st_size / (1024 * 1024)

                    print(f"✅ Completed in {generation_time:.1f}s -> {output_filename} ({file_size_mb:.1f}MB)")

                    results.append({
                        "test_id": test_id,
                        "phase": config["phase"],
                        "variation": config["variation"],
                        "description": config["description"],
                        "model": config["model"],
                        "prompt": config["prompt"],
                        "output_path": str(final_path),
                        "output_filename": output_filename,
                        "generation_time": generation_time,
                        "file_size_mb": file_size_mb,
                        "parameters": config["params"],
                        "status": "success"
                    })
                else:
                    print(f"⚠️ No output found")
                    results.append({
                        "test_id": test_id,
                        "phase": config["phase"],
                        "variation": config["variation"],
                        "description": config["description"],
                        "status": "no_output"
                    })

            except Exception as e:
                generation_time = time.time() - start_time
                print(f"❌ Failed after {generation_time:.1f}s: {e}")
                results.append({
                    "test_id": test_id,
                    "phase": config["phase"],
                    "variation": config["variation"],
                    "description": config["description"],
                    "status": "failed",
                    "error": str(e),
                    "generation_time": generation_time
                })

            # Brief pause between tests
            if test_id < len(test_configs):
                time.sleep(2)

    finally:
        # Unload model
        print(f"\nUnloading model...")
        orchestrator.unload_model()
        print("✅ Model unloaded")

    return results, test_run_id, output_dir


def main():
    parser = argparse.ArgumentParser(description="Test reward LoRA strength variations only")
    parser.add_argument("--output-dir", default="tests_output",
                       help="Output directory for test results")
    parser.add_argument("--videos", type=int, nargs='+', default=[1], choices=[1, 2],
                       help="Video numbers to test with (1, 2, or both). Default: [1]")

    args = parser.parse_args()

    try:
        results, test_run_id, output_dir = run_reward_lora_tests(
            base_output_dir=args.output_dir,
            video_numbers=args.videos
        )

        # Save simple results
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "test_run_id": test_run_id,
                "timestamp": datetime.now().isoformat(),
                "model": "vace_fun_14B_cocktail_lightning",
                "description": "Reward LoRA strength testing: 1.0, 0.75, 0.5, 0.25",
                "results": results
            }, f, indent=2)

        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "failed"]

        print(f"\n{'='*50}")
        print("REWARD LORA TEST SUMMARY")
        print(f"{'='*50}")
        print(f"Total Tests: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")

        if successful:
            avg_time = sum(r["generation_time"] for r in successful) / len(successful)
            print(f"Average Generation Time: {avg_time:.1f}s")

        print(f"\nResults saved: {results_file}")
        print(f"Video outputs saved: {output_dir}")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()