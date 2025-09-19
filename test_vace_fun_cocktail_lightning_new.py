#!/usr/bin/env python3
"""Test vace_fun_14B_cocktail_lightning model with parameter variations in new order:
1. Flow shift (3-5, every 1)
2. HIGH lightning LoRA at 1.0 strength
3. CFG guidance (1.0-1.5, every 0.25)
4. Switch threshold (600-900, every 100)
5. LoRA strength (2.0-3.0, every 0.5)
6. Reward LoRA strength (0.3-0.5, every 0.1)
"""

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


def generate_test_configurations():
    """Generate all test configurations with parameter variations."""

    # Define video configurations with specific videos, masks, and resolutions
    video_configs = [
        {
            "video": "samples/video1.mp4",
            "mask": "samples/mask1.mp4",
            "resolution": "768x576",  # Standard resolution for video1
            "video_number": 1
        },
        {
            "video": "samples/video2.mp4",
            "mask": "samples/mask2.mp4",
            "resolution": "496x896",  # Matches actual video2 size
            "video_number": 2
        }
    ]

    base_config = {
        "model": "vace_fun_14B_cocktail_lightning",
        "name": "Vace Fun Cocktail Lightning 14B",
        "test_type": "vace_video",
        "video_configs": video_configs,
        "prompt": "zooming out timelapse of a plant growing as the sun fades and the moon moves across the sky and becomes the sun, timlapsiagro",
        "base_params": {
            "video_prompt_type": "VM",
            # Resolution will be set per video config
            "video_length": 81,
            "num_inference_steps": 6,  # Lightning model uses fewer steps
            "guidance_scale": 1,       # From the lightning config
            "guidance2_scale": 1,      # From the lightning config
            "flow_shift": 5,           # From the lightning config
            "seed": 12345,
            "negative_prompt": "blurry, low quality, distorted, static, overexposed",
            "control_net_weight": 1.0,
            "control_net_weight2": 1.0,
            "sample_solver": "euler"  # Updated to use euler as default
        }
    }

    test_configs = []
    test_id = 1
    base_switch_threshold = 875  # Use the default from the model config

    # Phase 1: Test flow_shift variations (3-5, every 1)
    print("Generating Phase 1: Flow shift variations...")

    for flow_shift in range(3, 6):  # 3, 4, 5
        # Create test for each video configuration
        for video_config in base_config["video_configs"]:
            # Set prompt based on video number
            prompt = "zooming out timelapse of a plant growing as the sun fades and the moon moves across the sky and becomes the sun, timlapsiagro" if video_config["video_number"] == 1 else "he turns around and runs to the other painting"

            config = {
                **base_config,
                "test_id": test_id,
                "phase": "flow_shift",
                "video": video_config["video"],
                "mask": video_config["mask"],
                "video_number": video_config["video_number"],
                "prompt": prompt,
                "variation": f"flow_shift_{flow_shift}_video{video_config['video_number']}",
                "description": f"Flow shift: {flow_shift} (video{video_config['video_number']})",
                "params": {
                    **base_config["base_params"],
                    "resolution": video_config["resolution"],
                    "switch_threshold": base_switch_threshold,
                    "flow_shift": flow_shift,
                    # Add bloom LoRA for video1
                    **({
                        "activated_loras": ["Wan2GP/loras/bloom.safetensors"],
                        "loras_multipliers": "1.3"
                    } if video_config["video_number"] == 1 else {})
                }
            }
            test_configs.append(config)
            test_id += 1

    # Phase 2: Test with HIGH lightning LoRA at 1.0 strength
    print("Generating Phase 2: HIGH lightning LoRA test...")
    # Create test for each video configuration
    for video_config in base_config["video_configs"]:
        # Set prompt based on video number
        prompt = "zooming out timelapse of a plant growing as the sun fades and the moon moves across the sky and becomes the sun, timlapsiagro" if video_config["video_number"] == 1 else "he turns around and runs to the other painting"

        config = {
            **base_config,
            "test_id": test_id,
            "phase": "high_lightning",
            "video": video_config["video"],
            "mask": video_config["mask"],
            "video_number": video_config["video_number"],
            "prompt": prompt,
            "variation": f"high_lightning_1_0_video{video_config['video_number']}",
            "description": f"HIGH lightning LoRA at 1.0 strength (video{video_config['video_number']})",
            "params": {
                **base_config["base_params"],
                "resolution": video_config["resolution"],
                "switch_threshold": base_switch_threshold,
                # Switch to HIGH lightning LoRA + bloom for video1
                "activated_loras": [
                    "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/loras_accelerators/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
                    "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/loras_accelerators/Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_HIGH_fp16.safetensors",  # HIGH instead of LOW
                    "https://huggingface.co/alibaba-pai/Wan2.2-Fun-Reward-LoRAs/resolve/main/Wan2.2-Fun-A14B-InP-high-noise-MPS.safetensors",
                    "https://huggingface.co/alibaba-pai/Wan2.2-Fun-Reward-LoRAs/resolve/main/Wan2.2-Fun-A14B-InP-low-noise-HPS2.1.safetensors"
                ] + (["Wan2GP/loras/bloom.safetensors"] if video_config["video_number"] == 1 else []),
                "loras_multipliers": "3.0;1.0;0.5;0.5" + (";1.3" if video_config["video_number"] == 1 else "")  # HIGH lightning LoRA at 1.0 strength + bloom for video1
            }
        }
        test_configs.append(config)
        test_id += 1

    # Phase 3: Test CFG guidance variations (1.0-1.5, every 0.25)
    print("Generating Phase 3: CFG guidance variations...")

    # Generate guidance values: 1.0, 1.25, 1.5
    guidance_values = [round(1.0 + (i * 0.25), 2) for i in range(3)]  # [1.0, 1.25, 1.5]

    for guidance_scale in guidance_values:
        # Create test for each video configuration
        for video_config in base_config["video_configs"]:
            # Set prompt based on video number
            prompt = "zooming out timelapse of a plant growing as the sun fades and the moon moves across the sky and becomes the sun, timlapsiagro" if video_config["video_number"] == 1 else "he turns around and runs to the other painting"

            config = {
                **base_config,
                "test_id": test_id,
                "phase": "cfg_guidance",
                "video": video_config["video"],
                "mask": video_config["mask"],
                "video_number": video_config["video_number"],
                "prompt": prompt,
                "variation": f"cfg_{guidance_scale}_video{video_config['video_number']}",
                "description": f"CFG guidance: {guidance_scale} (video{video_config['video_number']})",
                "params": {
                    **base_config["base_params"],
                    "resolution": video_config["resolution"],
                    "switch_threshold": base_switch_threshold,
                    "guidance_scale": guidance_scale,
                    "guidance2_scale": guidance_scale,  # Also update guidance2_scale
                    # Add bloom LoRA for video1
                    **({
                        "activated_loras": ["Wan2GP/loras/bloom.safetensors"],
                        "loras_multipliers": "1.3"
                    } if video_config["video_number"] == 1 else {})
                }
            }
            test_configs.append(config)
            test_id += 1

    # Phase 4: Test switch threshold variations (600-900, every 100)
    print("Generating Phase 4: Switch threshold variations...")
    for switch_threshold in range(600, 1000, 100):  # 600, 700, 800, 900
        # Create test for each video configuration
        for video_config in base_config["video_configs"]:
            # Set prompt based on video number
            prompt = "zooming out timelapse of a plant growing as the sun fades and the moon moves across the sky and becomes the sun, timlapsiagro" if video_config["video_number"] == 1 else "he turns around and runs to the other painting"

            config = {
                **base_config,
                "test_id": test_id,
                "phase": "switch_threshold",
                "video": video_config["video"],
                "mask": video_config["mask"],
                "video_number": video_config["video_number"],
                "prompt": prompt,
                "variation": f"threshold_{switch_threshold}_video{video_config['video_number']}",
                "description": f"Switch threshold: {switch_threshold} (video{video_config['video_number']})",
                "params": {
                    **base_config["base_params"],
                    "resolution": video_config["resolution"],
                    "switch_threshold": switch_threshold,
                    # Add bloom LoRA for video1
                    **({
                        "activated_loras": ["Wan2GP/loras/bloom.safetensors"],
                        "loras_multipliers": "1.3"
                    } if video_config["video_number"] == 1 else {})
                }
            }
            test_configs.append(config)
            test_id += 1

    # Phase 5: Test first LoRA strength variations (2.0-3.0, every 0.5)
    print("Generating Phase 5: First LoRA strength variations...")

    # Generate strength values: 2.0, 2.5, 3.0
    strength_values = [round(2.0 + (i * 0.5), 1) for i in range(3)]  # [2.0, 2.5, 3.0]

    for strength in strength_values:
        # Create test for each video configuration
        for video_config in base_config["video_configs"]:
            # Set prompt based on video number
            prompt = "zooming out timelapse of a plant growing as the sun fades and the moon moves across the sky and becomes the sun, timlapsiagro" if video_config["video_number"] == 1 else "he turns around and runs to the other painting"

            config = {
                **base_config,
                "test_id": test_id,
                "phase": "lora_strength",
                "video": video_config["video"],
                "mask": video_config["mask"],
                "video_number": video_config["video_number"],
                "prompt": prompt,
                "variation": f"strength_{strength}_video{video_config['video_number']}",
                "description": f"First LoRA strength: {strength} (video{video_config['video_number']})",
                "params": {
                    **base_config["base_params"],
                    "resolution": video_config["resolution"],
                    "switch_threshold": base_switch_threshold,
                    # Override the first LoRA multiplier (lightx2v_cfg_step_distill_lora)
                    # Format: "3.0;0" becomes f"{strength};0"
                    "loras_multipliers": f"{strength};0;0.5;0;0.5" + (";1.3" if video_config["video_number"] == 1 else ""),  # Adjusted first LoRA strength + bloom for video1
                    # Add bloom LoRA for video1
                    **({
                        "activated_loras": ["Wan2GP/loras/bloom.safetensors"]
                    } if video_config["video_number"] == 1 else {})
                }
            }
            test_configs.append(config)
            test_id += 1

    # Phase 6: Test reward LoRA strength variations (0.3-0.5, every 0.1)
    print("Generating Phase 6: Reward LoRA strength variations...")

    # Generate strength values: 0.3, 0.4, 0.5
    reward_strength_values = [round(0.3 + (i * 0.1), 1) for i in range(3)]  # [0.3, 0.4, 0.5]

    for reward_strength in reward_strength_values:
        # Create test for each video configuration
        for video_config in base_config["video_configs"]:
            # Set prompt based on video number
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
                    # Format: "lightx2v;lightning;high_reward;low_reward"
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
    print(f"  - Phase 1 (Flow shift): 6 tests (3 parameters × 2 videos)")
    print(f"  - Phase 2 (HIGH lightning): 2 tests (1 parameter × 2 videos)")
    print(f"  - Phase 3 (CFG guidance): 6 tests (3 parameters × 2 videos)")
    print(f"  - Phase 4 (Switch threshold): 8 tests (4 parameters × 2 videos)")
    print(f"  - Phase 5 (LoRA strength): 6 tests (3 parameters × 2 videos)")
    print(f"  - Phase 6 (Reward LoRA strength): 6 tests (3 parameters × 2 videos)")
    print(f"  - Total: {len(test_configs)} tests using both video1.mp4 and video2.mp4")

    return test_configs


def run_vace_lightning_tests(base_output_dir: str = "tests_output"):
    """Run all vace_fun_14B_cocktail_lightning parameter tests."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_run_id = f"vace_lightning_{timestamp}"

    # Generate test configurations
    test_configs = generate_test_configurations()

    # Validate input files exist
    video1_path = Path("samples/video1.mp4").absolute()
    video2_path = Path("samples/video2.mp4").absolute()
    mask1_path = Path("samples/mask1.mp4").absolute()
    mask2_path = Path("samples/mask2.mp4").absolute()

    if not video1_path.exists():
        print(f"Error: Video file not found: {video1_path}")
        return [], test_run_id, None
    if not video2_path.exists():
        print(f"Error: Video file not found: {video2_path}")
        return [], test_run_id, None
    if not mask1_path.exists():
        print(f"Error: Mask file not found: {mask1_path}")
        return [], test_run_id, None
    if not mask2_path.exists():
        print(f"Error: Mask file not found: {mask2_path}")
        return [], test_run_id, None

    output_dir = (Path(base_output_dir).absolute() / test_run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize orchestrator
    wan_root = Path(__file__).parent / "Wan2GP"
    orchestrator = WanOrchestrator(str(wan_root))

    results = []

    print(f"\n{'='*70}")
    print(f"VACE FUN COCKTAIL LIGHTNING PARAMETER TEST")
    print(f"{'='*70}")
    print(f"Model: vace_fun_14B_cocktail_lightning")
    print(f"Total tests: {len(test_configs)}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")

    # Load model once (reuse for all tests)
    try:
        print(f"\nLoading model: vace_fun_14B_cocktail_lightning...")
        orchestrator.load_model("vace_fun_14B_cocktail_lightning")
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return [], test_run_id, output_dir

    try:
        current_phase = None

        for config in test_configs:
            # Print phase header when switching phases
            if config["phase"] != current_phase:
                current_phase = config["phase"]
                print(f"\n{'='*50}")
                print(f"PHASE: {current_phase.upper().replace('_', ' ')}")
                print(f"{'='*50}")

            test_id = config["test_id"]
            total_tests = len(test_configs)

            print(f"\n--- Test {test_id}/{total_tests}: {config['description']} ---")
            print(f"Variation: {config['variation']}")

            start_time = time.time()

            try:
                # Run the test
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
                    # Create descriptive filename based on test parameters
                    video_num = config.get('video_number', 1)

                    if config["phase"] == "flow_shift":
                        flow_shift_value = config["params"]["flow_shift"]
                        output_filename = f"video{video_num}_flow_shift_{flow_shift_value}.mp4"
                    elif config["phase"] == "high_lightning":
                        output_filename = f"video{video_num}_HIGH_lightning_1_0.mp4"
                    elif config["phase"] == "cfg_guidance":
                        guidance_value = config["params"]["guidance_scale"]
                        guidance_clean = str(guidance_value).replace(".", "_")  # Replace . with _ for filename
                        output_filename = f"video{video_num}_cfg_{guidance_clean}.mp4"
                    elif config["phase"] == "switch_threshold":
                        threshold_value = config["params"]["switch_threshold"]
                        output_filename = f"video{video_num}_threshold_{threshold_value}.mp4"
                    elif config["phase"] == "lora_strength":
                        # Extract strength from loras_multipliers (format: "strength;0;0.5;0;0.5")
                        strength_str = config["params"]["loras_multipliers"].split(";")[0]
                        strength_clean = strength_str.replace(".", "_")  # Replace . with _ for filename
                        output_filename = f"video{video_num}_lora_strength_{strength_clean}.mp4"
                    elif config["phase"] == "reward_lora_strength":
                        # Extract reward strength from loras_multipliers (format: "3.0;0;reward_strength;reward_strength")
                        reward_strength_str = config["params"]["loras_multipliers"].split(";")[2]
                        reward_strength_clean = reward_strength_str.replace(".", "_")
                        output_filename = f"video{video_num}_reward_lora_{reward_strength_clean}.mp4"
                    else:
                        # Fallback to generic naming
                        output_filename = f"video{video_num}_test_{test_id:02d}_{config['variation']}.mp4"

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
                        "model": config["model"],
                        "prompt": config["prompt"],
                        "parameters": config["params"],
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
                    "model": config["model"],
                    "prompt": config["prompt"],
                    "parameters": config["params"],
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


def run_quick_test(base_output_dir: str = "tests_output"):
    """Run one random test for each video configuration (quick test mode)."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_run_id = f"vace_lightning_quicktest_{timestamp}"

    # Generate all test configurations
    all_test_configs = generate_test_configurations()

    # Group tests by video number
    video1_tests = [config for config in all_test_configs if config["video_number"] == 1]
    video2_tests = [config for config in all_test_configs if config["video_number"] == 2]

    # Select one random test from each video group
    selected_configs = []
    if video1_tests:
        selected_configs.append(random.choice(video1_tests))
    if video2_tests:
        selected_configs.append(random.choice(video2_tests))

    print(f"\n{'='*70}")
    print(f"QUICK TEST MODE - RANDOM SAMPLE")
    print(f"{'='*70}")
    print(f"Selected {len(selected_configs)} random tests (1 per video):")
    for config in selected_configs:
        print(f"  - Video{config['video_number']}: {config['description']} ({config['phase']})")
    print(f"{'='*70}")

    # Validate input files exist
    video1_path = Path("samples/video1.mp4").absolute()
    video2_path = Path("samples/video2.mp4").absolute()
    mask1_path = Path("samples/mask1.mp4").absolute()
    mask2_path = Path("samples/mask2.mp4").absolute()

    if not video1_path.exists():
        print(f"Error: Video file not found: {video1_path}")
        return [], test_run_id, None
    if not video2_path.exists():
        print(f"Error: Video file not found: {video2_path}")
        return [], test_run_id, None
    if not mask1_path.exists():
        print(f"Error: Mask file not found: {mask1_path}")
        return [], test_run_id, None
    if not mask2_path.exists():
        print(f"Error: Mask file not found: {mask2_path}")
        return [], test_run_id, None

    output_dir = (Path(base_output_dir).absolute() / test_run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize orchestrator
    wan_root = Path(__file__).parent / "Wan2GP"
    orchestrator = WanOrchestrator(str(wan_root))

    results = []

    # Load model once
    try:
        print(f"\nLoading model: vace_fun_14B_cocktail_lightning...")
        orchestrator.load_model("vace_fun_14B_cocktail_lightning")
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return [], test_run_id, output_dir

    try:
        for i, config in enumerate(selected_configs, 1):
            print(f"\n--- Quick Test {i}/{len(selected_configs)}: {config['description']} ---")
            print(f"Phase: {config['phase']}")
            print(f"Prompt: \"{config['prompt']}\"")
            print(f"Video: {config['video']}")
            print(f"Mask: {config['mask']}")
            print(f"Resolution: {config['params']['resolution']}")

            start_time = time.time()

            try:
                # Run the test
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
                    # Create descriptive filename based on test parameters
                    video_num = config.get('video_number', 1)
                    output_filename = f"video{video_num}_{config['variation']}.mp4"
                    final_path = output_dir / output_filename
                    Path(output_path).rename(final_path)

                    print(f"✅ Completed in {generation_time:.1f}s -> {output_filename}")

                    results.append({
                        "test_id": config["test_id"],
                        "model": config["model"],
                        "model_name": config["name"],
                        "test_type": config["test_type"],
                        "phase": config["phase"],
                        "variation": config["variation"],
                        "description": config["description"],
                        "prompt": config["prompt"],
                        "video_number": config["video_number"],
                        "input_files": {
                            "video": config["video"],
                            "mask": config["mask"]
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
                        "test_id": config["test_id"],
                        "model": config["model"],
                        "phase": config["phase"],
                        "variation": config["variation"],
                        "description": config["description"],
                        "prompt": config["prompt"],
                        "video_number": config["video_number"],
                        "status": "no_output"
                    })

            except Exception as e:
                print(f"❌ Failed: {e}")
                results.append({
                    "test_id": config["test_id"],
                    "model": config["model"],
                    "phase": config["phase"],
                    "variation": config["variation"],
                    "description": config["description"],
                    "prompt": config["prompt"],
                    "video_number": config["video_number"],
                    "status": "failed",
                    "error": str(e)
                })

            # Brief pause between tests
            if i < len(selected_configs):
                time.sleep(2)

    finally:
        # Unload model
        orchestrator.unload_model()
        print(f"\n🔄 Model unloaded")

    return results, test_run_id, output_dir


def save_results(results: list, test_run_id: str, output_dir: Path):
    """Save test results to JSON with detailed analysis."""

    results_file = output_dir / "results.json"

    # Group results by phase for analysis
    phase_analysis = {}
    for result in results:
        phase = result.get("phase", "unknown")
        if phase not in phase_analysis:
            phase_analysis[phase] = []
        phase_analysis[phase].append(result)

    # Calculate statistics per phase
    phase_stats = {}
    for phase, phase_results in phase_analysis.items():
        successful = [r for r in phase_results if r.get("status") == "success"]
        failed = [r for r in phase_results if r.get("status") == "failed"]

        if successful:
            avg_time = sum(r["generation_time"] for r in successful) / len(successful)
            avg_size = sum(r["file_size_mb"] for r in successful) / len(successful)
        else:
            avg_time = 0
            avg_size = 0

        phase_stats[phase] = {
            "total_tests": len(phase_results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(phase_results) * 100 if phase_results else 0,
            "avg_generation_time": avg_time,
            "avg_file_size_mb": avg_size
        }

    # Save comprehensive results
    with open(results_file, 'w') as f:
        json.dump({
            "test_run_id": test_run_id,
            "timestamp": datetime.now().isoformat(),
            "model": "vace_fun_14B_cocktail_lightning",
            "description": "Parameter testing with reordered phases: Flow shift, HIGH lightning, CFG guidance, Switch threshold, LoRA strength, Reward LoRA strength",
            "test_phases": {
                "flow_shift": "Testing flow shift values 3-5 (every 1)",
                "high_lightning": "Testing HIGH lightning LoRA at 1.0 strength",
                "cfg_guidance": "Testing CFG guidance values 1.0-1.5 (every 0.25)",
                "switch_threshold": "Testing switch threshold values 600-900 (every 100)",
                "lora_strength": "Testing first LoRA strength 2.0-3.0 (every 0.5)",
                "reward_lora_strength": "Testing reward LoRA strength 0.3-0.5 (every 0.1)"
            },
            "summary": {
                "total_tests": len(results),
                "successful": len([r for r in results if r.get("status") == "success"]),
                "failed": len([r for r in results if r.get("status") == "failed"]),
                "no_output": len([r for r in results if r.get("status") == "no_output"])
            },
            "phase_statistics": phase_stats,
            "detailed_results": results
        }, f, indent=2)

    return results_file


def print_summary(results: list):
    """Print a detailed summary of test results by phase."""

    print(f"\n{'='*80}")
    print("VACE FUN COCKTAIL LIGHTNING TEST SUMMARY")
    print(f"{'='*80}")

    # Overall statistics
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]
    no_output = [r for r in results if r.get("status") == "no_output"]

    print(f"Overall Results:")
    print(f"  Total Tests: {len(results)}")
    print(f"  Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"  Failed: {len(failed)}")
    print(f"  No Output: {len(no_output)}")

    # Group by phase
    phases = {}
    for result in results:
        phase = result.get("phase", "unknown")
        if phase not in phases:
            phases[phase] = []
        phases[phase].append(result)

    # Phase-by-phase analysis
    for phase_name, phase_results in phases.items():
        phase_successful = [r for r in phase_results if r.get("status") == "success"]
        phase_failed = [r for r in phase_results if r.get("status") == "failed"]

        print(f"\n{phase_name.upper().replace('_', ' ')} PHASE:")
        print(f"  Tests: {len(phase_results)}")
        print(f"  Successful: {len(phase_successful)}")
        print(f"  Failed: {len(phase_failed)}")

        if phase_successful:
            print(f"\n  {'Test':<25} {'Time (s)':<10} {'Size (MB)':<10} {'Description'}")
            print("  " + "-" * 70)
            for result in phase_successful:
                time_str = f"{result['generation_time']:.1f}"
                size_str = f"{result['file_size_mb']:.1f}"
                variation = result['variation'][:23]
                description = result['description'][:30]
                print(f"  {variation:<25} {time_str:<10} {size_str:<10} {description}")

        if phase_failed:
            print(f"\n  Failed tests:")
            for result in phase_failed:
                error_msg = result.get('error', 'Unknown error')[:50]
                print(f"    ❌ {result['variation']}: {error_msg}")

    if successful:
        avg_time = sum(r["generation_time"] for r in successful) / len(successful)
        avg_size = sum(r["file_size_mb"] for r in successful) / len(successful)
        print(f"\nOverall Averages:")
        print(f"  Generation Time: {avg_time:.1f}s")
        print(f"  File Size: {avg_size:.1f}MB")


def main():
    parser = argparse.ArgumentParser(description="Test vace_fun_14B_cocktail_lightning parameter variations")
    parser.add_argument("--output-dir", default="tests_output",
                       help="Output directory for test results")
    parser.add_argument("--test", action="store_true",
                       help="Run one random test for each video (quick test mode)")

    args = parser.parse_args()

    try:
        if args.test:
            results, test_run_id, output_dir = run_quick_test(
                base_output_dir=args.output_dir
            )
        else:
            results, test_run_id, output_dir = run_vace_lightning_tests(
                base_output_dir=args.output_dir
            )

        if not output_dir:
            print("Failed: Missing input files")
            sys.exit(1)

        results_file = save_results(results, test_run_id, output_dir)

        print_summary(results)
        print(f"\nDetailed results saved: {results_file}")
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