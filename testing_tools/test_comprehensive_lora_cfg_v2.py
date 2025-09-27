#!/usr/bin/env python3
"""Comprehensive test for LoRA and CFG variations as requested:
1) Current Lightning config with 2,3,4 steps at phase 1
2) Light2x LoRA strength variations on phase 1 (0.5, 0.75, 1.0)
3) Current Lightning config with CFG 1,2,3,4,5
4) No Lightning on phase 1 with 2,3,4,5 steps
5) Light2x LoRA on phase 1 at strength 3.0 instead of Lightning
6) Light2x LoRA on phase 2 at strength 3.0 instead of Lightning
"""

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


def create_test_configurations():
    """Create all test configurations for comprehensive LoRA/CFG testing."""

    base_config = {
        "model_type": "vace_fun_14B_cocktail_lightning_3phase_light_distill_2_2_2",
        "video_length": 81,
        "guidance_phases": 3,
        "num_inference_steps": 6,
        "guidance_scale": 3,
        "guidance2_scale": 1,
        "guidance3_scale": 1,
        "flow_shift": 5,
        "switch_threshold": 667,
        "switch_threshold2": 333,
        "model_switch_phase": 2,
        "seed": 12345,
        "negative_prompt": "blurry, low quality, distorted, static, overexposed",
        "control_net_weight": 1.0,
        "control_net_weight2": 1.0,
        "sample_solver": "euler",
        "resolution": "768x576",
        "video_prompt_type": "VM"
    }

    configurations = []

    # Reward LoRA tests - 4 variants using 2-2-2 pattern from base config
    switch_threshold, switch_threshold2 = calculate_switch_thresholds(6, 2, 2)

    # Test configurations using different model variants for each video

# Skip baseline test - only run Reward LoRA variants

    # 1) Reward LoRA 1.0x on Phase 1, 0.5x on Phases 2+3
    config = {
        **base_config,
        "test_name": "reward_lora_1x_phase1_0_5x_phase2_3",
        "description": "Reward LoRA 1.0x Phase 1, 0.5x Phase 2+3 - 2-2-2 steps",
        "video1_model": "vace_fun_14B_cocktail_lightning_3phase_light_distill_2_2_2_reward_1x_0_5x_bloom",
        "video2_model": "vace_fun_14B_cocktail_lightning_3phase_light_distill_2_2_2_reward_1x_0_5x"
    }
    configurations.append(config)

    # 2) Reward LoRA 0.5x on all phases
    config = {
        **base_config,
        "test_name": "reward_lora_0_5x_all_phases",
        "description": "Reward LoRA 0.5x All Phases - 2-2-2 steps",
        "video1_model": "vace_fun_14B_cocktail_lightning_3phase_light_distill_2_2_2_reward_0_5x_all_bloom",
        "video2_model": "vace_fun_14B_cocktail_lightning_3phase_light_distill_2_2_2_reward_0_5x_all"
    }
    configurations.append(config)

    # 3) No Reward LoRA on Phase 1, 0.5x on Phases 2+3
    config = {
        **base_config,
        "test_name": "reward_lora_none_phase1_0_5x_phase2_3",
        "description": "No Reward LoRA Phase 1, 0.5x Phase 2+3 - 2-2-2 steps",
        "video1_model": "vace_fun_14B_cocktail_lightning_3phase_light_distill_2_2_2_reward_none_0_5x_bloom",
        "video2_model": "vace_fun_14B_cocktail_lightning_3phase_light_distill_2_2_2_reward_none_0_5x"
    }
    configurations.append(config)

    return configurations


def calculate_switch_thresholds(total_steps, first_phase_steps, second_phase_steps):
    """Calculate switch thresholds using actual Euler scheduler."""
    import numpy as np

    def timestep_transform(t, shift=5.0):
        return shift * t / (1 + (shift - 1) * t)

    t_schedule = np.linspace(1000, 1, total_steps)
    timesteps = [timestep_transform(t/1000) * 1000 for t in t_schedule]
    timesteps_int = [int(round(t)) for t in timesteps]

    if first_phase_steps < total_steps:
        switch_threshold = int(timesteps[first_phase_steps - 1]) + 1
    else:
        switch_threshold = 0

    phase_2_end = first_phase_steps + second_phase_steps
    if phase_2_end < total_steps:
        switch_threshold2 = int(timesteps[phase_2_end - 1]) + 1
    else:
        switch_threshold2 = 0

    return switch_threshold, switch_threshold2


def setup_lora_configuration(orchestrator, config, video_num=None):
    """Setup LoRA configuration based on test requirements."""

    test_name = config.get("test_name", "")

    if test_name == "baseline_no_reward_loras":
        # Baseline: No Reward LoRAs, only Bloom LoRA for video1
        if video_num == 1:
            lora_config = {
                "activated_loras": ["loras/bloom.safetensors"],
                "loras_multipliers": "1.3;1.3;1.3"  # Phase-based multiplier string for WGP
            }
        else:
            # No additional LoRAs for video2 - use base model only
            lora_config = {}

    elif test_name == "reward_lora_1x_phase1_0_5x_phase2_3":
        # Reward LoRA 1.0x on Phase 1, 0.5x on Phases 2+3
        base_loras = [
            "https://huggingface.co/alibaba-pai/Wan2.2-Fun-Reward-LoRAs/resolve/main/Wan2.2-Fun-A14B-InP-high-noise-MPS.safetensors",  # High noise for Phase 1
            "https://huggingface.co/alibaba-pai/Wan2.2-Fun-Reward-LoRAs/resolve/main/Wan2.2-Fun-A14B-InP-low-noise-HPS2.1.safetensors"   # Low noise for Phase 2+3
        ]
        base_multipliers = [
            "1.0;0;0",  # High noise Reward LoRA at 1.0x on Phase 1
            "0;0.5;0.5" # Low noise Reward LoRA at 0.5x on Phase 2+3
        ]

        # Add Bloom LoRA only for video1
        if video_num == 1:
            base_loras.append("loras/bloom.safetensors")
            base_multipliers.append("1.3;1.3;1.3")  # Bloom at 1.3x all phases

        lora_config = {
            "activated_loras": base_loras,
            "loras_multipliers": " ".join(base_multipliers)  # Space-separated string for WGP
        }

    elif test_name == "reward_lora_0_5x_all_phases":
        # Reward LoRA 0.5x on all phases
        base_loras = [
            "https://huggingface.co/alibaba-pai/Wan2.2-Fun-Reward-LoRAs/resolve/main/Wan2.2-Fun-A14B-InP-high-noise-MPS.safetensors",  # High noise for Phase 1
            "https://huggingface.co/alibaba-pai/Wan2.2-Fun-Reward-LoRAs/resolve/main/Wan2.2-Fun-A14B-InP-low-noise-HPS2.1.safetensors"   # Low noise for Phase 2+3
        ]
        base_multipliers = [
            "0.5;0;0",    # High noise Reward LoRA at 0.5x on Phase 1
            "0;0.5;0.5"   # Low noise Reward LoRA at 0.5x on Phase 2+3
        ]

        # Add Bloom LoRA only for video1
        if video_num == 1:
            base_loras.append("loras/bloom.safetensors")
            base_multipliers.append("1.3;1.3;1.3")  # Bloom at 1.3x all phases

        lora_config = {
            "activated_loras": base_loras,
            "loras_multipliers": " ".join(base_multipliers)  # Space-separated string for WGP
        }

    elif test_name == "reward_lora_none_phase1_0_5x_phase2_3":
        # No Reward LoRA on Phase 1, 0.5x on Phases 2+3
        base_loras = [
            "https://huggingface.co/alibaba-pai/Wan2.2-Fun-Reward-LoRAs/resolve/main/Wan2.2-Fun-A14B-InP-low-noise-HPS2.1.safetensors"   # Low noise for Phase 2+3 only
        ]
        base_multipliers = [
            "0;0.5;0.5"   # Low noise Reward LoRA at 0.5x on Phase 2+3
        ]

        # Add Bloom LoRA only for video1
        if video_num == 1:
            base_loras.append("loras/bloom.safetensors")
            base_multipliers.append("1.3;1.3;1.3")  # Bloom at 1.3x all phases

        lora_config = {
            "activated_loras": base_loras,
            "loras_multipliers": " ".join(base_multipliers)  # Space-separated string for WGP
        }
    else:
        # Default: Use base config (vace_fun_14B_cocktail_lightning_3phase_light_distill_2_2_2.json)
        lora_config = {}

    return lora_config


def run_comprehensive_test():
    """Run comprehensive LoRA and CFG test suite."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = Path(__file__).parent
    output_dir = script_dir / f"tests_output/comprehensive_lora_cfg_descriptive_names_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🧪 COMPREHENSIVE LORA & CFG TEST SUITE")
    print("=" * 70)

    # Load configurations
    configurations = create_test_configurations()

    print(f"Total test configurations: {len(configurations)}")
    print(f"Output directory: {output_dir}")
    print()

    # Initialize orchestrator
    wan_root = Path(__file__).parent / "Wan2GP"
    orchestrator = WanOrchestrator(str(wan_root))

    # Load model once
    orchestrator.load_model("vace_fun_14B_cocktail_lightning_3phase_light_distill_2_2_2")

    results = []

    for i, config in enumerate(configurations, 1):
        print(f"--- Test {i}/{len(configurations)}: {config['test_name']} ---")
        print(f"Description: {config['description']}")

        try:
            # Test on both videos
            for video_num in [1, 2]:
                # Setup LoRA configuration for this specific video
                lora_config = setup_lora_configuration(orchestrator, config, video_num)
                video_file = f"samples/video{video_num}.mp4"
                mask_file = f"samples/mask{video_num}.mp4"

                print(f"  Testing with video {video_num}...")

                # Create test parameters
                prompt = "zooming out timelapse of a plant growing as the sun fades and the moon moves across the sky and becomes the sun, timlapsiagro" if video_num == 1 else "he turns around quickly and runs into the next to to start painting as the camera pans"
                test_params = {
                    "prompt": prompt,
                    "video": video_file,
                    "mask": mask_file,
                    **config,
                    **lora_config
                }

                print(f"    🔍 DEBUG: lora_config = {lora_config}")
                print(f"    🔍 DEBUG: test_params keys = {list(test_params.keys())}")
                if "activated_loras" in test_params:
                    print(f"    🔍 DEBUG: activated_loras = {test_params['activated_loras']}")
                if "loras_multipliers" in test_params:
                    print(f"    🔍 DEBUG: loras_multipliers = {test_params['loras_multipliers']}")

                # Remove test-specific keys
                test_params.pop("test_name", None)
                test_params.pop("description", None)
                test_params.pop("custom_lora_config", None)

                start_time = time.time()

                # Run generation with proper parameter structure
                result_path = orchestrator.generate_vace(
                    prompt=test_params['prompt'],
                    video_guide=test_params['video'],
                    video_mask=test_params['mask'],
                    **{k: v for k, v in test_params.items() if k not in ['prompt', 'video', 'mask']}
                )

                generation_time = time.time() - start_time

                # Debug logging
                print(f"    🔍 DEBUG: result_path from orchestrator: {result_path}")

                # Construct absolute paths first
                wan_root = Path(__file__).parent / "Wan2GP"
                print(f"    🔍 DEBUG: wan_root: {wan_root}")

                if result_path:
                    abs_result_path = (wan_root / result_path).resolve()
                    print(f"    🔍 DEBUG: abs_result_path: {abs_result_path}")
                    print(f"    🔍 DEBUG: abs_result_path.exists(): {abs_result_path.exists()}")
                else:
                    abs_result_path = None
                    print(f"    🔍 DEBUG: result_path is None or empty")

                # Get file size
                if abs_result_path and abs_result_path.exists():
                    file_size = abs_result_path.stat().st_size / (1024 * 1024)  # MB

                    # Move to output directory with descriptive filename
                    descriptive_name = f"{config['test_name']}_video{video_num}_cfg{config['guidance_scale']}_steps{config['num_inference_steps']}.mp4"
                    new_path = output_dir / descriptive_name
                    abs_new_path = new_path.resolve()

                    print(f"    🔍 DEBUG: Moving file...")
                    print(f"    🔍 DEBUG: Source: {abs_result_path}")
                    print(f"    🔍 DEBUG: Destination: {abs_new_path}")
                    print(f"    🔍 DEBUG: Destination dir exists: {abs_new_path.parent.exists()}")

                    # Use shutil.move for more robust file moving
                    import shutil
                    try:
                        shutil.move(str(abs_result_path), str(abs_new_path))
                        print(f"    ✅ Video {video_num} completed in {generation_time:.1f}s -> {new_path.name} ({file_size:.1f}MB)")
                    except Exception as move_error:
                        print(f"    ❌ File move failed: {move_error}")
                        raise

                    results.append({
                        "test_name": f"{config['test_name']}_video{video_num}",
                        "descriptive_filename": descriptive_name,
                        "description": f"{config['description']} (Video {video_num})",
                        "video_number": video_num,
                        "generation_time": generation_time,
                        "file_size_mb": file_size,
                        "output_path": str(new_path),
                        "status": "success",
                        **config
                    })
                else:
                    print(f"    ❌ Video {video_num} failed - no output generated")
                    results.append({
                        "test_name": f"{config['test_name']}_video{video_num}",
                        "description": f"{config['description']} (Video {video_num})",
                        "video_number": video_num,
                        "status": "failed",
                        **config
                    })

        except Exception as e:
            print(f"❌ Failed with error: {e}")
            results.append({
                "test_name": config['test_name'],
                "description": config['description'],
                "status": "error",
                "error": str(e),
                **config
            })

        print()

    # Unload model
    orchestrator.unload_model()

    # Save results
    results_file = output_dir / "comprehensive_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "total_tests": len(configurations),
            "results": results,
            "summary": {
                "successful": len([r for r in results if r.get("status") == "success"]),
                "failed": len([r for r in results if r.get("status") in ["failed", "error"]])
            }
        }, f, indent=2)

    print("🏁 COMPREHENSIVE TEST COMPLETE")
    print(f"Results saved: {results_file}")
    print(f"Videos saved: {output_dir}")


if __name__ == "__main__":
    run_comprehensive_test()