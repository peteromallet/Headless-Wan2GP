#!/usr/bin/env python3
"""
Test script for phase_config override feature

This demonstrates how to use the comprehensive phase_config parameter
to control all phase-related settings in a single block.
"""

import json
import sys
from pathlib import Path

# Example 1: Balanced 2-2-2 configuration (Lightning baseline)
example_lightning_baseline = {
    "task_type": "vace",
    "prompt": "a cat walking through a beautiful garden with flowers",
    "model": "vace_14B_cocktail_2_2",
    "num_inference_steps": 6,
    "video_length": 81,
    "resolution": "1280x720",
    "seed": 12345,

    # PHASE_CONFIG OVERRIDE - Controls everything phase-related
    "phase_config": {
        "num_phases": 3,
        "steps_per_phase": [2, 2, 2],  # Must sum to num_inference_steps (6)
        "flow_shift": 5.0,
        "sample_solver": "euler",
        "model_switch_phase": 2,
        "phases": [
            {
                "phase": 1,
                "guidance_scale": 3.0,
                "loras": [
                    {
                        "url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors",
                        "multiplier": "0"  # No Lightning in phase 1
                    },
                    {
                        "url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/low_noise_model.safetensors",
                        "multiplier": "0"  # No low noise in phase 1
                    }
                ]
            },
            {
                "phase": 2,
                "guidance_scale": 1.0,
                "loras": [
                    {
                        "url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors",
                        "multiplier": "1.0"  # Full strength in phase 2
                    },
                    {
                        "url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/low_noise_model.safetensors",
                        "multiplier": "0"
                    }
                ]
            },
            {
                "phase": 3,
                "guidance_scale": 1.0,
                "loras": [
                    {
                        "url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors",
                        "multiplier": "0"
                    },
                    {
                        "url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/low_noise_model.safetensors",
                        "multiplier": "1.0"  # Low noise LoRA in phase 3
                    }
                ]
            }
        ]
    }
}

# Example 2: Fast 1-2-1 with ramping multipliers
example_fast_with_ramps = {
    "task_type": "vace",
    "prompt": "a dog running on a beach at sunset",
    "model": "vace_14B_cocktail_2_2",
    "num_inference_steps": 4,
    "video_length": 81,
    "resolution": "1280x720",

    "phase_config": {
        "num_phases": 3,
        "steps_per_phase": [1, 2, 1],  # Fast: 1 step phase1, 2 steps phase2, 1 step phase3
        "flow_shift": 5.0,
        "sample_solver": "euler",
        "model_switch_phase": 2,
        "phases": [
            {
                "phase": 1,
                "guidance_scale": 4.0,  # Higher guidance in phase 1
                "loras": [
                    {
                        "url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors",
                        "multiplier": "0.8"  # Single value for 1 step
                    }
                ]
            },
            {
                "phase": 2,
                "guidance_scale": 1.5,
                "loras": [
                    {
                        "url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors",
                        "multiplier": "0.9,1.0"  # RAMP: 0.9 → 1.0 over 2 steps
                    }
                ]
            },
            {
                "phase": 3,
                "guidance_scale": 1.0,
                "loras": [
                    {
                        "url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/low_noise_model.safetensors",
                        "multiplier": "1.0"  # Single value for 1 step
                    }
                ]
            }
        ]
    }
}

# Example 3: Quality-focused 3-3-4 with multiple LoRAs
example_quality_multi_lora = {
    "task_type": "vace",
    "prompt": "cinematic shot of a futuristic city at night",
    "model": "vace_14B_cocktail_2_2",
    "num_inference_steps": 10,
    "video_length": 81,
    "resolution": "1280x720",

    "phase_config": {
        "num_phases": 3,
        "steps_per_phase": [3, 3, 4],  # More steps for quality
        "flow_shift": 5.0,
        "sample_solver": "euler",
        "model_switch_phase": 3,  # Switch model in phase 3
        "phases": [
            {
                "phase": 1,
                "guidance_scale": 3.5,
                "loras": [
                    {
                        "url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors",
                        "multiplier": "0.5,0.7,0.9"  # Ramp up over 3 steps
                    },
                    {
                        "url": "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/loras_accelerators/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors",
                        "multiplier": "1.0"  # CausVid constant in phase 1
                    }
                ]
            },
            {
                "phase": 2,
                "guidance_scale": 2.0,
                "loras": [
                    {
                        "url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors",
                        "multiplier": "1.0"  # Full in phase 2
                    },
                    {
                        "url": "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/loras_accelerators/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors",
                        "multiplier": "0.8,0.6,0.4"  # Ramp down
                    }
                ]
            },
            {
                "phase": 3,
                "guidance_scale": 1.0,
                "loras": [
                    {
                        "url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/low_noise_model.safetensors",
                        "multiplier": "1.0"
                    },
                    {
                        "url": "https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/loras_accelerators/DetailEnhancerV1.safetensors",
                        "multiplier": "0.2"  # Detail enhancement in final phase
                    }
                ]
            }
        ]
    }
}

# Example 4: 2-phase configuration
example_2_phase = {
    "task_type": "vace",
    "prompt": "abstract art flowing liquid paint",
    "model": "vace_14B_cocktail_2_2",
    "num_inference_steps": 6,
    "video_length": 81,
    "resolution": "1280x720",

    "phase_config": {
        "num_phases": 2,
        "steps_per_phase": [3, 3],
        "flow_shift": 3.0,  # Lower flow_shift for different motion characteristic
        "sample_solver": "euler",
        "model_switch_phase": 2,
        "phases": [
            {
                "phase": 1,
                "guidance_scale": 5.0,  # High guidance for abstract
                "loras": [
                    {
                        "url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/high_noise_model.safetensors",
                        "multiplier": "0.7"
                    }
                ]
            },
            {
                "phase": 2,
                "guidance_scale": 1.5,
                "loras": [
                    {
                        "url": "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-250928/low_noise_model.safetensors",
                        "multiplier": "1.0"
                    }
                ]
            }
        ]
    }
}


def print_example(name: str, config: dict):
    """Print an example configuration"""
    print(f"\n{'='*80}")
    print(f"Example: {name}")
    print(f"{'='*80}")
    print(json.dumps(config, indent=2))
    print()

    # Show what parameters will be computed
    if "phase_config" in config:
        pc = config["phase_config"]
        print(f"Will compute:")
        print(f"  - {pc['num_phases']} phases with distribution: {pc['steps_per_phase']}")
        print(f"  - flow_shift: {pc['flow_shift']}")
        print(f"  - Guidance scales: {[p.get('guidance_scale') for p in pc['phases']]}")
        print(f"  - Total unique LoRAs: {len(set(lora['url'] for phase in pc['phases'] for lora in phase.get('loras', [])))}")


def main():
    print("="*80)
    print("PHASE_CONFIG TEST EXAMPLES")
    print("="*80)
    print()
    print("These examples show how to use the phase_config parameter to override")
    print("all phase-related settings in a single comprehensive block.")
    print()

    print_example("1. Lightning Baseline (2-2-2)", example_lightning_baseline)
    print_example("2. Fast with Ramps (1-2-1)", example_fast_with_ramps)
    print_example("3. Quality Multi-LoRA (3-3-4)", example_quality_multi_lora)
    print_example("4. Two-Phase (3-3)", example_2_phase)

    print("\n" + "="*80)
    print("HOW TO USE")
    print("="*80)
    print()
    print("1. Via Python API:")
    print("   queue.add_task(**example_lightning_baseline)")
    print()
    print("2. Via REST API:")
    print("   POST /api/tasks")
    print("   Body: <JSON from above>")
    print()
    print("3. Via Database:")
    print("   INSERT INTO tasks (task_type, prompt, model, task_params, status)")
    print("   VALUES ('vace', 'prompt', 'model', '<phase_config JSON>', 'queued')")
    print()
    print("IMPORTANT:")
    print("  - steps_per_phase MUST sum to num_inference_steps")
    print("  - Ramp multipliers MUST have correct number of values per phase")
    print("  - When phase_config is present, it overrides ALL phase settings")
    print("  - When phase_config is absent, existing behavior is preserved")
    print()


if __name__ == "__main__":
    main()
