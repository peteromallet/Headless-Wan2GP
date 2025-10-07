#!/usr/bin/env python3
"""
Standalone validator for phase_config parser

Tests the parse_phase_config function without requiring full worker imports
"""

import json
import numpy as np


def timestep_transform(t: float, shift: float = 5.0, num_timesteps: int = 1000) -> float:
    """Transform timestep using flow_shift"""
    t = t / num_timesteps
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t


def parse_phase_config_test(phase_config: dict, num_inference_steps: int) -> dict:
    """Test version of parse_phase_config"""

    # Validation
    num_phases = phase_config.get("num_phases", 3)
    steps_per_phase = phase_config.get("steps_per_phase", [2, 2, 2])
    flow_shift = phase_config.get("flow_shift", 5.0)

    if num_phases not in [2, 3]:
        raise ValueError(f"num_phases must be 2 or 3, got {num_phases}")

    if len(steps_per_phase) != num_phases:
        raise ValueError(f"steps_per_phase length must match num_phases")

    total_steps = sum(steps_per_phase)
    if total_steps != num_inference_steps:
        raise ValueError(f"steps_per_phase sum ({total_steps}) != num_inference_steps ({num_inference_steps})")

    phases_config = phase_config.get("phases", [])
    if len(phases_config) != num_phases:
        raise ValueError(f"Expected {num_phases} phase configs, got {len(phases_config)}")

    # Generate timesteps
    sample_solver = phase_config.get("sample_solver", "euler")
    use_transform = (sample_solver == "euler")

    timesteps = list(np.linspace(1000, 1, num_inference_steps, dtype=np.float32))
    timesteps.append(0.)

    if use_transform:
        timesteps = [timestep_transform(t, shift=flow_shift, num_timesteps=1000) for t in timesteps][:-1]
    else:
        timesteps = timesteps[:-1]

    print(f"✓ Generated timesteps: {[f'{t:.1f}' for t in timesteps]}")

    # Calculate switch thresholds
    switch_step_1 = steps_per_phase[0]
    switch_threshold = None
    switch_threshold2 = None

    if switch_step_1 < num_inference_steps:
        switch_threshold = float(timesteps[switch_step_1] + 1)
        print(f"✓ Calculated switch_threshold (phase 1→2): {switch_threshold:.1f} at step {switch_step_1}")

    if num_phases >= 3:
        switch_step_2 = steps_per_phase[0] + steps_per_phase[1]
        if switch_step_2 < num_inference_steps:
            switch_threshold2 = float(timesteps[switch_step_2] + 1)
            print(f"✓ Calculated switch_threshold2 (phase 2→3): {switch_threshold2:.1f} at step {switch_step_2}")

    # Build result
    result = {
        "guidance_phases": num_phases,
        "switch_threshold": switch_threshold,
        "switch_threshold2": switch_threshold2,
        "flow_shift": flow_shift,
        "sample_solver": sample_solver,
        "model_switch_phase": phase_config.get("model_switch_phase", 2),
    }

    # Extract guidance scales
    if num_phases >= 1:
        result["guidance_scale"] = phases_config[0].get("guidance_scale", 3.0)
    if num_phases >= 2:
        result["guidance2_scale"] = phases_config[1].get("guidance_scale", 1.0)
    if num_phases >= 3:
        result["guidance3_scale"] = phases_config[2].get("guidance_scale", 1.0)

    # Process LoRAs
    all_lora_urls = set()
    for phase_cfg in phases_config:
        for lora in phase_cfg.get("loras", []):
            all_lora_urls.add(lora["url"])

    all_lora_urls = sorted(all_lora_urls)
    lora_multipliers = []
    additional_loras = {}

    for lora_url in all_lora_urls:
        phase_mults = []

        for phase_idx, phase_cfg in enumerate(phases_config):
            lora_in_phase = None
            for lora in phase_cfg.get("loras", []):
                if lora["url"] == lora_url:
                    lora_in_phase = lora
                    break

            if lora_in_phase:
                multiplier_str = str(lora_in_phase["multiplier"])

                # Validate
                if "," in multiplier_str:
                    values = multiplier_str.split(",")
                    expected_count = steps_per_phase[phase_idx]
                    if len(values) != expected_count:
                        raise ValueError(f"Phase {phase_idx+1} multiplier has {len(values)} values, needs {expected_count}")
                    for val in values:
                        num = float(val)
                        if num < 0 or num > 2.0:
                            raise ValueError(f"Multiplier {val} out of range [0.0-2.0]")
                    phase_mults.append(multiplier_str)
                else:
                    num = float(multiplier_str)
                    if num < 0 or num > 2.0:
                        raise ValueError(f"Multiplier {multiplier_str} out of range [0.0-2.0]")
                    phase_mults.append(multiplier_str)
            else:
                phase_mults.append("0")

        # Combine
        if num_phases == 2:
            multiplier_string = f"{phase_mults[0]};{phase_mults[1]}"
        else:
            multiplier_string = f"{phase_mults[0]};{phase_mults[1]};{phase_mults[2]}"

        lora_multipliers.append(multiplier_string)

        try:
            first_val = float(phase_mults[0].split(",")[0])
            additional_loras[lora_url] = first_val
        except:
            additional_loras[lora_url] = 1.0

    result["lora_names"] = all_lora_urls
    result["lora_multipliers"] = lora_multipliers
    result["additional_loras"] = additional_loras

    print(f"✓ Processed {len(all_lora_urls)} LoRAs")

    return result


# Test cases
test_cases = [
    {
        "name": "Lightning 2-2-2",
        "phase_config": {
            "num_phases": 3,
            "steps_per_phase": [2, 2, 2],
            "flow_shift": 5.0,
            "sample_solver": "euler",
            "model_switch_phase": 2,
            "phases": [
                {
                    "phase": 1,
                    "guidance_scale": 3.0,
                    "loras": [
                        {"url": "https://example.com/high_noise.safetensors", "multiplier": "0"}
                    ]
                },
                {
                    "phase": 2,
                    "guidance_scale": 1.0,
                    "loras": [
                        {"url": "https://example.com/high_noise.safetensors", "multiplier": "1.0"}
                    ]
                },
                {
                    "phase": 3,
                    "guidance_scale": 1.0,
                    "loras": [
                        {"url": "https://example.com/low_noise.safetensors", "multiplier": "1.0"}
                    ]
                }
            ]
        },
        "num_steps": 6
    },
    {
        "name": "Fast 1-2-1 with ramps",
        "phase_config": {
            "num_phases": 3,
            "steps_per_phase": [1, 2, 1],
            "flow_shift": 5.0,
            "sample_solver": "euler",
            "phases": [
                {"phase": 1, "guidance_scale": 4.0, "loras": [{"url": "https://example.com/lora.safetensors", "multiplier": "0.8"}]},
                {"phase": 2, "guidance_scale": 1.5, "loras": [{"url": "https://example.com/lora.safetensors", "multiplier": "0.9,1.0"}]},
                {"phase": 3, "guidance_scale": 1.0, "loras": [{"url": "https://example.com/lora.safetensors", "multiplier": "1.0"}]}
            ]
        },
        "num_steps": 4
    }
]


def main():
    print("="*80)
    print("PHASE_CONFIG PARSER VALIDATION")
    print("="*80)

    for test in test_cases:
        print(f"\n{'='*80}")
        print(f"Test: {test['name']}")
        print(f"{'='*80}")

        try:
            result = parse_phase_config_test(test["phase_config"], test["num_steps"])

            print("\n✓ VALIDATION PASSED")
            print("\nComputed parameters:")
            print(f"  guidance_phases: {result['guidance_phases']}")
            print(f"  switch_threshold: {result['switch_threshold']}")
            print(f"  switch_threshold2: {result['switch_threshold2']}")
            print(f"  guidance_scale: {result['guidance_scale']}")
            print(f"  guidance2_scale: {result.get('guidance2_scale')}")
            print(f"  guidance3_scale: {result.get('guidance3_scale')}")
            print(f"  flow_shift: {result['flow_shift']}")
            print(f"  sample_solver: {result['sample_solver']}")
            print(f"  lora_multipliers: {result['lora_multipliers']}")
            print(f"\n✓ Test '{test['name']}' PASSED")

        except Exception as e:
            print(f"\n✗ Test '{test['name']}' FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("ALL TESTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
