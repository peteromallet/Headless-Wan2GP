#!/usr/bin/env python3
import json
import os

# Base configuration template
base_config = {
    "video_length": 81,
    "guidance_phases": 3,
    "num_inference_steps": 6,
    "guidance_scale": 3,
    "guidance2_scale": 1,
    "guidance3_scale": 1,
    "flow_shift": 5,
    "switch_threshold": 601,
    "switch_threshold2": 201,
    "model_switch_phase": 2,
    "seed": 12345,
    "sample_solver": "euler",
    "video_prompt_type": "VM",
    "negative_prompt": "fading, breaking, shot cuts, jumpcuts, blurry, noise, distorted"
}

# LoRA configurations
loras_configs = {
    "banostasis": {
        "url": "https://huggingface.co/Cseti/wan2.2-14B-Kinestasis_concept-lora-v1/resolve/main/246839-wan22_14B-high-banostasis_concept-e459.safetensors",
        "name": "Banostasis Concept"
    },
    "fractal": {
        "url": "https://huggingface.co/Cseti/wan2.2-14B-Kinestasis_concept-lora-v1/resolve/main/246838-wan22_14B-high-fractal_concept-e459.safetensors",
        "name": "Fractal Concept"
    },
    "hps21": {
        "url": "https://huggingface.co/alibaba-pai/Wan2.2-Fun-Reward-LoRAs/resolve/main/Wan2.2-Fun-A14B-InP-high-noise-HPS2.1.safetensors",
        "name": "HPS2.1 Reward LoRA"
    },
    "mps": {
        "url": "https://huggingface.co/alibaba-pai/Wan2.2-Fun-Reward-LoRAs/resolve/main/Wan2.2-Fun-A14B-InP-high-noise-MPS.safetensors",
        "name": "MPS Reward LoRA"
    }
}

# Lightning LoRAs (constant across all experiments)
lightning_loras = [
    "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/loras_accelerators/Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_HIGH_fp16.safetensors",
    "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/loras_accelerators/Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_LOW_fp16.safetensors"
]

def create_experiment(lora_type, phase, strength):
    """Create a single experiment configuration"""
    lora_config = loras_configs[lora_type]

    # Create experiment name
    phase_name = "first" if phase == 1 else "second"
    strength_str = "025" if strength == 0.25 else "05"
    exp_name = f"lightning_high_{lora_type}_{phase_name}_2_2_2_{strength_str}"

    # Create directory
    exp_dir = f"testing/banostasis/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)

    # Create LoRA multipliers based on phase
    if phase == 1:
        # Phase 1: [primary_lora, lightning_high, lightning_low]
        lora_multipliers = [f"{strength};0;0", "0;1.0;0", "0;0;1"]
    else:
        # Phase 2: [primary_lora, lightning_high, lightning_low]
        lora_multipliers = [f"0;{strength};0", "0;1.0;0", "0;0;1"]

    # Create model configuration
    model_config = {
        "name": f"Wan2.2 Vace Fun Cocktail Lightning 14B (3-Phase) 2-2-2 Steps - {lora_config['name']} {phase_name.title()} Phase {strength}x",
        "architecture": "vace_14B",
        "description": f"3-phase model with {lora_config['name']} at {strength}x in Phase {phase}, Lightning HIGH Phase 2 + Lightning LOW Phase 3 - 2-2-2 steps, standard CFG, model switch at phase 2",
        "URLs": "vace_fun_14B_2_2",
        "URLs2": "vace_fun_14B_2_2",
        "loras": [lora_config["url"]] + lightning_loras,
        "loras_multipliers": lora_multipliers,
        "lock_guidance_phases": True,
        "group": "wan2_2"
    }

    # Combine configurations
    full_config = base_config.copy()
    full_config["model"] = model_config

    # Write settings file
    settings_file = f"{exp_dir}/settings.json"
    with open(settings_file, 'w') as f:
        json.dump(full_config, f, indent=2)

    print(f"Created: {exp_name}")

# Create all 16 experiments
for lora_type in ["banostasis", "fractal", "hps21", "mps"]:
    for phase in [1, 2]:
        for strength in [0.25, 0.5]:
            create_experiment(lora_type, phase, strength)

print("All 16 experiments created!")