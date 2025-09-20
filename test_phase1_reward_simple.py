#!/usr/bin/env python3
"""Test reward LoRA strength variations on Phase 1 only - Simplified Version"""

import sys
import json
from pathlib import Path
from headless_wgp import WanOrchestrator

def create_reward_configs():
    """Create reward LoRA configs for Phase 1 testing"""

    reward_strengths = [1.0, 0.75, 0.5, 0.25]
    configs_created = []

    defaults_dir = Path("Wan2GP/defaults")
    defaults_dir.mkdir(parents=True, exist_ok=True)

    for reward_strength in reward_strengths:
        config = {
            "model": {
                "name": f"Wan2.2 Vace Fun Lightning 3Phase - Reward {reward_strength} (P1)",
                "architecture": "vace_14B",
                "description": f"3-phase model with reward LoRA {reward_strength} on Phase 1 only",
                "URLs": "vace_fun_14B_2_2",
                "URLs2": "vace_fun_14B_2_2",
                "loras": [
                    "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/loras_accelerators/Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_HIGH_fp16.safetensors",
                    "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/loras_accelerators/Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_LOW_fp16.safetensors",
                    "https://huggingface.co/DeepBeepMeep/Wan2.2/resolve/main/loras_add_ons/bloom.safetensors",
                    "https://huggingface.co/alibaba-pai/Wan2.2-Fun-Reward-LoRAs/resolve/main/Wan2.2-Fun-A14B-InP-high-noise-MPS.safetensors"
                ],
                "loras_multipliers": [
                    "0;1;0",           # Lightning HIGH: Phase 2 only
                    "0;0;1",           # Lightning LOW: Phase 3 only
                    "1.3;1.3;1.3",     # Bloom: All phases
                    f"{reward_strength};0;0"  # Reward: Phase 1 ONLY
                ],
                "lock_guidance_phases": True,
                "group": "wan2_2"
            },
            "guidance_phases": 3,
            "num_inference_steps": 6,
            "guidance_scale": 3,
            "guidance2_scale": 1,
            "guidance3_scale": 1,
            "flow_shift": 5,
            "switch_threshold": 800,
            "switch_threshold2": 600,
            "model_switch_phase": 3,
            "sample_solver": "euler"
        }

        config_name = f"test_reward_p1_{str(reward_strength).replace('.', '_')}"
        config_path = defaults_dir / f"{config_name}.json"

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        configs_created.append((config_name, config_path, reward_strength))
        print(f"✅ Created config: {config_name} (reward {reward_strength} on Phase 1)")

    return configs_created

def test_phase1_reward_lora():
    """Test reward LoRA at different strengths on Phase 1 only"""

    print("🧪 Phase 1 Reward LoRA Strength Testing")
    print("Creating configurations for reward LoRA testing...")

    # Create configs
    configs = create_reward_configs()

    orchestrator = WanOrchestrator("Wan2GP")
    results = []

    print("\n🚀 Phase 1 Reward LoRA Strength Tests")
    print("=" * 60)
    print(f"Testing {len(configs)} different reward LoRA strengths on Phase 1 ONLY:")
    for i, (_, _, strength) in enumerate(configs, 1):
        print(f"  Test {i}: Reward LoRA strength {strength} (Phase 1 only)")
    print("  • Bloom LoRA: 1.3 (all phases)")
    print("  • Lightning HIGH: Phase 2 only")
    print("  • Lightning LOW: Phase 3 only")
    print("=" * 60)

    try:
        for i, (config_name, config_path, reward_strength) in enumerate(configs, 1):
            print(f"\n--- Test {i}/{len(configs)}: Reward LoRA strength {reward_strength} (Phase 1 ONLY) ---")

            try:
                # Load model with the specific config
                print(f"🔄 Loading model with reward LoRA {reward_strength} on Phase 1...")
                orchestrator.load_model(config_name)
                print("✅ Model loaded successfully")

                # Base parameters
                params = {
                    "video_prompt_type": "VM",
                    "resolution": "768x576",
                    "video_length": 81,
                    "seed": 12345,
                    "negative_prompt": "blurry, low quality, distorted, static, overexposed",
                    "activated_loras": ["Wan2GP/loras/bloom.safetensors"],
                    "loras_multipliers": "1.3"
                }

                prompt = "zooming out timelapse of a plant growing as the sun fades and the moon moves across the sky and becomes the sun, timlapsiagro"

                print(f"📋 Configuration:")
                print(f"   • Phases: 3 (CFG 3→1→1, steps 0-2→2-4→4-6)")
                print(f"   • Bloom LoRA: 1.3 (all phases)")
                print(f"   • Reward LoRA: {reward_strength} (Phase 1 ONLY)")
                print(f"   • Lightning: Phase 2 HIGH, Phase 3 LOW")

                output_path = orchestrator.generate_vace(
                    prompt=prompt,
                    video_guide="samples/video1.mp4",
                    video_mask="samples/mask1.mp4",
                    **params
                )

                if output_path:
                    # Create descriptive filename
                    output_file = Path(output_path)
                    new_filename = f"phase1_reward_{str(reward_strength).replace('.', '_')}.mp4"
                    new_path = output_file.parent / new_filename
                    output_file.rename(new_path)

                    file_size = new_path.stat().st_size / (1024 * 1024)
                    print(f"✅ SUCCESS: {new_filename} ({file_size:.1f}MB)")

                    results.append({
                        "reward_strength": reward_strength,
                        "output_path": str(new_path),
                        "file_size_mb": file_size,
                        "status": "success"
                    })
                else:
                    print("❌ FAILED: No output generated")
                    results.append({
                        "reward_strength": reward_strength,
                        "status": "failed"
                    })

                # Unload model after each test
                orchestrator.unload_model()

            except Exception as e:
                print(f"❌ ERROR: {e}")
                results.append({
                    "reward_strength": reward_strength,
                    "status": "error",
                    "error": str(e)
                })

            # Brief pause between tests
            if i < len(configs):
                print("⏳ Pausing 3 seconds...")
                import time
                time.sleep(3)

    finally:
        # Clean up temporary config files
        print("\n🧹 Cleaning up temporary configs...")
        for _, config_path, strength in configs:
            try:
                config_path.unlink()
                print(f"  ✅ Removed: {config_path.name}")
            except Exception as e:
                print(f"  ⚠️ Could not remove {config_path.name}: {e}")

    return results

def main():
    results = test_phase1_reward_lora()

    print("\n" + "=" * 60)
    print("🏁 FINAL RESULTS")
    print("=" * 60)

    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]

    print(f"Total Tests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print("\n✅ Successful Tests:")
        for result in successful:
            strength = result["reward_strength"]
            filename = Path(result["output_path"]).name
            size = result["file_size_mb"]
            print(f"  • Phase 1 Reward {strength}: {filename} ({size:.1f}MB)")

    if failed:
        print("\n❌ Failed Tests:")
        for result in failed:
            strength = result["reward_strength"]
            print(f"  • Phase 1 Reward {strength}: {result['status']}")

    print("\n🎯 Phase 1 Reward LoRA testing completed!")

if __name__ == "__main__":
    main()