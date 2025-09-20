#!/usr/bin/env python3
import sys
from pathlib import Path
from headless_wgp import WanOrchestrator

def test_lightning_3phase_only():
    """Test ONLY vace_fun_14B_cocktail_lightning_3phase"""

    # Use the exact parameters from the config file
    params = {
        "video_prompt_type": "VM",
        "resolution": "768x576",
        "video_length": 81,
        "num_inference_steps": 6,        # From config
        "guidance_phases": 3,            # From config: 3 phases
        "guidance_scale": 3,             # From config: Phase 1 CFG=3
        "guidance2_scale": 1,            # From config: Phase 2 CFG=1
        "guidance3_scale": 1,            # From config: Phase 3 CFG=1
        "switch_threshold": 900,         # From config: 900
        "switch_threshold2": 700,        # From config: 700
        "model_switch_phase": 3,         # From config: switch at phase 3
        "flow_shift": 5,                 # From config: 5
        "seed": 12345,
        "negative_prompt": "blurry, low quality, distorted, static, overexposed",
        "sample_solver": "euler",        # From config
        "activated_loras": ["Wan2GP/loras/bloom.safetensors"],
        "loras_multipliers": "1.3"
    }

    prompt = "zooming out timelapse of a plant growing as the sun fades and the moon moves across the sky and becomes the sun, timlapsiagro"

    orchestrator = WanOrchestrator("Wan2GP")

    try:
        print("=" * 80)
        print("🔥 TESTING vace_fun_14B_cocktail_lightning_3phase ONLY")
        print("=" * 80)
        print("📋 Configuration from vace_fun_14B_cocktail_lightning_3phase.json:")
        print(f"   • Inference Steps: {params['num_inference_steps']}")
        print(f"   • Guidance Phases: {params['guidance_phases']}")
        print(f"   • CFG Scales: {params['guidance_scale']} → {params['guidance2_scale']} → {params['guidance3_scale']}")
        print(f"   • Switch Thresholds: {params['switch_threshold']} → {params['switch_threshold2']}")
        print(f"   • Flow Shift: {params['flow_shift']}")
        print(f"   • Sample Solver: {params['sample_solver']}")
        print("=" * 80)

        print("\n🔄 Loading vace_fun_14B_cocktail_lightning_3phase...")
        orchestrator.load_model("vace_fun_14B_cocktail_lightning_3phase")
        print("✅ Model loaded successfully")

        print("\n🎬 Generating with video1 and mask1...")
        print(f"📝 Prompt: {prompt}")

        output_path = orchestrator.generate_vace(
            prompt=prompt,
            video_guide="samples/video1.mp4",
            video_mask="samples/mask1.mp4",
            **params
        )

        if output_path:
            print(f"\n🎯 ✅ SUCCESS: {output_path}")

            # Get file size
            file_size = Path(output_path).stat().st_size / (1024 * 1024)
            print(f"📊 File Size: {file_size:.1f}MB")

            return output_path
        else:
            print("\n❌ FAILED: No output generated")
            return None

    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        orchestrator.unload_model()
        print("\n🔄 Model unloaded")

def main():
    print("🚀 Single Model Test: vace_fun_14B_cocktail_lightning_3phase")

    result = test_lightning_3phase_only()

    print("\n" + "=" * 80)
    print("🏁 FINAL RESULT")
    print("=" * 80)

    if result:
        print(f"✅ SUCCESS: Generated video at {result}")
    else:
        print("❌ FAILED: No video generated")

    print("\n🎯 Test completed!")

if __name__ == "__main__":
    main()