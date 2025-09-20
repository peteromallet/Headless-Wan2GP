#!/usr/bin/env python3
import sys
from pathlib import Path
from headless_wgp import WanOrchestrator

def test_vace_2_2():
    """Test vace_fun_14B_cocktail_2_2 model first"""
    params = {
        "video_prompt_type": "VM",
        "resolution": "768x576",
        "video_length": 81,
        "num_inference_steps": 10,       # From config: 10 steps
        "guidance_phases": 2,            # From config: 2 phases
        "guidance_scale": 1,             # From config: CFG=1
        "guidance2_scale": 1,            # From config: CFG=1
        "switch_threshold": 875,         # From config: 875
        "flow_shift": 2,                 # From config: 2
        "seed": 12345,
        "negative_prompt": "blurry, low quality, distorted, static, overexposed",
        "sample_solver": "euler",
        "activated_loras": ["Wan2GP/loras/bloom.safetensors"],
        "loras_multipliers": "1.3"
    }

    prompt = "zooming out timelapse of a plant growing as the sun fades and the moon moves across the sky and becomes the sun, timlapsiagro"

    orchestrator = WanOrchestrator("Wan2GP")

    try:
        print("=" * 70)
        print("TESTING vace_fun_14B_cocktail_2_2 FIRST")
        print("=" * 70)
        print("Loading vace_fun_14B_cocktail_2_2...")
        orchestrator.load_model("vace_fun_14B_cocktail_2_2")
        print("✅ Model loaded")

        print("Running test with video1 and mask1...")
        output_path = orchestrator.generate_vace(
            prompt=prompt,
            video_guide="samples/video1.mp4",
            video_mask="samples/mask1.mp4",
            **params
        )

        if output_path:
            print(f"✅ Generated vace_2_2: {output_path}")
            return output_path
        else:
            print("❌ No output from vace_2_2")
            return None

    except Exception as e:
        print(f"❌ Error with vace_2_2: {e}")
        return None

    finally:
        orchestrator.unload_model()
        print("✅ vace_2_2 model unloaded")

def test_vace_3p_lightning():
    """Test vace_fun_14B_cocktail_lightning_3phase second"""
    params = {
        "video_prompt_type": "VM",
        "resolution": "768x576",
        "video_length": 81,
        "num_inference_steps": 6,        # Total steps 0-6
        "guidance_phases": 3,            # 3 KSamplers
        "guidance_scale": 3,             # KSampler 1: CFG=3
        "guidance2_scale": 1,            # KSampler 2: CFG=1
        "guidance3_scale": 1,            # KSampler 3: CFG=1
        "switch_threshold": 800,         # Step 2 threshold (KSampler 1→2)
        "switch_threshold2": 600,        # Step 4 threshold (KSampler 2→3)
        "model_switch_phase": 3,         # Switch to low noise model at phase 3
        "flow_shift": 5,                 # From config
        "seed": 12345,
        "negative_prompt": "blurry, low quality, distorted, static, overexposed",
        "sample_solver": "euler",
        "activated_loras": ["Wan2GP/loras/bloom.safetensors"],
        "loras_multipliers": "1.3"
    }

    prompt = "zooming out timelapse of a plant growing as the sun fades and the moon moves across the sky and becomes the sun, timlapsiagro"

    orchestrator = WanOrchestrator("Wan2GP")

    try:
        print("\n" + "=" * 70)
        print("TESTING vace_fun_14B_cocktail_lightning_3phase SECOND")
        print("=" * 70)
        print("Loading vace_fun_14B_cocktail_lightning_3phase...")
        orchestrator.load_model("vace_fun_14B_cocktail_lightning_3phase")
        print("✅ Model loaded")

        print("Running test with video1 and mask1...")
        output_path = orchestrator.generate_vace(
            prompt=prompt,
            video_guide="samples/video1.mp4",
            video_mask="samples/mask1.mp4",
            **params
        )

        if output_path:
            print(f"✅ Generated lightning_3phase: {output_path}")
            return output_path
        else:
            print("❌ No output from lightning_3phase")
            return None

    except Exception as e:
        print(f"❌ Error with lightning_3phase: {e}")
        return None

    finally:
        orchestrator.unload_model()
        print("✅ lightning_3phase model unloaded")

def main():
    """Run both models back to back"""
    print("🚀 Starting back-to-back model comparison")
    print("📝 Testing same prompt with both models for comparison")

    results = {}

    # Test vace_fun_14B_cocktail_2_2 FIRST
    print("\n🥇 FIRST: vace_fun_14B_cocktail_2_2")
    results["vace_2_2"] = test_vace_2_2()

    # Brief pause between tests
    import time
    print("\n⏳ Pausing 5 seconds between tests...")
    time.sleep(5)

    # Test vace_fun_14B_cocktail_lightning_3phase SECOND
    print("\n🥈 SECOND: vace_fun_14B_cocktail_lightning_3phase")
    results["lightning_3phase"] = test_vace_3p_lightning()

    # Summary
    print("\n" + "=" * 70)
    print("🏁 COMPARISON RESULTS")
    print("=" * 70)

    for model, output in results.items():
        status = "✅ SUCCESS" if output else "❌ FAILED"
        print(f"{model:25} | {status} | {output or 'No output'}")

    print("\n🎯 Test completed! Check outputs directory for results.")

if __name__ == "__main__":
    main()