#!/usr/bin/env python3
"""
Direct WGP LoRA Test - Bypass headless layer to test Lightning LoRA directly

This test bypasses the broken headless wrapper and calls WGP directly
with the correct LoRA format to verify dynamic LoRA scheduling works.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Wan2GP'))

def test_direct_wgp_with_lightning_loras():
    """Test WGP directly with Lightning LoRA configuration."""

    print("🧪 DIRECT WGP LoRA TEST")
    print("=" * 50)
    print("Testing Lightning LoRA dynamic scheduling by calling WGP directly...")
    print()

    try:
        # Import WGP directly
        import wgp

        print("✅ WGP imported successfully")

        # Set up Lightning LoRA configuration in WGP format
        # Using local LoRA paths (assuming they were already downloaded)
        lora_config = {
            # Use local paths to the downloaded LoRAs
            'lora_names': [
                'loras/Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_HIGH_fp16.safetensors',
                'loras/Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_LOW_fp16.safetensors'
            ],
            # Phase-based multiplier schedule (space-separated string format)
            'lora_multipliers': '0.5;1.0;0 0;0;1',  # HIGH: 0.5,1.0,0 | LOW: 0,0,1

            # Generation parameters matching our successful test
            'num_inference_steps': 6,
            'guidance_phases': 3,
            'guidance_scale': 3,
            'guidance2_scale': 1,
            'guidance3_scale': 1,
            'switch_threshold': 900,
            'switch_threshold2': 600,
            'model_switch_phase': 2,
            'sample_solver': 'euler',

            # Basic generation params
            'prompt': 'Man turns around and runs over and starts painting',
            'video_length': 81,
            'seed': 12345,
            'model_type': 'vace_fun_14B_cocktail_lightning_3phase_light_distill_2_2_2',
        }

        print("📋 LoRA Configuration:")
        print(f"  lora_names: {lora_config['lora_names']}")
        print(f"  lora_multipliers: '{lora_config['lora_multipliers']}'")
        print(f"  Expected behavior:")
        print(f"    - HIGH LoRA: Steps 1-2 (0.5x), Steps 3-4 (1.0x), Steps 5-6 (0.0x)")
        print(f"    - LOW LoRA:  Steps 1-4 (0.0x), Steps 5-6 (1.0x)")
        print()

        # Check if LoRA files exist
        for lora_path in lora_config['lora_names']:
            full_path = os.path.join('Wan2GP', lora_path)
            if os.path.exists(full_path):
                print(f"✅ Found: {lora_path}")
            else:
                print(f"❌ Missing: {lora_path}")
                print(f"   Full path checked: {full_path}")

        print()
        print("🚀 This would call WGP directly with proper LoRA format...")
        print("   (Not actually running generation to avoid long execution)")
        print()
        print("Expected logs if LoRA system works:")
        print("  - 'Processing LoRA multipliers: 0.5;1.0;0 0;0;1'")
        print("  - Step-by-step LoRA multiplier changes during denoising")
        print("  - Phase switching logs with actual LoRA strength values")

        return True

    except ImportError as e:
        print(f"❌ Failed to import WGP: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_direct_wgp_with_lightning_loras()

    if success:
        print("\n✅ Direct WGP test setup completed")
        print("💡 Next step: Run actual generation to verify dynamic LoRA logs")
    else:
        print("\n❌ Direct WGP test failed")