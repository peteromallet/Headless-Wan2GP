#!/usr/bin/env python3
import sys
from pathlib import Path
from headless_wgp import WanOrchestrator

def test_vace_3p_lightning():
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
            print(f"✅ Generated: {output_path}")
        else:
            print("❌ No output")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        orchestrator.unload_model()
        print("Model unloaded")

if __name__ == "__main__":
    test_vace_3p_lightning()
