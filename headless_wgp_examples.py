#!/usr/bin/env python3
"""
WanGP Generation Examples

This script demonstrates how to use the WanOrchestrator for different generation scenarios:
1. T2V with LoRA
2. VACE with video guide + mask + LoRA  
3. VACE with video guide + optical flow + mask + LoRA
4. Flux with LoRA

Prerequisites:
- WanGP repository cloned and set up
- Required models downloaded (happens automatically on first use)
- Input videos/masks in the inputs/ directory for VACE examples
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import orchestrator
sys.path.insert(0, str(Path(__file__).parent.parent))

from headless_wgp import WanOrchestrator


def setup_example_inputs():
    """Create inputs directory and example files if they don't exist."""
    inputs_dir = Path("inputs")
    inputs_dir.mkdir(exist_ok=True)
    
    # Create placeholder files if they don't exist
    example_files = {
        "dance_control.mp4": "Place your control video here (e.g., pose/dance video)",
        "dance_mask.mp4": "Place your mask video here (black=keep, white=overwrite)",
        "runner_control.mp4": "Place your running/motion video here",
        "runner_mask.mp4": "Place corresponding mask video here"
    }
    
    for filename, description in example_files.items():
        filepath = inputs_dir / filename
        if not filepath.exists():
            print(f"📝 Create {filepath} - {description}")


def example_1_t2v_with_lora(orchestrator: WanOrchestrator):
    """Example 1: Text-to-Video with LoRA"""
    print("\n" + "="*60)
    print("🎬 EXAMPLE 1: T2V with LoRA")
    print("="*60)
    
    # Load T2V model
    orchestrator.load_model("t2v")
    
    # Generate T2V video with LoRA
    result = orchestrator.generate_t2v(
        prompt="a majestic dragon flying over a mystical forest at sunset, cinematic lighting, 4k",
        resolution="1280x720",
        video_length=49,
        num_inference_steps=25,
        guidance_scale=7.5,
        seed=42,
        # LoRA configuration
        lora_names=["cinematic_style.safetensors"],  # Replace with actual LoRA filename
        lora_multipliers=[1.0],
        negative_prompt="blurry, low quality, distorted"
    )
    
    print(f"✅ T2V with LoRA completed: {result}")
    return result


def example_2_vace_with_guide_mask_lora(orchestrator: WanOrchestrator):
    """Example 2: VACE with video guide + mask + LoRA"""
    print("\n" + "="*60)
    print("🎯 EXAMPLE 2: VACE with Guide + Mask + LoRA")
    print("="*60)
    
    # Load VACE model
    orchestrator.load_model("vace_14B")
    
    # Check if input files exist
    video_guide = "inputs/dance_control.mp4"
    video_mask = "inputs/dance_mask.mp4"
    
    if not os.path.exists(video_guide):
        print(f"⚠️  Warning: {video_guide} not found. Create this file first.")
        return None
    
    # Generate VACE video with guide, mask, and LoRA
    result = orchestrator.generate_vace(
        prompt="a graceful dancer in flowing robes, ethereal lighting, fantasy art style",
        video_guide=video_guide,
        video_mask=video_mask,
        video_prompt_type="VPA",  # V=video guide, P=pose, A=mask
        control_net_weight=1.0,   # Pose strength
        resolution="1280x720",
        video_length=49,
        num_inference_steps=25,
        guidance_scale=7.5,
        seed=123,
        # LoRA configuration
        lora_names=["fantasy_style.safetensors", "quality_boost.safetensors"],
        lora_multipliers=[1.2, 0.8],
        negative_prompt="ugly, deformed, blurry, low quality"
    )
    
    print(f"✅ VACE with guide + mask + LoRA completed: {result}")
    return result


def example_3_vace_with_optical_flow_mask_lora(orchestrator: WanOrchestrator):
    """Example 3: VACE with video guide + optical flow + mask + LoRA"""
    print("\n" + "="*60)
    print("🌊 EXAMPLE 3: VACE with Pose + Optical Flow + Mask + LoRA")
    print("="*60)
    
    # Load VACE model (reuse if already loaded)
    orchestrator.load_model("vace_14B")
    
    # Check if input files exist
    video_guide = "inputs/runner_control.mp4"
    video_mask = "inputs/runner_mask.mp4"
    
    if not os.path.exists(video_guide):
        print(f"⚠️  Warning: {video_guide} not found. Create this file first.")
        return None
    
    # Generate VACE video with pose + optical flow + mask + LoRA
    result = orchestrator.generate_vace(
        prompt="a cyberpunk runner in neon-lit city streets, futuristic atmosphere, dramatic motion blur",
        video_guide=video_guide,
        video_mask=video_mask,
        video_prompt_type="VPLA",  # V=video, P=pose, L=optical flow, A=mask
        control_net_weight=1.0,    # Pose strength (first encoding)
        control_net_weight2=0.8,   # Optical flow strength (second encoding)
        resolution="1280x720",
        video_length=49,
        num_inference_steps=30,
        guidance_scale=8.0,
        seed=456,
        # LoRA configuration
        lora_names=["cyberpunk_style.safetensors", "motion_enhancement.safetensors"],
        lora_multipliers=[1.0, 0.6],
        negative_prompt="static, boring, low quality, blurry"
    )
    
    print(f"✅ VACE with pose + optical flow + mask + LoRA completed: {result}")
    
    # ===============================
    # OTHER CONTROL NET COMBINATIONS
    # ===============================
    # You can replace optical flow (L) with other control types:
    
    # 🎯 POSE + DEPTH: Great for preserving human poses and scene geometry
    # result = orchestrator.generate_vace(
    #     prompt="a warrior in ancient armor standing on a cliff overlooking the ocean",
    #     video_guide=video_guide,
    #     video_mask=video_mask,
    #     video_prompt_type="VPDA",  # V=video, P=pose, D=depth, A=mask
    #     control_net_weight=1.2,    # Pose strength (higher for precise pose control)
    #     control_net_weight2=0.6,   # Depth strength (lower to avoid over-constraining)
    #     ...
    # )
    
    # 🖋️ POSE + SCRIBBLE: Useful for artistic style control with pose guidance
    # result = orchestrator.generate_vace(
    #     prompt="an artist painting in a studio, impressionist style",
    #     video_guide=video_guide,
    #     video_mask=video_mask,
    #     video_prompt_type="VPSA",  # V=video, P=pose, S=scribble, A=mask
    #     control_net_weight=0.9,    # Pose strength
    #     control_net_weight2=1.1,   # Scribble strength (higher for artistic effect)
    #     ...
    # )
    
    # 📐 POSE + CANNY: Excellent for preserving pose with sharp edge definition
    # result = orchestrator.generate_vace(
    #     prompt="a dancer in silhouette against dramatic lighting",
    #     video_guide=video_guide,
    #     video_mask=video_mask,
    #     video_prompt_type="VPEA",  # V=video, P=pose, E=canny, A=mask
    #     control_net_weight=1.0,    # Pose strength
    #     control_net_weight2=0.7,   # Canny strength (moderate for clean edges)
    #     ...
    # )
    
    # 🎨 DEPTH + CANNY: Combine 3D understanding with edge preservation
    # result = orchestrator.generate_vace(
    #     prompt="architectural marvel with intricate details",
    #     video_guide=video_guide,
    #     video_mask=video_mask,
    #     video_prompt_type="VDEA",  # V=video, D=depth, E=canny, A=mask
    #     control_net_weight=0.8,    # Depth strength
    #     control_net_weight2=1.0,   # Canny strength
    #     ...
    # )
    
    # 🌫️ DEPTH + SCRIBBLE: 3D structure with artistic interpretation
    # result = orchestrator.generate_vace(
    #     prompt="mystical landscape with ethereal fog, concept art style",
    #     video_guide=video_guide,
    #     video_mask=video_mask,
    #     video_prompt_type="VDSA",  # V=video, D=depth, S=scribble, A=mask
    #     control_net_weight=0.7,    # Depth strength (lower for dreamlike effect)
    #     control_net_weight2=1.2,   # Scribble strength (higher for artistic style)
    #     ...
    # )
    
    # 🎭 POSE + GRAYSCALE: Preserve pose while allowing color reinterpretation
    # result = orchestrator.generate_vace(
    #     prompt="vintage film noir detective in rain-soaked streets",
    #     video_guide=video_guide,
    #     video_mask=video_mask,
    #     video_prompt_type="VPCA",  # V=video, P=pose, C=grayscale, A=mask
    #     control_net_weight=1.1,    # Pose strength
    #     control_net_weight2=0.5,   # Grayscale strength (low for color freedom)
    #     ...
    # )
    
    # 🔧 INPAINT + OPTICAL FLOW: Motion-aware object replacement
    # result = orchestrator.generate_vace(
    #     prompt="magical creature running through enchanted forest",
    #     video_guide=video_guide,
    #     video_mask=video_mask,
    #     video_prompt_type="VMLA",  # V=video, M=inpaint, L=optical flow, A=mask
    #     control_net_weight=1.0,    # Inpaint strength
    #     control_net_weight2=0.9,   # Flow strength (preserve motion)
    #     ...
    # )
    
    # 📝 CONTROL TYPE REFERENCE:
    # P = Pose (OpenPose)      - Human pose detection, best for people/characters
    # D = Depth (MiDaS)        - 3D depth understanding, great for scene geometry  
    # S = Scribble             - Artistic line control, good for stylization
    # E = Canny                - Edge detection, preserves sharp boundaries
    # L = Optical Flow         - Motion vectors, excellent for movement preservation
    # C = Grayscale            - Luminance control, allows color reinterpretation
    # M = Inpaint              - Object replacement within mask areas
    # U = Unprocessed RGB      - Raw pixel control, strongest constraint
    # A = Apply Mask           - Enable mask usage (not a control type)
    
    # 💡 STRENGTH TUNING TIPS:
    # - Higher values (1.0-2.0) = stronger control, less creative freedom
    # - Lower values (0.3-0.7) = looser control, more artistic interpretation  
    # - Balance the two weights based on which aspect is more important
    # - For pose+depth: often pose=1.0-1.2, depth=0.6-0.8
    # - For artistic combos: lower first weight, higher second weight
    
    return result


def example_3b_vace_dual_source_example(orchestrator: WanOrchestrator):
    """Example 3B: VACE with SEPARATE videos for each encoding (NEW FEATURE) - Using Wan 2.2 for speed"""
    print("\n" + "="*60)
    print("🔥 EXAMPLE 3B: VACE with Dual-Source Control (NEW!) - Wan 2.2 Accelerated")
    print("="*60)
    
    # Load VACE 2.2 model (faster, pre-optimized with built-in LoRAs)
    orchestrator.load_model("vace_14B_cocktail_2_2")
    
    # Check if input files exist
    pose_video = "inputs/dance_control.mp4"  # For pose extraction
    flow_video = "inputs/runner_control.mp4"  # For optical flow extraction
    mask_video = "inputs/dance_mask.mp4"
    
    if not (os.path.exists(pose_video) and os.path.exists(flow_video)):
        print(f"⚠️  Warning: Need both {pose_video} and {flow_video} for dual-source example")
        print("This example shows how to use DIFFERENT videos for each ControlNet encoding")
        return None
    
    # Generate VACE video with SEPARATE sources for pose and flow
    result = orchestrator.generate_vace(
        prompt="a dancer with flowing motion in a cyberpunk environment, neon lights, dynamic energy",
        
        # Primary control (pose from dance video)
        video_guide=pose_video,           # Pose extracted from dance video
        video_mask=mask_video,
        
        # Secondary control (flow from runner video) - NEW!
        video_guide2=flow_video,          # Flow extracted from runner video  
        video_mask2=None,                 # Use same mask for both
        
        video_prompt_type="VPLA",         # V=video, P=pose, L=optical flow, A=mask
        control_net_weight=1.2,           # Strong pose control (from dance)
        control_net_weight2=0.7,          # Moderate flow control (from runner)
        
        resolution="1280x720",
        video_length=49,
        # Wan 2.2 optimized settings (much faster than 2.1)
        num_inference_steps=10,    # 🚀 Reduced from 25 (2.5x faster)
        guidance_scale=1.0,        # 🎯 Distilled guidance (was 7.5)
        seed=999,
        
        # LoRA configuration (Wan 2.2 has built-in acceleration LoRAs)
        lora_names=["cyberpunk_style.safetensors"],  # Only custom style (speed LoRAs built-in)
        lora_multipliers=[1.0],
        negative_prompt="static, boring, low quality"
    )
    
    print(f"✅ Dual-source VACE completed: {result}")
    print("🎯 This used Wan 2.2 features:")
    print(f"   • Pose from: {pose_video}")
    print(f"   • Motion flow from: {flow_video}")
    print(f"   • Combined both with independent strength control!")
    print(f"   • Wan 2.2: 10 steps vs 25 steps (2.5x faster generation)")
    print(f"   • Built-in acceleration LoRAs for speed optimization")
    
    return result


def example_4_flux_with_lora(orchestrator: WanOrchestrator):
    """Example 4: Flux image generation with LoRA"""
    print("\n" + "="*60)
    print("🖼️  EXAMPLE 4: Flux with LoRA")
    print("="*60)
    
    # Load Flux model
    orchestrator.load_model("flux")
    
    # Generate multiple Flux images with LoRA
    result = orchestrator.generate_flux(
        prompt="portrait of a mystical sorceress with glowing eyes, intricate magical symbols, ultra detailed, fantasy art",
        num_images=2,
        resolution="1024x1024",
        num_inference_steps=20,
        embedded_guidance_scale=3.5,
        seed=789,
        # LoRA configuration
        lora_names=["fantasy_portrait.safetensors", "detail_enhancer.safetensors"],
        lora_multipliers=[1.1, 0.7],
        negative_prompt="blurry, low quality, deformed, ugly"
    )
    
    print(f"✅ Flux with LoRA completed: {result}")
    return result


def main():
    """Run all generation examples."""
    print("🚀 WanGP Generation Examples")
    print("This script demonstrates T2V, VACE, and Flux generation with LoRAs")
    
    # Initialize orchestrator
    # Update this path to your WanGP directory
    WAN_DIR = "/path/to/WanGP"  # ⚠️ CHANGE THIS TO YOUR ACTUAL PATH
    
    if not os.path.exists(WAN_DIR):
        print(f"❌ Error: WanGP directory not found at {WAN_DIR}")
        print("Please update the WAN_DIR variable with the correct path to your WanGP installation")
        return
    
    if not os.path.exists(os.path.join(WAN_DIR, "wgp.py")):
        print(f"❌ Error: wgp.py not found in {WAN_DIR}")
        print("Please ensure you're pointing to the correct WanGP directory")
        return
    
    # Set up example inputs
    setup_example_inputs()
    
    try:
        # Initialize orchestrator
        orchestrator = WanOrchestrator(WAN_DIR)
        
        # Run examples
        results = []
        
        # Example 1: T2V with LoRA
        result1 = example_1_t2v_with_lora(orchestrator)
        if result1:
            results.append(("T2V + LoRA", result1))
        
        # Example 2: VACE with guide + mask + LoRA
        result2 = example_2_vace_with_guide_mask_lora(orchestrator)
        if result2:
            results.append(("VACE + Guide + Mask + LoRA", result2))
        
        # Example 3: VACE with guide + optical flow + mask + LoRA
        result3 = example_3_vace_with_optical_flow_mask_lora(orchestrator)
        if result3:
            results.append(("VACE + Pose + Flow + Mask + LoRA", result3))
        
        # Example 3B: VACE with dual-source control (NEW!)
        result3b = example_3b_vace_dual_source_example(orchestrator)
        if result3b:
            results.append(("VACE + Dual-Source Control (NEW!)", result3b))
        
        # Example 4: Flux with LoRA
        result4 = example_4_flux_with_lora(orchestrator)
        if result4:
            results.append(("Flux + LoRA", result4))
        
        # Summary
        print("\n" + "="*60)
        print("📋 GENERATION SUMMARY")
        print("="*60)
        
        if results:
            for name, path in results:
                print(f"✅ {name}: {path}")
        else:
            print("⚠️  No generations completed. Check input files and paths.")
            
        print(f"\n🎉 Generated {len(results)} outputs successfully!")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure WAN_DIR points to the correct WanGP directory")
        print("2. Check that all required models are available")
        print("3. For VACE examples, ensure input videos exist in inputs/ directory")
        print("4. Verify LoRA files exist in the loras/ directory")


if __name__ == "__main__":
    main()