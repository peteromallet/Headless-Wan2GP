#!/usr/bin/env python3
"""
Example: Multi-VACE with reference images and guidance video
This example shows how to structure multi-VACE inputs in the same way as
headless.py and travel_between_images.py process them.
"""

import json

def extract_frames_from_video_mock(video_path, max_frames=None):
    """
    Mock function to simulate frame extraction from video.
    In reality, this would use cv2 to extract frames.
    """
    print(f"Mock: Extracting frames from {video_path} (max: {max_frames})")
    # Simulate extracting frames
    num_frames = min(max_frames or 81, 81)
    mock_frames = [f"frame_{i:03d}.png" for i in range(num_frames)]
    print(f"Mock: Extracted {len(mock_frames)} frames")
    return mock_frames

def load_reference_images_mock(image_paths):
    """
    Mock function to simulate loading reference images.
    In reality, this would load PIL Images.
    """
    loaded_images = []
    for img_path in image_paths:
        print(f"Mock: Loading reference image: {img_path}")
        loaded_images.append(f"PIL_Image({img_path})")
    return loaded_images

# Example usage matching the headless.py workflow
def create_multi_vace_task_example():
    """
    Create multi-VACE inputs in the same format used by headless.py
    """
    
    # --- Input file paths (similar to what would come from JSON task params) ---
    ref_image_paths = [
        "/path/to/frame_1.png",  # First reference image
        "/path/to/frame_2.png"   # Second reference image
    ]
    
    guidance_video_path = "/path/to/input.mp4"  # Guidance video
    
    # --- Process the inputs the same way headless.py does ---
    
    # 1. Load reference images (similar to sm_load_pil_images in headless.py)
    ref_images_pil = load_reference_images_mock(ref_image_paths)
    
    # 2. Extract frames from guidance video (similar to preprocess_video workflow)
    guidance_frames_pil = extract_frames_from_video_mock(guidance_video_path, max_frames=81)
    
    # 3. Structure multi_vace_inputs exactly like headless.py does
    multi_vace_inputs = [
        {
            # First VACE stream: Reference images for character consistency
            'frames': None,                    # No control frames for this stream
            'masks': None,                     # No masks for this stream  
            'ref_images': ref_images_pil,      # PIL Images (processed from paths)
            'strength': 0.25,                  # Lower strength for identity preservation
            'start_percent': 0.0,              # Apply from beginning
            'end_percent': 1.0                 # Apply until end
        },
        {
            # Second VACE stream: Guidance video for motion/pose control
            'frames': guidance_frames_pil,     # PIL Images (extracted from video)
            'masks': None,                     # No masks for this stream
            'ref_images': None,                # No reference images for this stream
            'strength': 1.0,                   # Full strength for motion guidance
            'start_percent': 0.0,              # Apply from beginning
            'end_percent': 0.9                 # Apply only for first 90% of steps
        }
    ]
    
    return multi_vace_inputs

# Example of how this would be used with WanT2V (matching wgp.py interface)
def generate_with_multi_vace():
    """
    Example showing how to call wan_model.generate() with the processed multi-VACE inputs
    """
    # Process the inputs first
    multi_vace_inputs = create_multi_vace_task_example()
    
    # This matches the parameter structure that wgp.py passes to wan_model.generate()
    generation_params = {
        "input_prompt": "A person dancing in a beautiful garden with flowers",
        "width": 1280,
        "height": 720, 
        "frame_num": 81,                       # Total frames to generate
        "sampling_steps": 30,                  # Number of diffusion steps
        "guide_scale": 5.0,                    # CFG guidance scale
        "seed": 42,                           # Random seed
        "multi_vace_inputs": multi_vace_inputs, # Our processed multi-VACE streams
        # Standard parameters that wgp.py typically passes
        "shift": 5.0,                         # Flow shift parameter
        "n_prompt": "",                       # Negative prompt
        "offload_model": True,                # Memory optimization
        "VAE_tile_size": 0,                   # VAE tiling (0 = auto)
        "enable_RIFLEx": True,                # Enable RIFLEx optimization
        "joint_pass": False,                  # Joint attention pass
    }
    
    # This is how wgp.py would call the model
    # video = wan_model.generate(**generation_params)
    
    print("Multi-VACE inputs processed and ready for generation:")
    print(f"Stream 1 (Reference): {len(multi_vace_inputs[0]['ref_images'])} reference images at strength {multi_vace_inputs[0]['strength']}")
    print(f"Stream 2 (Guidance): {len(multi_vace_inputs[1]['frames'])} guidance frames at strength {multi_vace_inputs[1]['strength']}")
    
    return generation_params

# Example showing the headless.py JSON task format
def create_headless_task_json():
    """
    Example of how this would be structured as a JSON task for headless.py
    This matches the format that headless.py expects in task_params_dict
    """
    task_json = {
        "task_type": "generate_video",
        "model": "vace",
        "prompt": "A person dancing in a beautiful garden with flowers",
        "negative_prompt": "ugly, blurry, distorted",
        "resolution": "1280x720",
        "frames": 81,
        "steps": 30,
        "guidance_scale": 5.0,
        "seed": 42,
        
        # Multi-VACE configuration (what headless.py processes)
        "multi_vace_inputs": [
            {
                # Reference images stream
                "ref_image_paths": ["/path/to/frame_1.png", "/path/to/frame_2.png"],
                "strength": 0.25,
                "start_percent": 0.0,
                "end_percent": 1.0
            },
            {
                # Guidance video stream (headless.py will extract frames)
                "frame_paths": ["/path/to/input.mp4"],  # Single video file
                "strength": 1.0,
                "start_percent": 0.0,
                "end_percent": 0.9
            }
        ]
    }
    
    return task_json

def compare_with_headless_processing():
    """
    Compare this example with the actual headless.py processing logic
    """
    print("=== COMPARISON: Example vs headless.py ===\n")
    
    # Show the JSON format for headless.py
    task_json = create_headless_task_json()
    print("1. JSON INPUT (for headless.py):")
    print(json.dumps(task_json, indent=2))
    
    print("\n" + "="*50 + "\n")
    
    print("2. HEADLESS.PY PROCESSING:")
    print("   • Receives multi_vace_inputs from task_params_dict")
    print("   • For each vace_input:")
    print("     - Gets frame_paths/frames → loads with sm_load_pil_images()")
    print("     - Gets ref_image_paths/ref_images → loads with sm_load_pil_images()")  
    print("     - Gets mask_paths/masks → loads with sm_load_pil_images()")
    print("     - Validates: at least frames OR masks must be present")
    print("     - Creates processed_vace dict with PIL Images")
    print("   • Sets ui_defaults['multi_vace_inputs'] = processed_multi_vace")
    
    print("\n" + "="*50 + "\n")
    
    print("3. PROCESSED OUTPUT (what gets passed to wgp.py):")
    processed = create_multi_vace_task_example()
    for i, stream in enumerate(processed):
        print(f"   Stream {i+1}:")
        for key, value in stream.items():
            if key in ['frames', 'ref_images', 'masks']:
                count = len(value) if value else 0
                print(f"     {key}: {count} items ({'PIL Images' if value else 'None'})")
            else:
                print(f"     {key}: {value}")
        print()
    
    print("4. KEY DIFFERENCES TO NOTE:")
    print("   ✓ JSON uses 'frame_paths' but headless.py accepts 'frame_paths' OR 'frames'")
    print("   ✓ JSON uses 'ref_image_paths' but headless.py accepts 'ref_image_paths' OR 'ref_images'")
    print("   ✓ Processed output has PIL Images, not file paths")
    print("   ✓ headless.py validates streams (needs frames OR masks)")
    print("   ✓ headless.py uses sm_load_pil_images() for robust file loading")

if __name__ == "__main__":
    print("=== Multi-VACE Example ===")
    print("This example shows how to structure multi-VACE inputs")
    print("in the same way as headless.py processes them\n")
    
    compare_with_headless_processing()
    
    print("\n=== Notes ===")
    print("- Reference images are loaded as individual PNG files")
    print("- Guidance video is extracted into individual frames") 
    print("- All processing matches headless.py → wgp.py → text2video.py workflow")
    print("- Multi-VACE streams can have different strengths and timing") 