import torch
import numpy as np
import os
import sys

# --- Path Setup ---
# Add the 'Wan2GP' directory to the Python path to allow imports from 'wan'
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "Wan2GP")
if not os.path.isdir(project_root):
    raise FileNotFoundError("Could not find the 'Wan2GP' directory. Please run this script from the project root.")
sys.path.insert(0, project_root)

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import cache_video
from PIL import Image

# --- Test Configuration ---
MODEL_FILENAME = "ckpts/wan2.1_Vace_1.3B_mbf16.safetensors"
TEXT_ENCODER_FILENAME = "ckpts/umt5-xxl/models_t5_umt5-xxl-enc-bf16.safetensors"
VAE_FILENAME = "ckpts/Wan2.1_VAE.safetensors"

# Use smaller dimensions and fewer steps for faster testing
WIDTH, HEIGHT = 480, 272  # Must be multiples of 16
FRAMES = 17  # Should be 4n+1
SAMPLING_STEPS = 10  # Low for speed
OUTPUT_DIR = "test_outputs"

# --- Helper Functions ---
def create_dummy_vace_input(width, height, frames, color='white'):
    """
    Creates dummy tensors for a single VACE stream input.
    Generates a moving square pattern.
    """
    # Create tensors in [-1, 1] range. -1 is black, 1 is white.
    frame_tensors = torch.full((3, frames, height, width), -1.0)
    # Mask is [0, 1]. 1 means the area is to be inpainted/controlled.
    mask_tensors = torch.zeros((1, frames, height, width))
    
    square_size = 32
    for i in range(frames):
        x_pos = int((i / (frames - 1)) * (width - square_size)) if frames > 1 else int((width - square_size) / 2)
        y_pos = int((height - square_size) / 2)

        if color == 'white':
            frame_tensors[:, i, y_pos:y_pos+square_size, x_pos:x_pos+square_size] = 1.0
        elif color == 'red':
            frame_tensors[0, i, y_pos:y_pos+square_size, x_pos:x_pos+square_size] = 1.0
            frame_tensors[1, i, y_pos:y_pos+square_size, x_pos:x_pos+square_size] = -1.0
            frame_tensors[2, i, y_pos:y_pos+square_size, x_pos:x_pos+square_size] = -1.0
            
        mask_tensors[:, i, y_pos:y_pos+square_size, x_pos:x_pos+square_size] = 1.0
        
    # generate() expects a list of tensors for each param (for batching, although we use batch size 1)
    return [frame_tensors], [mask_tensors]

def run_test(wan_model, test_name, prompt, vace_configs):
    """Runs a single generation test case."""
    print(f"--- Running Test: {test_name} ---")
    
    output_filename = os.path.join(OUTPUT_DIR, f"{test_name}.mp4")
    
    try:
        result = wan_model.generate(
            input_prompt=prompt,
            vace_configs=vace_configs,
            width=WIDTH,
            height=HEIGHT,
            frame_num=FRAMES,
            sampling_steps=SAMPLING_STEPS,
            seed=42, # Use a fixed seed for reproducibility
        )
        
        if result is None:
            raise RuntimeError("Generation returned None.")
        
        # The output of generate is a tensor (C, F, H, W) with range [-1, 1]
        cache_video(result.unsqueeze(0), output_filename, fps=8)
        
        print(f"✅ SUCCESS: {test_name}. Output saved to {output_filename}")
        
    except Exception as e:
        print(f"❌ FAILED: {test_name}")
        import traceback
        traceback.print_exc()

# --- Main Test Execution ---
def main():
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading VACE model... this may take a while.")
    
    # Check if necessary model files exist
    for f in [MODEL_FILENAME, TEXT_ENCODER_FILENAME, VAE_FILENAME]:
        if not os.path.exists(f):
            print(f"ERROR: Model file not found at {f}.")
            print("Please ensure you have downloaded the necessary models into the 'ckpts' directory.")
            sys.exit(1)

    try:
        # We need to use the 't2v-14B' config as VACE models are based on it
        cfg = WAN_CONFIGS['t2v-14B']
        wan_model = wan.WanT2V(
            config=cfg,
            checkpoint_dir="ckpts",
            model_filename=[MODEL_FILENAME], # expects a list
            text_encoder_filename=TEXT_ENCODER_FILENAME,
        )
        wan_model.vae.vae_pth = VAE_FILENAME # Correct the VAE path
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- Test Cases ---
    
    # Test Case 1: No VACE conditioning (baseline)
    run_test(wan_model, "no_vace", "a red car driving on a highway", None)

    # Test Case 2: Single VACE stream
    frames, masks = create_dummy_vace_input(WIDTH, HEIGHT, FRAMES, color='white')
    scale_array = np.ones(SAMPLING_STEPS)
    vace_config_single = [{
        'id': 'stream1',
        'frames': frames, 'masks': masks, 'ref_images': None,
        'scale_per_step_array': scale_array, 'start_percent': 0.0, 'end_percent': 1.0
    }]
    run_test(wan_model, "single_vace_stream", "a moving object", vace_config_single)

    # Test Case 3: Multiple VACEs, non-overlapping
    frames1, masks1 = create_dummy_vace_input(WIDTH, HEIGHT, FRAMES, color='white')
    frames2, masks2 = create_dummy_vace_input(WIDTH, HEIGHT, FRAMES, color='red')
    
    vace_config_non_overlap = [
        { 'id': 'stream1_first_half', 'frames': frames1, 'masks': masks1, 'ref_images': None,
          'scale_per_step_array': np.ones(SAMPLING_STEPS), 'start_percent': 0.0, 'end_percent': 0.5 },
        { 'id': 'stream2_second_half', 'frames': frames2, 'masks': masks2, 'ref_images': None,
          'scale_per_step_array': np.ones(SAMPLING_STEPS), 'start_percent': 0.5, 'end_percent': 1.0 }
    ]
    run_test(wan_model, "multi_vace_non_overlapping", "a moving object changing color", vace_config_non_overlap)
    
    # Test Case 4: Multiple VACEs, overlapping
    vace_config_overlap = [
        { 'id': 'stream1_full', 'frames': frames1, 'masks': masks1, 'ref_images': None,
          'scale_per_step_array': np.ones(SAMPLING_STEPS) * 0.5, 'start_percent': 0.0, 'end_percent': 1.0 }, # Scale down to see both
        { 'id': 'stream2_full', 'frames': frames2, 'masks': masks2, 'ref_images': None,
          'scale_per_step_array': np.ones(SAMPLING_STEPS) * 0.5, 'start_percent': 0.0, 'end_percent': 1.0 }
    ]
    run_test(wan_model, "multi_vace_overlapping", "a pink moving object", vace_config_overlap)

    # Test Case 5: VACE with varying scale
    scale_array_varying = np.linspace(0, 1.5, SAMPLING_STEPS)
    vace_config_varying_scale = [{
        'id': 'stream_varying_scale', 'frames': frames, 'masks': masks, 'ref_images': None,
        'scale_per_step_array': scale_array_varying, 'start_percent': 0.0, 'end_percent': 1.0
    }]
    run_test(wan_model, "single_vace_varying_scale", "a fading-in moving object", vace_config_varying_scale)
    
    print("\n--- All tests completed. Check 'test_outputs' directory for results. ---")

if __name__ == "__main__":
    main() 