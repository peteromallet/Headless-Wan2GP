import torch
import numpy as np
import os
import sys
from PIL import Image

# --- Path Setup ---
# Add the project's root directory to the Python path.
# This allows us to import 'Wan2GP' as a top-level package.
script_dir = os.path.dirname(os.path.abspath(__file__))
wan2gp_path = os.path.join(script_dir, 'Wan2GP')

# Validate submodule path
if not os.path.isdir(wan2gp_path):
    print("❌ ERROR: Wan2GP submodule not found!")
    print(f"Looked for directory: {wan2gp_path}")
    print("Please make sure you have cloned the submodule by running:")
    print("git submodule update --init --recursive")
    sys.exit(1)

# Add the script's directory to the path to allow top-level import
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Also add the Wan2GP directory to allow absolute imports within the package
if wan2gp_path not in sys.path:
    sys.path.insert(0, wan2gp_path)

# Now we can import from the Wan2GP package correctly.
try:
    from Wan2GP.wan.text2video import WanT2V
except ImportError as e:
    print(f"❌ ERROR: Failed to import from Wan2GP submodule.")
    print(f"Error details: {e}")
    print("This might be due to an incomplete or corrupted submodule.")
    print("Try running: git submodule update --init --recursive")
    sys.exit(1)

# --- Test Configuration ---
MODEL_FILENAME = "ckpts/wan2.1_Vace_1.3B_mbf16.safetensors"
# The new model seems to bundle the text encoder, so we might not need this explicitly.
TEXT_ENCODER_FILENAME = "ckpts/umt5-xxl/models_t5_umt5-xxl-enc-bf16.safetensors"
VAE_FILENAME = "ckpts/Wan2.1_VAE.safetensors"

# Use smaller dimensions and fewer steps for faster testing
WIDTH, HEIGHT = 480, 272  # Must be multiples of 16
FRAMES = 17  # Should be 4n+1
SAMPLING_STEPS = 10  # Low for speed
OUTPUT_DIR = "test_outputs"

# --- Helper Functions ---
def save_video_as_gif(video_tensor, filename, fps=8):
    """Saves a tensor of video frames as a GIF."""
    # Convert tensor to a list of PIL Images
    # Tensor shape: (C, F, H, W), range [-1, 1]
    video_tensor = video_tensor.permute(1, 2, 3, 0) # F, H, W, C
    video_tensor = (video_tensor * 0.5 + 0.5) * 255 # Denormalize to [0, 255]
    video_tensor = video_tensor.clamp(0, 255).to(torch.uint8).cpu().numpy()
    
    images = [Image.fromarray(frame) for frame in video_tensor]
    
    # Save as GIF
    duration = 1000 / fps
    images[0].save(filename, save_all=True, append_images=images[1:], duration=duration, loop=0)

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
    
    output_filename = os.path.join(OUTPUT_DIR, f"{test_name}.gif") # Save as gif now
    
    try:
        result = wan_model.generate(
            input_prompt=prompt,
            vace_configs=vace_configs,
            width=WIDTH,
            height=HEIGHT,
            frame_num=FRAMES,
            sampling_steps=SAMPLING_STEPS,
            seed=42, # Use a fixed seed for reproducibility
            model_filename=MODEL_FILENAME, # Pass model filename here now
        )
        
        if result is None:
            raise RuntimeError("Generation returned None.")
        
        # The output of generate is a tensor (C, F, H, W) with range [-1, 1]
        save_video_as_gif(result, output_filename, fps=8)
        
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
    # NOTE: Text encoder path check removed as it seems to be handled differently now.
    for f in [MODEL_FILENAME, VAE_FILENAME, TEXT_ENCODER_FILENAME]:
        if not os.path.exists(f):
            print(f"ERROR: Model file not found at {f}.")
            print("Please ensure you have downloaded the necessary models into the 'ckpts' directory.")
            sys.exit(1)

    try:
        # Create a mock config object based on what the new WanT2V expects
        class MockConfig:
            def __init__(self):
                self.num_train_timesteps = 1000
                self.param_dtype = torch.float32
                self.t5_dtype = torch.float16
                self.t5_tokenizer = "t5-v1_1-xxl"
                self.vae_checkpoint = VAE_FILENAME
                self.vae_stride = [4, 8, 8]
                self.patch_size = [1, 2, 2]
                self.sample_neg_prompt = ""
                self.sample_fps = 24
                self.text_len = 512  # Add the missing text_len attribute

        cfg = MockConfig()

        wan_model = WanT2V(
            config=cfg,
            checkpoint_dir="ckpts",
            model_filename=[MODEL_FILENAME],
            text_encoder_filename=TEXT_ENCODER_FILENAME, # Pass the text encoder filename
        )
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