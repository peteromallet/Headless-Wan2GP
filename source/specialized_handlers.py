"""Specialized task handlers for headless.py."""

import sys
from pathlib import Path

# Add the parent directory to Python path to allow Wan2GP module import
# This is crucial for ensuring that imports like `from ltx_video...` work correctly.
wan2gp_dir = Path(__file__).resolve().parent.parent / "Wan2GP"
if str(wan2gp_dir) not in sys.path:
    sys.path.insert(0, str(wan2gp_dir))

import traceback
import tempfile
import numpy as np
from PIL import Image
import cv2

try:
    from preprocessing.dwpose.pose import PoseBodyFaceVideoAnnotator
except ImportError:
    PoseBodyFaceVideoAnnotator = None

from . import db_operations as db_ops
from .common_utils import sm_get_unique_target_path
from .common_utils import parse_resolution as sm_parse_resolution
from .video_utils import rife_interpolate_images_to_video as sm_rife_interpolate_images_to_video

def handle_generate_openpose_task(task_params_dict: dict, main_output_dir_base: Path, task_id: str, dprint: callable):
    """Handles the 'generate_openpose' task."""
    print(f"[Task ID: {task_id}] Handling 'generate_openpose' task.")
    input_image_path_str = task_params_dict.get("input_image_path")
    suggested_output_image_path_str = task_params_dict.get("output_path")

    if PoseBodyFaceVideoAnnotator is None:
        msg = "PoseBodyFaceVideoAnnotator not imported. Cannot process 'generate_openpose' task."
        print(f"[ERROR Task ID: {task_id}] {msg}")
        return False, "PoseBodyFaceVideoAnnotator module not available."

    if not input_image_path_str:
        print(f"[ERROR Task ID: {task_id}] 'input_image_path' not specified for generate_openpose task.")
        return False, "Missing input_image_path"

    if not suggested_output_image_path_str:
        default_output_dir = main_output_dir_base / task_id
        default_output_dir.mkdir(parents=True, exist_ok=True)
        output_image_path = default_output_dir / f"{task_id}_openpose.png"
        print(f"[WARNING Task ID: {task_id}] 'output_path' not specified. Defaulting to {output_image_path}")
    else:
        output_image_path = Path(suggested_output_image_path_str)

    input_image_path = Path(input_image_path_str)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_image_path.is_file():
        print(f"[ERROR Task ID: {task_id}] Input image file not found: {input_image_path}")
        return False, f"Input image not found: {input_image_path}"

    final_save_path = output_image_path
    db_output_location = str(output_image_path.resolve())

    if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH:
        sqlite_db_file_path = Path(db_ops.SQLITE_DB_PATH).resolve()
        target_files_dir = sqlite_db_file_path.parent / "public" / "files"
        target_files_dir.mkdir(parents=True, exist_ok=True)

        file_stem = f"{task_id}_openpose"
        final_save_path = sm_get_unique_target_path(target_files_dir, file_stem, ".png")
        final_save_path.parent.mkdir(parents=True, exist_ok=True)
        db_output_location = f"files/{final_save_path.name}"
        dprint(f"[Task ID: {task_id}] SQLite mode: OpenPose will be saved to {final_save_path} (DB: {db_output_location})")

    try:
        pil_input_image = Image.open(input_image_path).convert("RGB")

        pose_cfg_dict = {
            "DETECTION_MODEL": "ckpts/pose/yolox_l.onnx",
            "POSE_MODEL": "ckpts/pose/dw-ll_ucoco_384.onnx",
            "RESIZE_SIZE": 1024
        }
        if PoseBodyFaceVideoAnnotator is None:
             raise ImportError("PoseBodyFaceVideoAnnotator could not be imported.")

        pose_annotator = PoseBodyFaceVideoAnnotator(pose_cfg_dict)

        openpose_np_frames_bgr = pose_annotator.forward([pil_input_image])

        if not openpose_np_frames_bgr or openpose_np_frames_bgr[0] is None:
            print(f"[ERROR Task ID: {task_id}] OpenPose generation failed or returned no frame.")
            return False, "OpenPose generation returned no data."

        openpose_np_frame_bgr = openpose_np_frames_bgr[0]

        openpose_pil_image = Image.fromarray(openpose_np_frame_bgr.astype(np.uint8))
        openpose_pil_image.save(final_save_path)

        print(f"[Task ID: {task_id}] Successfully generated OpenPose image to: {final_save_path.resolve()}")
        return True, db_output_location

    except ImportError as ie:
        print(f"[ERROR Task ID: {task_id}] Import error during OpenPose generation: {ie}. Ensure 'preprocessing' module is in PYTHONPATH and dependencies are installed.")
        traceback.print_exc()
        return False, f"Import error: {ie}"
    except FileNotFoundError as fnfe:
        print(f"[ERROR Task ID: {task_id}] ONNX model file not found for OpenPose: {fnfe}. Ensure 'ckpts/pose/*' models are present.")
        traceback.print_exc()
        return False, f"ONNX model not found: {fnfe}"
    except Exception as e:
        print(f"[ERROR Task ID: {task_id}] Failed during OpenPose image generation: {e}")
        traceback.print_exc()
        return False, f"OpenPose generation exception: {e}"


def handle_rife_interpolate_task(wgp_mod, task_params_dict: dict, main_output_dir_base: Path, task_id: str, dprint: callable):
    """Handles the 'rife_interpolate_images' task."""
    print(f"[Task ID: {task_id}] Handling 'rife_interpolate_images' task.")

    input_image_path1_str = task_params_dict.get("input_image_path1")
    input_image_path2_str = task_params_dict.get("input_image_path2")
    output_video_path_str = task_params_dict.get("output_path")
    num_rife_frames = task_params_dict.get("frames")
    resolution_str = task_params_dict.get("resolution")

    required_params = {
        "input_image_path1": input_image_path1_str,
        "input_image_path2": input_image_path2_str,
        "output_path": output_video_path_str,
        "frames": num_rife_frames,
        "resolution": resolution_str
    }
    missing_params = [key for key, value in required_params.items() if value is None]
    if missing_params:
        error_msg = f"Missing required parameters for rife_interpolate_images: {', '.join(missing_params)}"
        print(f"[ERROR Task ID: {task_id}] {error_msg}")
        return False, error_msg

    input_image1_path = Path(input_image_path1_str)
    input_image2_path = Path(input_image_path2_str)
    output_video_path = Path(output_video_path_str)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    generation_success = False
    output_location_to_db = None

    final_save_path_for_video = output_video_path
    db_output_location_for_rife = str(output_video_path.resolve())

    if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH:
        sqlite_db_file_path = Path(db_ops.SQLITE_DB_PATH).resolve()
        target_files_dir = sqlite_db_file_path.parent / "public" / "files"
        target_files_dir.mkdir(parents=True, exist_ok=True)

        file_stem = f"{task_id}_rife_interpolated"
        final_save_path_for_video = sm_get_unique_target_path(target_files_dir, file_stem, ".mp4")
        final_save_path_for_video.parent.mkdir(parents=True, exist_ok=True)
        db_output_location_for_rife = f"files/{final_save_path_for_video.name}"
        dprint(f"[Task ID: {task_id}] SQLite mode: RIFE video will be saved to {final_save_path_for_video} (DB: {db_output_location_for_rife})")
    else:
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

    dprint(f"[Task ID: {task_id}] Checking input image paths.")
    if not input_image1_path.is_file():
        print(f"[ERROR Task ID: {task_id}] Input image 1 not found: {input_image1_path}")
        return False, f"Input image 1 not found: {input_image1_path}"
    if not input_image2_path.is_file():
        print(f"[ERROR Task ID: {task_id}] Input image 2 not found: {input_image2_path}")
        return False, f"Input image 2 not found: {input_image2_path}"
    dprint(f"[Task ID: {task_id}] Input images found.")

    temp_output_dir = tempfile.mkdtemp(prefix=f"wgp_rife_{task_id}_")
    original_wgp_save_path = wgp_mod.save_path
    wgp_mod.save_path = str(temp_output_dir)

    try:
        pil_image_start = Image.open(input_image1_path).convert("RGB")
        pil_image_end = Image.open(input_image2_path).convert("RGB")

        print(f"[Task ID: {task_id}] Starting RIFE interpolation via video_utils.")
        dprint(f"  Input 1: {input_image1_path}")
        dprint(f"  Input 2: {input_image2_path}")

        rife_success = sm_rife_interpolate_images_to_video(
            image1=pil_image_start,
            image2=pil_image_end,
            num_frames=int(num_rife_frames),
            resolution_wh=sm_parse_resolution(resolution_str),
            output_path=final_save_path_for_video,
            fps=16,
            dprint_func=lambda msg: dprint(f"[Task ID: {task_id}] (rife_util) {msg}")
        )

        if rife_success:
            if final_save_path_for_video.exists() and final_save_path_for_video.stat().st_size > 0:
                generation_success = True
                output_location_to_db = db_output_location_for_rife
                print(f"[Task ID: {task_id}] RIFE video saved to: {final_save_path_for_video.resolve()} (DB: {output_location_to_db})")
            else:
                print(f"[ERROR Task ID: {task_id}] RIFE utility reported success, but output file is missing or empty: {final_save_path_for_video}")
                generation_success = False
        else:
            print(f"[ERROR Task ID: {task_id}] RIFE interpolation using video_utils failed.")
            generation_success = False

    except Exception as e:
        print(f"[ERROR Task ID: {task_id}] Overall _handle_rife_interpolate_task failed: {e}")
        traceback.print_exc()
        generation_success = False
    finally:
        wgp_mod.save_path = original_wgp_save_path

    try:
        shutil.rmtree(temp_output_dir)
        dprint(f"[Task ID: {task_id}] Cleaned up temporary directory: {temp_output_dir}")
    except Exception as e_clean:
        print(f"[WARNING Task ID: {task_id}] Failed to clean up temporary directory {temp_output_dir}: {e_clean}")


    return generation_success, output_location_to_db 

def handle_ltxv_upscale_task(task_params_dict: dict, main_output_dir_base: Path, task_id: str, *, dprint) -> tuple[bool, str]:
    """
    Handle LTXV upscaling task using only the necessary components (VAE + spatial upsampler).
    Optimized to avoid loading the full 13B transformer which isn't needed for upscaling.
    """
    try:
        from source.video_utils import extract_frames_from_video, create_video_from_frames_list
        from source.common_utils import sm_get_unique_target_path
        from source import db_operations as db_ops
        import torch
        
        # Import wgp module to access model downloading functions
        import sys
        import os
        wan2gp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Wan2GP")
        if wan2gp_path not in sys.path:
            sys.path.append(wan2gp_path)
        from Wan2GP import wgp as wgp_mod
        
        dprint(f"LTXV Upscale Task {task_id}: Starting optimized video upscaling (VAE + spatial upsampler only)")
        dprint(f"LTXV Upscale Task {task_id}: Input video: {task_params_dict.get('video_source_path')}")
        dprint(f"LTXV Upscale Task {task_id}: Upscale factor: {task_params_dict.get('upscale_factor')}")

        input_video_path = task_params_dict.get("video_source_path")
        upscale_factor = task_params_dict.get("upscale_factor", 2.0)
        output_path = task_params_dict.get("output_path")

        if not input_video_path or not Path(input_video_path).exists():
            return False, f"Input video not found: {input_video_path}"

        if upscale_factor <= 1.0:
            return False, f"Invalid upscale_factor: {upscale_factor}"

        # Download only the required components for upscaling
        dprint(f"LTXV Upscale Task {task_id}: Downloading required upscaling components...")
        required_files = [
            "ltxv_0.9.7_VAE.safetensors",
            "ltxv_0.9.7_spatial_upscaler.safetensors"
        ]
        for file in required_files:
            wgp_mod.download_models(file)
        
        # Load only VAE and spatial upsampler (no transformer needed for upscaling)
        dprint(f"LTXV Upscale Task {task_id}: Loading VAE and spatial upsampler components directly...")
        
        # Import LTXV components directly
        from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
        from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler
        
        # Load VAE directly
        vae_dtype = torch.float16 if wgp_mod.server_config.get("vae_precision", "16") == "16" else torch.float32
        dprint(f"LTXV Upscale Task {task_id}: Loading VAE with dtype {vae_dtype}")
        
        vae = CausalVideoAutoencoder.from_pretrained("ckpts/ltxv_0.9.7_VAE.safetensors")
        vae = vae.to(dtype=vae_dtype).eval()
        
        # Load spatial upsampler directly
        dprint(f"LTXV Upscale Task {task_id}: Loading spatial upsampler")
        latent_upsampler = LatentUpsampler.from_pretrained("ckpts/ltxv_0.9.7_spatial_upsampler.safetensors")
        latent_upsampler = latent_upsampler.to(dtype=vae_dtype).eval()
        
        dprint(f"LTXV Upscale Task {task_id}: Successfully loaded VAE and spatial upsampler (skipped 13B transformer)")
        
        # Extract frames from input video
        dprint(f"LTXV Upscale Task {task_id}: Extracting frames from input video...")
        input_frames = extract_frames_from_video(input_video_path)
        if not input_frames:
            return False, f"Failed to extract frames from {input_video_path}"
        
        dprint(f"LTXV Upscale Task {task_id}: Extracted {len(input_frames)} frames")
        
        # Process frames in batches for memory efficiency
        import torch
        import torchvision.transforms as transforms
        
        device = next(vae.parameters()).device
        upscaled_frames = []
        batch_size = 4  # Process 4 frames at a time

        # Get original frame dimensions and calculate VAE-compatible size (multiple of 32)
        first_frame = input_frames[0]
        # Assuming extract_frames_from_video returns numpy arrays (H, W, C)
        original_h, original_w, _ = first_frame.shape
        # The LTXV VAE requires dimensions to be divisible by 2 at each of its
        # downsampling stages. The total downsampling requires divisibility by 32.
        new_h = (original_h // 32) * 32
        new_w = (original_w // 32) * 32
        dprint(f"LTXV Upscale Task {task_id}: Resizing input frames from {original_w}x{original_h} to {new_w}x{new_h} for VAE compatibility")
        
        # This transform pipeline expects PIL Images
        transform = transforms.Compose([
            transforms.Resize((new_h, new_w), antialias=True), # Ensure dimensions are VAE-compatible
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
        
        dprint(f"LTXV Upscale Task {task_id}: Processing frames...")
        for i in range(0, len(input_frames), batch_size):
            batch_frames = input_frames[i:i + batch_size]
            dprint(f"LTXV Upscale Task {task_id}: Processing batch {i//batch_size + 1}/{(len(input_frames) + batch_size - 1)//batch_size}")
            
            # Convert frames to tensor batch
            batch_tensors = []
            for frame in batch_frames:
                # Convert numpy array to PIL Image before applying transforms
                pil_frame = Image.fromarray(frame)
                frame_tensor = transform(pil_frame).unsqueeze(0)  # Add batch dimension
                batch_tensors.append(frame_tensor)
            
            # Use the actual dtype of the VAE model from the pipe, not the calculated one.
            # This handles cases where ltx_video internally forces a dtype (e.g. bfloat16).
            batch_tensor = torch.cat(batch_tensors, dim=0).to(device, dtype=vae.dtype)
            
            # Add temporal dimension for LTXV VAE (expects 5D: B, C, T, H, W)
            batch_tensor_5d = batch_tensor.unsqueeze(2)  # Add time dimension
            
            with torch.no_grad():
                # Encode to latent space
                latents = vae.encode(batch_tensor_5d).latent_dist.sample()
                
                # Use wgp.py's upscaling approach with proper normalization
                # Import the required functions from ltx_video
                try:
                    from ltx_video.models.autoencoders.vae_encode import normalize_latents, un_normalize_latents
                    from ltx_video.pipelines.pipeline_ltx_video import adain_filter_latent
                    
                    # Store original latents for AdaIN filtering
                    original_latents = latents.clone()
                    
                    # Un-normalize latents before upscaling (like wgp.py does)
                    latents = un_normalize_latents(latents, vae, vae_per_channel_normalize=True)
                    
                    # Upscale latents
                    upscaled_latents = latent_upsampler(latents)
                    
                    # Re-normalize latents after upscaling
                    upscaled_latents = normalize_latents(upscaled_latents, vae, vae_per_channel_normalize=True)
                    
                    # Apply AdaIN filtering for consistency (like wgp.py does)
                    upscaled_latents = adain_filter_latent(latents=upscaled_latents, reference_latents=original_latents)
                    
                except ImportError:
                    # Fallback to our original approach if the functions aren't available
                    dprint(f"LTXV Upscale Task {task_id}: Using fallback upscaling method (missing ltx_video utils)")
                    upscaled_latents = latent_upsampler(latents)
                
                # Calculate the target shape for decoding
                # The upsampler doubles the latent dimensions, so the pixel dimensions will also be doubled.
                upscaled_h = new_h * 2
                upscaled_w = new_w * 2
                target_shape = (batch_tensor.shape[0], batch_tensor.shape[1], 1, upscaled_h, upscaled_w)

                # Create a timestep tensor for the batch, as required by the VAE decoder for batch processing.
                timestep_tensor = torch.zeros(batch_tensor.shape[0], device=device, dtype=torch.long)
                
                # Decode back to pixels, providing the target shape and a timestep tensor for the batch.
                upscaled_batch_5d = vae.decode(upscaled_latents, target_shape=target_shape, timestep=timestep_tensor).sample
                
                # Remove temporal dimension and convert back to images
                upscaled_batch = upscaled_batch_5d.squeeze(2)  # Remove time dimension
                
                # Convert tensors back to PIL images
                for j in range(upscaled_batch.shape[0]):
                    frame_tensor = upscaled_batch[j]
                    # Denormalize from [-1, 1] to [0, 1]
                    frame_tensor = (frame_tensor + 1.0) / 2.0
                    frame_tensor = torch.clamp(frame_tensor, 0.0, 1.0)
                    
                    # Convert to float32 before converting to PIL, as bfloat16 is not supported by ToPILImage
                    frame_pil = transforms.ToPILImage()(frame_tensor.cpu().float())
                    upscaled_frames.append(frame_pil)
        
        if not upscaled_frames:
            return False, "No frames were upscaled"
        
        # Determine output resolution from upscaled frames
        output_width, output_height = upscaled_frames[0].size
        dprint(f"LTXV Upscale Task {task_id}: Output resolution: {output_width}x{output_height}")
        
        # Get original video FPS
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        
        # Determine output path
        if db_ops.DB_TYPE == "sqlite" and db_ops.SQLITE_DB_PATH:
            sqlite_db_parent = Path(db_ops.SQLITE_DB_PATH).resolve().parent
            target_dir = sqlite_db_parent / "public" / "files"
            target_dir.mkdir(parents=True, exist_ok=True)
            output_video_path = sm_get_unique_target_path(target_dir, f"{task_id}_upscaled", ".mp4")
            output_location = f"files/{output_video_path.name}"
        else:
            if output_path:
                output_video_path = Path(output_path)
                output_video_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_video_path = main_output_dir_base / f"{task_id}_upscaled.mp4"
            output_location = str(output_video_path.resolve())
        
        # Create output video from upscaled frames
        dprint(f"LTXV Upscale Task {task_id}: Creating output video...")
        
        # Convert PIL Images to NumPy arrays in BGR format for video creation
        upscaled_frames_np = []
        for pil_frame in upscaled_frames:
            # Convert PIL to RGB numpy array
            rgb_array = np.array(pil_frame)
            # Convert RGB to BGR for OpenCV/video creation
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            upscaled_frames_np.append(bgr_array)
        
        output_video_obj = create_video_from_frames_list(
            upscaled_frames_np,  # Now using NumPy BGR arrays
            output_video_path, 
            fps, 
            (output_width, output_height)
        )
        
        if output_video_obj and output_video_obj.exists():
            dprint(f"LTXV Upscale Task {task_id}: Successfully created upscaled video: {output_video_path}")
            return True, output_location
        else:
            return False, f"Failed to create output video at {output_video_path}"
            
    except Exception as e:
        dprint(f"LTXV Upscale Task {task_id}: Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False, f"LTXV upscaling failed: {e}" 