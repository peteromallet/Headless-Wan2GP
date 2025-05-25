"""Different-pose task handler."""

import json
import shutil
import traceback
from pathlib import Path

import cv2 # pip install opencv-python
from PIL import Image # pip install Pillow

# Import from the new common_utils module
from .common_utils import (
    DEBUG_MODE, dprint, generate_unique_task_id, add_task_to_db, poll_task_status,
    save_frame_from_video, create_pose_interpolated_guide_video,
    generate_different_pose_debug_video_summary # For debug mode
)

def run_different_pose_task(task_args, common_args, parsed_resolution, main_output_dir, db_file_path):
    print("--- Running Task: Different Pose (Modified Workflow: User Input + OpenPose T2I) ---") 
    dprint(f"Task Args: {task_args}")
    dprint(f"Common Args: {common_args}")

    num_target_frames = common_args.output_video_frames
    if num_target_frames < 2: 
        print(f"Error: --output_video_frames (set to {num_target_frames}) must be at least 2.") 
        return 1

    user_input_image_path = Path(task_args.input_image).resolve()
    if not user_input_image_path.exists():
        print(f"Error: Input image not found: {user_input_image_path}")
        return 1

    task_run_id = generate_unique_task_id("sm_pose_rife_")
    task_work_dir = main_output_dir / f"different_pose_run_{task_run_id}"
    task_work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Working directory for this 'different_pose' run: {task_work_dir.resolve()}")

    debug_video_stages_data = [] 
    default_image_display_frames = common_args.fps_helpers * 2 

    if DEBUG_MODE:
        debug_video_stages_data.append({
            'label': f"0. User Input ({user_input_image_path.name})", 
            'type': 'image',
            'path': str(user_input_image_path),
            'display_frames': default_image_display_frames
        })

    # --- Step 1: Generate OpenPose from User Input Image ---
    print("\nStep 1: Generating OpenPose from user input image...")
    openpose_user_input_task_id = generate_unique_task_id("pose_op_user_")
    openpose_user_input_output_path = task_work_dir / f"{openpose_user_input_task_id}_openpose_from_user.png"

    openpose_user_input_payload = {
        "task_id": openpose_user_input_task_id,
        "task_type": "generate_openpose",
        "model": "core_pose_processor", 
        "input_image_path": str(user_input_image_path.resolve()),
        "output_path": str(openpose_user_input_output_path.resolve())
    }
    dprint(f"OpenPose (from user input) generation payload: {json.dumps(openpose_user_input_payload, indent=2)}")
    add_task_to_db(openpose_user_input_payload, db_file_path)
    openpose_user_input_path_str = poll_task_status(openpose_user_input_task_id, db_file_path, common_args.poll_interval, common_args.poll_timeout)

    if not openpose_user_input_path_str:
        print("Failed to generate OpenPose from user input image.")
        return 1
    openpose_user_input_path = Path(openpose_user_input_path_str)
    print(f"Successfully generated OpenPose from user input image: {openpose_user_input_path}")
    if DEBUG_MODE:
        debug_video_stages_data.append({
            'label': f"1. OpenPose from User Input ({openpose_user_input_path.name})", 
            'type': 'image',
            'path': str(openpose_user_input_path),
            'display_frames': default_image_display_frames
        })

    # --- Step 2: Initial Image Generation (Text-to-Image for target pose reference) ---
    print("\nStep 2: Generating target image from prompt (Text-to-Image)...")
    t2i_target_task_id = generate_unique_task_id("pose_t2i_target_")
    t2i_target_guide_video_path = task_work_dir / f"{t2i_target_task_id}_guide.mp4"
    t2i_target_headless_output_path = task_work_dir / f"{t2i_target_task_id}_video_raw.mp4"
    generated_image_for_target_pose_path = task_work_dir / f"{t2i_target_task_id}_generated_from_prompt.png"
    
    temp_neutral_guide_image_path_s2 = task_work_dir / "_temp_neutral_guide_frame_s2.png"
    try:
        neutral_pil_image_s2 = Image.new('RGB', (parsed_resolution[0] // 16, parsed_resolution[1] // 16), (128, 128, 128))
        neutral_pil_image_s2.save(temp_neutral_guide_image_path_s2)
        create_pose_interpolated_guide_video(
            output_video_path=t2i_target_guide_video_path,
            resolution=parsed_resolution,
            total_frames=1, # Single frame for T2I guide
            start_image_path=temp_neutral_guide_image_path_s2, # Use neutral image for both
            end_image_path=temp_neutral_guide_image_path_s2,
            fps=common_args.fps_helpers,
            confidence_threshold=0.01, 
            include_face=False, 
            include_hands=False
        )
    except Exception as e:
        print(f"Error creating neutral guide video for T2I step: {e}")
        if temp_neutral_guide_image_path_s2.exists(): temp_neutral_guide_image_path_s2.unlink(missing_ok=True)
        return 1
    finally:
        if temp_neutral_guide_image_path_s2.exists(): temp_neutral_guide_image_path_s2.unlink(missing_ok=True)

    t2i_target_payload = {
        "task_id": t2i_target_task_id,
        "prompt": task_args.prompt,
        "model": common_args.model_name, 
        "resolution": common_args.resolution,
        "frames": 1, 
        "seed": common_args.seed, 
        "video_guide_path": str(t2i_target_guide_video_path.resolve()),
        "output_path": str(t2i_target_headless_output_path.resolve())
    }
    if common_args.use_causvid_lora: t2i_target_payload["use_causvid_lora"] = True    
    dprint(f"Added lora_name: 'jump' to T2I target payload.") 
    dprint(f"T2I for target pose payload: {json.dumps(t2i_target_payload, indent=2)}")
    add_task_to_db(t2i_target_payload, db_file_path)
    raw_video_from_headless_s2 = poll_task_status(t2i_target_task_id, db_file_path, common_args.poll_interval, common_args.poll_timeout)

    if not raw_video_from_headless_s2:
        print("Failed to generate target image from prompt.")
        return 1
    t2i_video_output_path_s2 = Path(raw_video_from_headless_s2)
    if not save_frame_from_video(t2i_video_output_path_s2, 0, generated_image_for_target_pose_path, parsed_resolution):
        print(f"Failed to extract frame from target T2I video: {t2i_video_output_path_s2}")
        return 1
    print(f"Successfully generated target image from prompt: {generated_image_for_target_pose_path}")
    if DEBUG_MODE:
        debug_video_stages_data.append({
            'label': f"2. Target T2I Image ({generated_image_for_target_pose_path.name})", 
            'type': 'image',
            'path': str(generated_image_for_target_pose_path),
            'display_frames': default_image_display_frames
        })

    # --- Step 3: Generate OpenPose from the Target T2I Image ---
    print("\nStep 3: Generating OpenPose from the target text-generated image...")
    openpose_t2i_task_id = generate_unique_task_id("pose_op_t2i_")
    openpose_t2i_output_path = task_work_dir / f"{openpose_t2i_task_id}_openpose_from_t2i.png"

    openpose_t2i_payload = {
        "task_id": openpose_t2i_task_id,
        "task_type": "generate_openpose",
        "model": "core_pose_processor", 
        "input_image_path": str(generated_image_for_target_pose_path.resolve()), 
        "output_path": str(openpose_t2i_output_path.resolve())
    }
    dprint(f"OpenPose (from T2I) generation payload: {json.dumps(openpose_t2i_payload, indent=2)}")
    add_task_to_db(openpose_t2i_payload, db_file_path)
    openpose_t2i_path_str = poll_task_status(openpose_t2i_task_id, db_file_path, common_args.poll_interval, common_args.poll_timeout)

    if not openpose_t2i_path_str:
        print("Failed to generate OpenPose from T2I image.")
        return 1
    openpose_t2i_path = Path(openpose_t2i_path_str)
    print(f"Successfully generated OpenPose from T2I image: {openpose_t2i_path}")
    if DEBUG_MODE:
        debug_video_stages_data.append({
            'label': f"3. OpenPose from T2I ({openpose_t2i_path.name})", 
            'type': 'image',
            'path': str(openpose_t2i_path),
            'display_frames': default_image_display_frames
        })

    print("\nStep 4: RIFE Interpolation SKIPPED as per modification.")

    # --- Step 5: Create Final Composite Guide Video ---
    print("\nStep 5: Creating custom guide video (OpenPose T2I only)...")
    custom_guide_video_id = generate_unique_task_id("pose_custom_guide_")
    custom_guide_video_path = task_work_dir / f"{custom_guide_video_id}_custom_guide.mp4"
    
    try:
        create_pose_interpolated_guide_video(
            output_video_path=custom_guide_video_path,
            resolution=parsed_resolution,
            total_frames=num_target_frames, 
            start_image_path=user_input_image_path, 
            end_image_path=generated_image_for_target_pose_path, 
            fps=common_args.fps_helpers,
            confidence_threshold=0.1, 
            include_face=True,
            include_hands=True
        )
        print(f"Successfully created pose-interpolated guide video: {custom_guide_video_path}")
    except Exception as e_custom_guide:
        print(f"Error creating custom guide video: {e_custom_guide}")
        traceback.print_exc()
        return 1

    final_video_guide_path = custom_guide_video_path 

    if DEBUG_MODE: 
        debug_video_stages_data.append({
            'label': f"4. Custom Guide Video ({custom_guide_video_path.name})", 
            'type': 'video',
            'path': str(custom_guide_video_path.resolve()),
            'display_frames': common_args.fps_helpers * (num_target_frames // common_args.fps_helpers + 1)
        })
    
    # --- Step 6: Final Video Generation ---
    print("\nStep 6: Generating final video using custom guide (OpenPose T2I only) and user input reference...")
    final_video_task_id = generate_unique_task_id("pose_finalvid_custom_") 
    final_video_headless_output_path = task_work_dir / f"{final_video_task_id}_video_raw.mp4"

    final_video_payload = {
        "task_id": final_video_task_id,
        "prompt": task_args.prompt, 
        "model": common_args.model_name,
        "resolution": common_args.resolution,
        "frames": num_target_frames, 
        "seed": common_args.seed + 1, 
        "video_guide_path": str(final_video_guide_path.resolve()), 
        "reference_image_path": str(user_input_image_path.resolve()), 
        "image_refs_paths": [str(user_input_image_path.resolve())], 
        "image_prompt_type": "IV", 
        "output_path": str(final_video_headless_output_path.resolve())
    }

    if common_args.use_causvid_lora: final_video_payload["use_causvid_lora"] = True    
    dprint(f"Added lora_name: 'jump' to final video payload.") 
    dprint(f"Final video generation payload (with OpenPose T2I only guide and user input reference): {json.dumps(final_video_payload, indent=2)}")
    add_task_to_db(final_video_payload, db_file_path)
    raw_final_video_from_headless = poll_task_status(final_video_task_id, db_file_path, common_args.poll_interval, common_args.poll_timeout)

    if not raw_final_video_from_headless:
        print("Failed to generate final video using custom guide and reference.")
        return 1
    final_posed_video_path = Path(raw_final_video_from_headless)
    print(f"Successfully generated final video with custom guide and reference: {final_posed_video_path}")

    print("\nTrimming logic for final video SKIPPED as per modification.")

    if DEBUG_MODE:
         debug_video_stages_data.append({
            'label': f"5. Final Video with Custom Guide ({final_posed_video_path.name})", 
            'type': 'video',
            'path': str(final_posed_video_path),
            'display_frames': common_args.fps_helpers * (num_target_frames // common_args.fps_helpers +1) 
        })

    # --- Step 7: Result Extraction ---
    print(f"\nStep 7: Extracting the final posed image (last frame of {num_target_frames}-frame video)...")
    final_posed_image_output_path = main_output_dir / f"final_posed_image_{task_run_id}.png"

    cap_tmp_s7 = cv2.VideoCapture(str(final_posed_video_path))
    total_frames_generated_s7 = int(cap_tmp_s7.get(cv2.CAP_PROP_FRAME_COUNT)) if cap_tmp_s7.isOpened() else 0
    if cap_tmp_s7.isOpened(): cap_tmp_s7.release()

    if total_frames_generated_s7 == 0:
        print(f"Error: Final generated video is empty: {final_posed_video_path}")
        return 1
    target_frame_index_s7 = num_target_frames - 1 
    if target_frame_index_s7 >= total_frames_generated_s7 : 
        print(f"Warning: Target frame index {target_frame_index_s7} is out of bounds for video with {total_frames_generated_s7} frames. Using last available frame.")
        target_frame_index_s7 = max(0, total_frames_generated_s7 - 1)

    if not save_frame_from_video(final_posed_video_path, target_frame_index_s7, final_posed_image_output_path, parsed_resolution):
        print(f"Failed to extract final posed image from custom-guided video with reference: {final_posed_video_path}")
        return 1
    
    print(f"\nSuccessfully completed 'different_pose' task with custom (OpenPose T2I only guide + user input reference) workflow!")
    print(f"Final posed image saved to: {final_posed_image_output_path.resolve()}")

    if DEBUG_MODE:
        debug_video_stages_data.append({
            'label': f"6. Final Extracted Image ({final_posed_image_output_path.name})", 
            'type': 'image',
            'path': str(final_posed_image_output_path),
            'display_frames': default_image_display_frames
        })
        if debug_video_stages_data:
            video_collage_file_name_s7 = f"debug_video_summary_pose_rife_{task_run_id}.mp4"
            video_collage_output_path_s7 = main_output_dir / video_collage_file_name_s7
            generate_different_pose_debug_video_summary(
                debug_video_stages_data, video_collage_output_path_s7, 
                fps=common_args.fps_helpers, target_resolution=parsed_resolution
            )

    if not common_args.skip_cleanup and not DEBUG_MODE:
        print(f"\nCleaning up intermediate files in {task_work_dir}...")
        try:
            shutil.rmtree(task_work_dir)
            print(f"Removed intermediate directory: {task_work_dir}")
        except OSError as e_clean:
            print(f"Error removing intermediate directory {task_work_dir}: {e_clean}")
    else:
        print(f"Skipping cleanup of intermediate files in {task_work_dir}.")

    return 0 