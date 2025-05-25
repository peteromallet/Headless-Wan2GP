"""Travel-between-images task handler."""

import json
import shutil
import traceback
from pathlib import Path

import cv2 # pip install opencv-python
# from PIL import Image # Not directly used in this task handler, but could be if image ops were added

# Import from the new common_utils module
from .common_utils import (
    DEBUG_MODE, dprint, generate_unique_task_id, add_task_to_db, poll_task_status,
    extract_video_segment_ffmpeg, stitch_videos_ffmpeg,
    create_pose_interpolated_guide_video, # Used for guide video generation
    generate_debug_summary_video # For debug mode
)

def run_travel_between_images_task(task_args, common_args, parsed_resolution, main_output_dir, db_file_path):
    print("--- Running Task: Travel Between Images ---")
    dprint(f"Task Args: {task_args}")
    dprint(f"Common Args: {common_args}")

    all_segments_debug_data = [] # For debug collage

    if len(task_args.input_images) < 2:
        print("Error: At least two input images are required for 'travel_between_images' task.")
        return 1  # Indicate error
    if len(task_args.base_prompts) != len(task_args.input_images) - 1:
        print(f"Error: Number of prompts ({len(task_args.base_prompts)}) must be one less than "
              f"the number of input images ({len(task_args.input_images)}) for 'travel_between_images' task.")
        return 1
    
    if task_args.segment_frames <= 0:
        print("Error: --segment_frames must be positive.")
        return 1
    if task_args.frame_overlap < 0:
        print("Error: --frame_overlap cannot be negative.")
        return 1
    
    if len(task_args.input_images) > 2 and task_args.frame_overlap >= task_args.segment_frames:
         print("Error: --segment_frames must be strictly greater than --frame_overlap when generating multiple overlapping segments.")
         return 1
    if task_args.frame_overlap > 0 and task_args.segment_frames <= task_args.frame_overlap and len(task_args.input_images) > 1:
        print(f"Warning: --frame_overlap ({task_args.frame_overlap}) is >= --segment_frames ({task_args.segment_frames}). "
              "For subsequent segments, the portion extracted for stitching might be very short or empty.")

    generated_segments_for_stitching = [] 
    # previous_segment_headless_output_video = None # Not needed here, create_pose_interpolated_guide_video handles its own logic
    num_segments_to_generate = len(task_args.input_images) - 1

    for i in range(num_segments_to_generate):
        print(f"\n--- Processing Segment {i+1} / {num_segments_to_generate} (Travel Task) ---")
        # dprint(f"Starting segment {i+1}. Previous headless output: {previous_segment_headless_output_video}") # Not relevant with current guide gen
        
        task_id = generate_unique_task_id(f"sm_travel_seg{i:02d}_")
        segment_work_dir = main_output_dir / f"segment_travel_{i:02d}_{task_id}"
        segment_work_dir.mkdir(parents=True, exist_ok=True)

        current_segment_start_anchor_image = Path(task_args.input_images[i]).resolve()
        current_segment_end_anchor_image = Path(task_args.input_images[i+1]).resolve()
        current_prompt = task_args.base_prompts[i]

        if not current_segment_start_anchor_image.exists():
            print(f"Error: Start anchor image not found: {current_segment_start_anchor_image} for segment {i+1}. Skipping segment.")
            continue
        if not current_segment_end_anchor_image.exists():
            print(f"Error: End anchor image not found: {current_segment_end_anchor_image} for segment {i+1}. Skipping segment.")
            continue

        guide_video_path = segment_work_dir / f"{task_id}_guide.mp4"

        try:
            print(f"Creating guide video: {guide_video_path}")
            create_pose_interpolated_guide_video(
                output_video_path=guide_video_path,
                resolution=parsed_resolution,
                total_frames=task_args.segment_frames,
                start_image_path=current_segment_start_anchor_image, 
                end_image_path=current_segment_end_anchor_image,     
                fps=common_args.fps_helpers,
                confidence_threshold=0.1, # Default confidence from original steerable_motion
                include_face=True,        # Default include face from original steerable_motion
                include_hands=True        # Default include hands from original steerable_motion
            )
        except Exception as e_helpers:
            print(f"Error creating helper videos for segment {i+1} (Task ID: {task_id}): {e_helpers}. Skipping segment.")
            traceback.print_exc()
            continue
        
        task_payload = {
            "task_id": task_id,
            "prompt": current_prompt,
            "model": common_args.model_name,
            "resolution": common_args.resolution,
            "frames": task_args.segment_frames,
            "seed": common_args.seed + i,
            "video_guide_path": str(guide_video_path.resolve()),            
            "output_path": str((segment_work_dir / f"{task_id}.mp4").resolve())
        }

        if common_args.use_causvid_lora:
            task_payload["use_causvid_lora"] = True
            dprint(f"CausVid LoRA enabled for task {task_id}.")
        
        dprint(f"Task payload for headless.py: {json.dumps(task_payload, indent=4)}")
        
        try:
            add_task_to_db(task_payload, db_file_path)
        except Exception as e_db_add:
            print(f"Failed to add task {task_id} to DB: {e_db_add}. Skipping segment.")
            continue

        raw_output_video_from_headless = poll_task_status(task_id, db_file_path, common_args.poll_interval, common_args.poll_timeout)
        
        dprint(f"Raw output video from headless for task {task_id}: {raw_output_video_from_headless}")
        
        if not raw_output_video_from_headless:
            print(f"Task {task_id} for segment {i+1} failed or timed out. Skipping.")
            # dprint(f"Task {task_id} poll returned None. previous_segment_headless_output_video remains: {previous_segment_headless_output_video}") # Not relevant
            continue 
        
        current_headless_output_video_path = Path(raw_output_video_from_headless)
        if not current_headless_output_video_path.exists() or current_headless_output_video_path.stat().st_size == 0:
            print(f"Error: Headless output video {current_headless_output_video_path} is missing or empty for task {task_id}. Skipping segment.")
            continue

        processed_segment_for_stitch_path = segment_work_dir / f"{task_id}_processed_stitch.mp4"

        if DEBUG_MODE and current_headless_output_video_path and current_headless_output_video_path.exists():
            segment_debug_info = {
                "segment_index": i,
                "task_id": task_id,
                "guide_video_path": str(guide_video_path.resolve()),
                "raw_headless_output_path": str(current_headless_output_video_path.resolve()),
                "task_payload": task_payload 
            }
            all_segments_debug_data.append(segment_debug_info)
            dprint(f"Collected debug info for segment {i}: {task_id}")

        source_fps_headless = float(common_args.fps_helpers) 
        num_frames_in_headless_output = task_args.segment_frames 
        try:
            temp_cap = cv2.VideoCapture(str(current_headless_output_video_path))
            if temp_cap.isOpened():
                s_fps = temp_cap.get(cv2.CAP_PROP_FPS)
                actual_frame_count = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if s_fps is not None and s_fps > 0: source_fps_headless = float(s_fps)
                if actual_frame_count > 0: num_frames_in_headless_output = actual_frame_count
                temp_cap.release()
            else:
                 print(f"Warning: Could not open {current_headless_output_video_path} to get FPS/frame count. Using defaults.")
        except Exception as e_fps_read:
            print(f"Warning: Could not determine FPS/frame count of {current_headless_output_video_path} ({e_fps_read}). Using defaults.")

        start_extraction_idx_for_stitch = 0
        is_last_segment = (i == num_segments_to_generate - 1)

        if not is_last_segment and task_args.frame_overlap > 0:
            if num_frames_in_headless_output > task_args.frame_overlap:
                num_frames_to_keep_for_stitch = num_frames_in_headless_output - task_args.frame_overlap
            else:
                actual_frames_kept = max(0, num_frames_in_headless_output - task_args.frame_overlap)
                print(f"Warning: Headless output for segment {i+1} (Task {task_id}) has {num_frames_in_headless_output} frames. "
                      f"It is not the last segment, and its end is meant to be trimmed by {task_args.frame_overlap} overlap frames. "
                      f"This segment will contribute {actual_frames_kept} frames.")
                num_frames_to_keep_for_stitch = actual_frames_kept
        else:
            num_frames_to_keep_for_stitch = num_frames_in_headless_output
        
        if num_frames_to_keep_for_stitch <= 0:
             print(f"Segment {i+1} (Task {task_id}) results in {num_frames_to_keep_for_stitch} frames after overlap processing. Not adding to final video.")
        else:
            print(f"Processing output from headless.py ({current_headless_output_video_path}) for stitching.")
            dprint(f"  Calling segment extraction: Input: {current_headless_output_video_path}, Output: {processed_segment_for_stitch_path}, Start Index: {start_extraction_idx_for_stitch}, Num Frames: {num_frames_to_keep_for_stitch}, FPS: {source_fps_headless}")
            extract_video_segment_ffmpeg(current_headless_output_video_path, processed_segment_for_stitch_path,
                                     start_extraction_idx_for_stitch, num_frames_to_keep_for_stitch,
                                     input_fps=source_fps_headless, resolution=parsed_resolution)

            if processed_segment_for_stitch_path.exists() and processed_segment_for_stitch_path.stat().st_size > 0:
                 generated_segments_for_stitching.append(str(processed_segment_for_stitch_path.resolve()))
                 dprint(f"Added '{processed_segment_for_stitch_path.name}' to stitch list.")
            else:
                 print(f"Warning: Processed segment {processed_segment_for_stitch_path} is empty or missing after extraction. Not adding to stitch list.")
                 dprint(f"Segment '{processed_segment_for_stitch_path.name}' was not added to stitch list (missing or empty).")
        
        # previous_segment_headless_output_video = current_headless_output_video_path # Not needed for this guide creation logic
        print(f"Segment {i+1} (Task ID: {task_id}) processing complete.")

    if not generated_segments_for_stitching:
        print("\nNo video segments were successfully generated/processed for stitching.")
    else:
        final_video_output_path = main_output_dir / "final_travel_video.mp4"
        print(f"\nStitching {len(generated_segments_for_stitching)} segments into {final_video_output_path}...")
        try:
            stitch_videos_ffmpeg(generated_segments_for_stitching, final_video_output_path)
            print(f"Final video successfully created: {final_video_output_path.resolve()}")
        except Exception as e_stitch:
            print(f"Error during final video stitching: {e_stitch}")
            dprint(f"Exception during stitching: {traceback.format_exc()}")
            print("You may find processed segments in the subdirectories of:", main_output_dir)
            for p_stitch_fail in generated_segments_for_stitching: print(f"- {p_stitch_fail}")
    
    if not common_args.skip_cleanup and not DEBUG_MODE: 
        print("\nCleaning up intermediate segment directories for 'travel_between_images' task...")
        cleaned_count = 0
        for item in main_output_dir.iterdir():
            if item.is_dir() and item.name.startswith("segment_travel_"):
                try:
                    shutil.rmtree(item)
                    print(f"Removed intermediate directory: {item}")
                    cleaned_count +=1
                except OSError as e_clean:
                    print(f"Error removing directory {item}: {e_clean}")
        if cleaned_count > 0: print(f"Cleaned up {cleaned_count} intermediate segment directories.")
        else: print("No intermediate segment directories found to clean for this task.")

    if DEBUG_MODE and all_segments_debug_data:
        collage_output_path = main_output_dir / "debug_travel_summary_collage.mp4"
        dprint(f"Generating debug summary collage video at: {collage_output_path}")
        try:
            generate_debug_summary_video(all_segments_debug_data, collage_output_path, 
                                         fps=common_args.fps_helpers, 
                                         num_frames_for_collage=task_args.segment_frames)
            print(f"Debug summary collage video created: {collage_output_path}")
        except Exception as e_collage:
            print(f"Error generating debug summary collage video: {e_collage}")
            dprint(f"Exception during debug collage generation: {traceback.format_exc()}")
    return 0 