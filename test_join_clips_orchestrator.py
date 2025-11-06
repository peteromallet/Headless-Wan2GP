#!/usr/bin/env python3
"""
Test script for join_clips_orchestrator

This script demonstrates how to submit a join_clips_orchestrator task
that takes multiple video clips and progressively joins them together.

Example usage:
    python test_join_clips_orchestrator.py --clip1 path/to/clip1.mp4 --clip2 path/to/clip2.mp4 --clip3 path/to/clip3.mp4
"""

import uuid
import argparse
from pathlib import Path
from source import db_operations as db_ops


def submit_join_clips_orchestrator_task(
    clip_paths: list[str],
    clip_names: list[str] | None = None,
    context_frame_count: int = 8,
    gap_frame_count: int = 53,
    replace_mode: bool = False,
    blend_frames: int = 3,
    prompt: str = "smooth cinematic transition",
    negative_prompt: str = "blurry, distorted, artifacts",
    model: str = "lightning_baseline_2_2_2",
    aspect_ratio: str | None = None,
    output_base_dir: str = "./outputs/join_clips_orchestrator/",
    project_id: str | None = None,
) -> str:
    """
    Submit a join_clips_orchestrator task to the database.

    Args:
        clip_paths: List of video file paths or URLs (must be at least 2)
        clip_names: Optional list of names for each clip (for logging)
        context_frame_count: Number of frames to extract from each clip boundary
        gap_frame_count: Number of frames to generate/replace at transitions
        replace_mode: If True, replace boundary frames instead of inserting
        blend_frames: Number of frames to use for crossfade blending
        prompt: Generation prompt for transitions
        negative_prompt: Negative prompt for generation
        model: Model to use for generation
        aspect_ratio: Optional aspect ratio to standardize all clips (e.g., "16:9")
        output_base_dir: Base directory for outputs
        project_id: Optional project ID for authorization

    Returns:
        Task ID of the submitted orchestrator task
    """
    if len(clip_paths) < 2:
        raise ValueError("At least 2 clips are required")

    # Generate clip names if not provided
    if clip_names is None:
        clip_names = [f"clip_{i+1}" for i in range(len(clip_paths))]
    elif len(clip_names) != len(clip_paths):
        raise ValueError("clip_names must have same length as clip_paths")

    # Create clip_list structure
    clip_list = [
        {"url": str(Path(path).resolve()), "name": name}
        for path, name in zip(clip_paths, clip_names)
    ]

    # Generate unique run_id
    run_id = f"run_{uuid.uuid4().hex[:8]}"

    # Build orchestrator payload
    orchestrator_payload = {
        "clip_list": clip_list,
        "context_frame_count": context_frame_count,
        "gap_frame_count": gap_frame_count,
        "replace_mode": replace_mode,
        "blend_frames": blend_frames,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "model": model,
        "regenerate_anchors": True,
        "num_anchor_frames": 3,
        "run_id": run_id,
        "output_base_dir": output_base_dir,
    }

    # Add optional parameters
    if aspect_ratio:
        orchestrator_payload["aspect_ratio"] = aspect_ratio

    if project_id:
        orchestrator_payload["project_id"] = project_id

    # Submit task to database
    print(f"[TEST] Submitting join_clips_orchestrator task")
    print(f"[TEST] Number of clips: {len(clip_list)}")
    print(f"[TEST] Number of joins to create: {len(clip_list) - 1}")
    print(f"[TEST] Run ID: {run_id}")
    print(f"[TEST] Clips: {[c['name'] for c in clip_list]}")

    task_id = db_ops.add_task_to_db(
        task_payload={"orchestrator_details": orchestrator_payload},
        task_type_str="join_clips_orchestrator",
        dependant_on=None
    )

    print(f"[TEST] ✓ Task submitted successfully!")
    print(f"[TEST] Task ID: {task_id}")
    print(f"[TEST] Monitor progress in worker logs or database")
    print(f"[TEST] Expected output location: {output_base_dir}join_clips_run_{run_id}/")

    return task_id


def main():
    parser = argparse.ArgumentParser(
        description="Test join_clips_orchestrator by submitting a multi-clip join task"
    )
    parser.add_argument(
        "--clip1",
        type=str,
        required=True,
        help="Path to first video clip"
    )
    parser.add_argument(
        "--clip2",
        type=str,
        required=True,
        help="Path to second video clip"
    )
    parser.add_argument(
        "--clip3",
        type=str,
        help="Path to third video clip (optional)"
    )
    parser.add_argument(
        "--clip4",
        type=str,
        help="Path to fourth video clip (optional)"
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=8,
        help="Number of context frames from each clip boundary (default: 8)"
    )
    parser.add_argument(
        "--gap-frames",
        type=int,
        default=53,
        help="Number of transition frames to generate (default: 53)"
    )
    parser.add_argument(
        "--replace-mode",
        action="store_true",
        help="Use replace mode (replace boundary frames instead of inserting)"
    )
    parser.add_argument(
        "--blend-frames",
        type=int,
        default=3,
        help="Number of frames for crossfade blending (default: 3)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="smooth cinematic transition",
        help="Generation prompt for transitions"
    )
    parser.add_argument(
        "--aspect-ratio",
        type=str,
        help="Aspect ratio to standardize clips (e.g., '16:9', '9:16', '1:1')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/join_clips_orchestrator/",
        help="Base output directory"
    )
    parser.add_argument(
        "--project-id",
        type=str,
        help="Optional project ID for authorization"
    )

    args = parser.parse_args()

    # Collect clip paths
    clip_paths = [args.clip1, args.clip2]
    if args.clip3:
        clip_paths.append(args.clip3)
    if args.clip4:
        clip_paths.append(args.clip4)

    # Verify clips exist
    for i, clip_path in enumerate(clip_paths, 1):
        if not Path(clip_path).exists():
            print(f"[ERROR] Clip {i} not found: {clip_path}")
            return 1

    # Submit task
    try:
        task_id = submit_join_clips_orchestrator_task(
            clip_paths=clip_paths,
            context_frame_count=args.context_frames,
            gap_frame_count=args.gap_frames,
            replace_mode=args.replace_mode,
            blend_frames=args.blend_frames,
            prompt=args.prompt,
            aspect_ratio=args.aspect_ratio,
            output_base_dir=args.output_dir,
            project_id=args.project_id,
        )
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to submit task: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
