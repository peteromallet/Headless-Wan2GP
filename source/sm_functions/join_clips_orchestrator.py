"""
Join Clips Orchestrator - Sequentially join multiple video clips

This orchestrator takes a list of video clips and creates a chain of join_clips_child
tasks to progressively build them into a single seamless video.

Pattern:
    Input: [clip_A, clip_B, clip_C, clip_D]

    Creates:
        join_0: clip_A + clip_B → AB.mp4 (no dependency)
        join_1: AB.mp4 + clip_C → ABC.mp4 (depends on join_0)
        join_2: ABC.mp4 + clip_D → ABCD.mp4 (depends on join_1)

    Each join task fetches the output of its predecessor via get_predecessor_output_via_edge_function()
"""

import traceback
from pathlib import Path
from typing import Tuple

from .. import db_operations as db_ops


def _handle_join_clips_orchestrator_task(
    task_params_from_db: dict,
    main_output_dir_base: Path,
    orchestrator_task_id_str: str,
    orchestrator_project_id: str | None,
    *,
    dprint
) -> Tuple[bool, str]:
    """
    Handle join_clips_orchestrator task - creates chained join_clips_child tasks.

    Args:
        task_params_from_db: Task parameters containing orchestrator_details
        main_output_dir_base: Base output directory
        orchestrator_task_id_str: Orchestrator task ID
        orchestrator_project_id: Project ID for authorization
        dprint: Debug print function

    Returns:
        (success: bool, message: str)
    """
    dprint(f"[JOIN_ORCHESTRATOR] Starting orchestrator task {orchestrator_task_id_str}")

    generation_success = False
    output_message = f"Join orchestration for {orchestrator_task_id_str} initiated."

    try:
        # === 1. PARSE ORCHESTRATOR PAYLOAD ===
        if 'orchestrator_details' not in task_params_from_db:
            dprint("[JOIN_ORCHESTRATOR] ERROR: orchestrator_details missing")
            return False, "orchestrator_details missing"

        orchestrator_payload = task_params_from_db['orchestrator_details']
        dprint(f"[JOIN_ORCHESTRATOR] Orchestrator payload keys: {list(orchestrator_payload.keys())}")

        # Extract required fields
        clip_list = orchestrator_payload.get("clip_list", [])
        run_id = orchestrator_payload.get("run_id")

        if not clip_list or len(clip_list) < 2:
            return False, "clip_list must contain at least 2 clips"

        if not run_id:
            return False, "run_id is required"

        num_joins = len(clip_list) - 1
        dprint(f"[JOIN_ORCHESTRATOR] Processing {len(clip_list)} clips = {num_joins} join tasks")

        # Extract join settings
        join_settings = {
            "context_frame_count": orchestrator_payload.get("context_frame_count", 8),
            "gap_frame_count": orchestrator_payload.get("gap_frame_count", 53),
            "replace_mode": orchestrator_payload.get("replace_mode", False),
            "blend_frames": orchestrator_payload.get("blend_frames", 3),
            "prompt": orchestrator_payload.get("prompt", "smooth transition"),
            "negative_prompt": orchestrator_payload.get("negative_prompt", ""),
            "model": orchestrator_payload.get("model", "lightning_baseline_2_2_2"),
            "regenerate_anchors": orchestrator_payload.get("regenerate_anchors", True),
            "num_anchor_frames": orchestrator_payload.get("num_anchor_frames", 3),
            "aspect_ratio": orchestrator_payload.get("aspect_ratio"),
            # LoRA parameters
            "use_causvid_lora": orchestrator_payload.get("use_causvid_lora", False),
            "use_lighti2x_lora": orchestrator_payload.get("use_lighti2x_lora", False),
            "apply_reward_lora": orchestrator_payload.get("apply_reward_lora", False),
            "additional_loras": orchestrator_payload.get("additional_loras", {}),
        }

        per_join_settings = orchestrator_payload.get("per_join_settings", [])
        output_base_dir = orchestrator_payload.get("output_base_dir", str(main_output_dir_base.resolve()))

        # Create run-specific output directory
        current_run_output_dir = Path(output_base_dir) / f"join_clips_run_{run_id}"
        current_run_output_dir.mkdir(parents=True, exist_ok=True)
        dprint(f"[JOIN_ORCHESTRATOR] Run output directory: {current_run_output_dir}")


        # === 2. IDEMPOTENCY CHECK ===
        dprint(f"[JOIN_ORCHESTRATOR] Checking for existing child tasks")
        existing_child_tasks = db_ops.get_orchestrator_child_tasks(orchestrator_task_id_str)
        existing_joins = existing_child_tasks.get('join_clips_segment', [])

        if existing_joins:
            dprint(f"[JOIN_ORCHESTRATOR] Found {len(existing_joins)} existing join tasks")

            # Check if we have the expected number
            if len(existing_joins) >= num_joins:
                dprint(f"[JOIN_ORCHESTRATOR] All {num_joins} join tasks already exist")

                # Check completion status
                def is_complete(task):
                    return task.get('status') == 'complete'

                def is_terminal_failure(task):
                    status = task.get('status', '').lower()
                    return status in ('failed', 'cancelled', 'canceled', 'error')

                all_joins_complete = all(is_complete(join) for join in existing_joins)
                any_join_failed = any(is_terminal_failure(join) for join in existing_joins)

                # If any failed, mark orchestrator as failed
                if any_join_failed:
                    failed_joins = [j for j in existing_joins if is_terminal_failure(j)]
                    error_msg = f"{len(failed_joins)} join task(s) failed/cancelled"
                    dprint(f"[JOIN_ORCHESTRATOR] FAILED: {error_msg}")
                    return False, f"[ORCHESTRATOR_FAILED] {error_msg}"

                # If all complete, return final output
                if all_joins_complete:
                    # Sort by join_index to get the last one
                    sorted_joins = sorted(existing_joins, key=lambda x: x.get('task_params', {}).get('join_index', 0))
                    final_output = sorted_joins[-1].get('output_location', 'Completed via idempotency')
                    dprint(f"[JOIN_ORCHESTRATOR] COMPLETE: All joins finished, final output: {final_output}")
                    return True, f"[ORCHESTRATOR_COMPLETE]{final_output}"

                # Still in progress
                complete_count = sum(1 for j in existing_joins if is_complete(j))
                dprint(f"[JOIN_ORCHESTRATOR] IDEMPOTENT: {complete_count}/{num_joins} joins complete")
                return True, f"[IDEMPOTENT] Join tasks in progress: {complete_count}/{num_joins} complete"


        # === 3. CREATE JOIN_CLIPS_CHILD TASKS ===
        dprint(f"[JOIN_ORCHESTRATOR] Creating {num_joins} join tasks in dependency chain")

        previous_join_task_id = None
        joins_created = 0

        for idx in range(num_joins):
            clip_start = clip_list[idx]
            clip_end = clip_list[idx + 1]

            dprint(f"[JOIN_ORCHESTRATOR] Creating join {idx}: {clip_start.get('name', 'clip')} + {clip_end.get('name', 'clip')}")

            # Merge global settings with per-join overrides
            task_join_settings = join_settings.copy()
            if idx < len(per_join_settings):
                task_join_settings.update(per_join_settings[idx])
                dprint(f"[JOIN_ORCHESTRATOR] Applied per-join overrides for join {idx}")

            # Build join payload
            join_payload = {
                "orchestrator_task_id_ref": orchestrator_task_id_str,
                "orchestrator_run_id": run_id,
                "project_id": orchestrator_project_id,
                "join_index": idx,
                "is_first_join": (idx == 0),
                "is_last_join": (idx == num_joins - 1),

                # First join has explicit starting path, rest fetch from dependency
                "starting_video_path": clip_start.get("url") if idx == 0 else None,
                "ending_video_path": clip_end.get("url"),

                # Join settings
                **task_join_settings,

                # Output configuration
                "current_run_base_output_dir": str(current_run_output_dir.resolve()),
                "join_output_dir": str((current_run_output_dir / f"join_{idx}").resolve()),

                # Reference to full orchestrator payload
                "full_orchestrator_payload": orchestrator_payload,
            }

            dprint(f"[JOIN_ORCHESTRATOR] Submitting join {idx} to database, depends_on={previous_join_task_id}")

            # Create task with dependency chain
            actual_db_row_id = db_ops.add_task_to_db(
                task_payload=join_payload,
                task_type_str="join_clips_segment",
                dependant_on=previous_join_task_id
            )

            dprint(f"[JOIN_ORCHESTRATOR] Join {idx} created with DB ID: {actual_db_row_id}")

            # Update for next iteration
            previous_join_task_id = actual_db_row_id
            joins_created += 1


        # === 4. SUCCESS ===
        generation_success = True
        output_message = f"Successfully enqueued {joins_created} join tasks for run {run_id}"
        dprint(f"[JOIN_ORCHESTRATOR] {output_message}")

    except Exception as e:
        msg = f"Failed during join orchestration: {e}"
        dprint(f"[JOIN_ORCHESTRATOR] ERROR: {msg}")
        traceback.print_exc()
        generation_success = False
        output_message = msg

    return generation_success, output_message
