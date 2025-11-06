"""
Visualize join_clips frame alignment logic to debug boundary issues.

This script simulates the frame calculations for both INSERT and REPLACE modes
to help identify alignment issues.
"""

def visualize_join_clips_alignment(
    start_frame_count=100,
    end_frame_count=100,
    context_frame_count=8,
    gap_frame_count=53,
    blend_frames=3,
    regenerate_anchors=True,
    num_anchor_frames=3,
    replace_mode=False
):
    """Simulate and visualize join_clips frame alignment."""

    print("=" * 80)
    print(f"JOIN_CLIPS FRAME ALIGNMENT VISUALIZATION")
    print(f"Mode: {'REPLACE' if replace_mode else 'INSERT'}")
    print("=" * 80)
    print(f"Clip1 frames: {start_frame_count}")
    print(f"Clip2 frames: {end_frame_count}")
    print(f"Context frames: {context_frame_count}")
    print(f"Gap frames: {gap_frame_count}")
    print(f"Blend frames: {blend_frames}")
    if not replace_mode:
        print(f"Regenerate anchors: {regenerate_anchors}")
        print(f"Num anchor frames: {num_anchor_frames if regenerate_anchors else 0}")
    print()

    # Step 1: Context extraction
    print("--- STEP 1: Context Extraction ---")
    start_context_start_idx = start_frame_count - context_frame_count
    end_context_start_idx = 0

    print(f"Clip1 context: clip1[{start_context_start_idx}:{start_frame_count}]")
    print(f"Clip2 context: clip2[{end_context_start_idx}:{context_frame_count}]")
    print()

    # Step 2: Transition video structure
    print("--- STEP 2: Transition Video Structure ---")

    if replace_mode:
        # REPLACE mode logic from vace_frame_utils.py
        frames_to_replace_from_before = gap_frame_count // 2
        frames_to_replace_from_after = gap_frame_count - frames_to_replace_from_before
        num_preserved_before = context_frame_count - frames_to_replace_from_before
        num_preserved_after = context_frame_count - frames_to_replace_from_after
        total_frames = context_frame_count * 2

        if num_preserved_before < 0 or num_preserved_after < 0:
            print(f"⚠️  WARNING: gap_frame_count ({gap_frame_count}) > context_frame_count*2 ({context_frame_count*2})")
            print(f"   num_preserved_before = {num_preserved_before} (INVALID!)")
            print(f"   num_preserved_after = {num_preserved_after} (INVALID!)")
            print(f"   REPLACE mode requires gap_frame_count <= context_frame_count * 2")
            return

        print(f"Frames to replace: {frames_to_replace_from_before} from before, {frames_to_replace_from_after} from after")
        print(f"Preserved frames: {num_preserved_before} from before, {num_preserved_after} from after")
        print(f"Total transition frames: {total_frames}")
        print()
        print("Transition video layout:")

        # Preserved from clip1
        print(f"  [{0}:{num_preserved_before}] (preserved from clip1)")
        print(f"    = clip1_context[0:{num_preserved_before}]")
        print(f"    = clip1[{start_context_start_idx}:{start_context_start_idx + num_preserved_before}]")

        # Gap (generated)
        gap_start = num_preserved_before
        gap_end = gap_start + gap_frame_count
        print(f"  [{gap_start}:{gap_end}] (generated gap, replaces boundary)")

        # Preserved from clip2
        preserved_clip2_start = gap_end
        print(f"  [{preserved_clip2_start}:{total_frames}] (preserved from clip2)")
        print(f"    = clip2_context[{frames_to_replace_from_after}:{context_frame_count}]")
        print(f"    = clip2[{frames_to_replace_from_after}:{context_frame_count}]")
        print()

    else:
        # INSERT mode logic from vace_frame_utils.py
        if regenerate_anchors:
            num_anchor_frames_before = min(num_anchor_frames, context_frame_count)
            num_anchor_frames_after = min(num_anchor_frames, context_frame_count)
        else:
            num_anchor_frames_before = 0
            num_anchor_frames_after = 0

        num_preserved_before = context_frame_count - num_anchor_frames_before
        num_preserved_after = context_frame_count - num_anchor_frames_after
        total_frames = context_frame_count + gap_frame_count + context_frame_count

        print(f"Anchor frames: {num_anchor_frames_before} from before, {num_anchor_frames_after} from after")
        print(f"Preserved frames: {num_preserved_before} from before, {num_preserved_after} from after")
        print(f"Total transition frames: {total_frames}")
        print()
        print("Transition video layout:")

        # Preserved from clip1 (before anchors)
        print(f"  [0:{num_preserved_before}] (preserved from clip1)")
        print(f"    = clip1_context[0:{num_preserved_before}]")
        print(f"    = clip1[{start_context_start_idx}:{start_context_start_idx + num_preserved_before}]")

        # Regenerated anchors from end of clip1
        anchors1_start = num_preserved_before
        anchors1_end = anchors1_start + num_anchor_frames_before
        if num_anchor_frames_before > 0:
            print(f"  [{anchors1_start}:{anchors1_end}] (regenerated anchors from clip1)")
            print(f"    = regenerated from clip1_context[{num_preserved_before}:{context_frame_count}]")

        # Gap
        gap_start = anchors1_end
        gap_end = gap_start + gap_frame_count
        print(f"  [{gap_start}:{gap_end}] (generated gap)")

        # Regenerated anchors from start of clip2
        anchors2_start = gap_end
        anchors2_end = anchors2_start + num_anchor_frames_after
        if num_anchor_frames_after > 0:
            print(f"  [{anchors2_start}:{anchors2_end}] (regenerated anchors from clip2)")
            print(f"    = regenerated from clip2_context[0:{num_anchor_frames_after}]")

        # Preserved from clip2 (after anchors)
        preserved_clip2_start = anchors2_end
        print(f"  [{preserved_clip2_start}:{total_frames}] (preserved from clip2)")
        print(f"    = clip2_context[{num_anchor_frames_after}:{context_frame_count}]")
        print(f"    = clip2[{num_anchor_frames_after}:{context_frame_count}]")
        print()

    # Step 3: Clip trimming
    print("--- STEP 3: Clip Trimming ---")

    # Clip1
    if replace_mode:
        frames_to_remove_clip1 = context_frame_count - blend_frames
        frames_to_keep_clip1 = start_frame_count - frames_to_remove_clip1
    else:
        frames_to_keep_clip1 = start_frame_count - (context_frame_count - blend_frames)

    print(f"Clip1 trimmed: [0:{frames_to_keep_clip1}]")
    print(f"  Last {blend_frames} frames: [{frames_to_keep_clip1 - blend_frames}:{frames_to_keep_clip1}]")

    # Clip2
    if replace_mode:
        frames_to_skip_clip2 = frames_to_replace_from_after + (num_preserved_after - blend_frames)
    else:
        frames_to_skip_clip2 = context_frame_count - blend_frames

    print(f"Clip2 trimmed: [{frames_to_skip_clip2}:{end_frame_count}]")
    print(f"  First {blend_frames} frames: [{frames_to_skip_clip2}:{frames_to_skip_clip2 + blend_frames}]")
    print()

    # Step 4: Blend alignment verification
    print("--- STEP 4: Blend Alignment Verification ---")

    # Clip1 → Transition blend
    print("Boundary 1: Clip1 → Transition")
    clip1_blend_frames = f"clip1[{frames_to_keep_clip1 - blend_frames}:{frames_to_keep_clip1}]"

    if replace_mode:
        transition_start_frames = f"transition[0:{blend_frames}]"
        transition_start_content = f"clip1[{start_context_start_idx}:{start_context_start_idx + blend_frames}]"
    else:
        transition_start_frames = f"transition[0:{blend_frames}]"
        transition_start_content = f"clip1[{start_context_start_idx}:{start_context_start_idx + blend_frames}]"

    print(f"  Clip1 last {blend_frames}: {clip1_blend_frames}")
    print(f"    = clip1[{frames_to_keep_clip1 - blend_frames}:{frames_to_keep_clip1}]")
    print(f"  Transition first {blend_frames}: {transition_start_frames}")
    print(f"    = preserved {transition_start_content}")

    # Check if they match
    clip1_last_start = frames_to_keep_clip1 - blend_frames
    transition_preserved_start = start_context_start_idx
    if clip1_last_start == transition_preserved_start:
        print(f"  ✓ MATCH: Both reference clip1[{clip1_last_start}:{clip1_last_start + blend_frames}]")
    else:
        print(f"  ✗ MISMATCH: clip1 ends at {clip1_last_start}, transition starts at {transition_preserved_start}")
        print(f"    Off by {clip1_last_start - transition_preserved_start} frames")
    print()

    # Transition → Clip2 blend
    print("Boundary 2: Transition → Clip2")
    transition_end_start = total_frames - blend_frames
    transition_end_frames = f"transition[{transition_end_start}:{total_frames}]"

    if replace_mode:
        # The last blend_frames of transition are from the preserved clip2 section
        # Preserved section starts at clip2[frames_to_replace_from_after]
        # And occupies transition[preserved_clip2_start:total_frames]
        offset_into_preserved = transition_end_start - preserved_clip2_start
        clip2_frame_in_transition_start = frames_to_replace_from_after + offset_into_preserved
        transition_end_content = f"clip2[{clip2_frame_in_transition_start}:{clip2_frame_in_transition_start + blend_frames}]"
    else:
        # The last blend_frames of transition are from the preserved clip2 section
        # Preserved section is clip2[num_anchor_frames_after:context_frame_count]
        # And occupies transition[preserved_clip2_start:total_frames]
        offset_into_preserved = transition_end_start - preserved_clip2_start
        clip2_frame_in_transition_start = num_anchor_frames_after + offset_into_preserved
        transition_end_content = f"clip2[{clip2_frame_in_transition_start}:{clip2_frame_in_transition_start + blend_frames}]"

    clip2_blend_frames = f"clip2[{frames_to_skip_clip2}:{frames_to_skip_clip2 + blend_frames}]"

    print(f"  Transition last {blend_frames}: {transition_end_frames}")
    print(f"    = preserved {transition_end_content}")
    print(f"  Clip2 first {blend_frames}: {clip2_blend_frames}")
    print(f"    = clip2[{frames_to_skip_clip2}:{frames_to_skip_clip2 + blend_frames}]")

    # Check if they match
    if clip2_frame_in_transition_start == frames_to_skip_clip2:
        print(f"  ✓ MATCH: Both reference clip2[{frames_to_skip_clip2}:{frames_to_skip_clip2 + blend_frames}]")
    else:
        print(f"  ✗ MISMATCH: transition references clip2[{clip2_frame_in_transition_start}], clip2 starts at [{frames_to_skip_clip2}]")
        print(f"    Off by {frames_to_skip_clip2 - clip2_frame_in_transition_start} frames")
    print()


if __name__ == "__main__":
    print("\n🔍 Testing REPLACE mode with small gap:")
    visualize_join_clips_alignment(
        replace_mode=True,
        gap_frame_count=6,
        blend_frames=3
    )

    print("\n" + "=" * 80)
    print("\n🔍 Testing REPLACE mode with odd gap:")
    visualize_join_clips_alignment(
        replace_mode=True,
        gap_frame_count=7,
        blend_frames=3
    )

    print("\n" + "=" * 80)
    print("\n🔍 Testing INSERT mode with regenerate_anchors:")
    visualize_join_clips_alignment(
        replace_mode=False,
        gap_frame_count=53,
        blend_frames=3,
        regenerate_anchors=True,
        num_anchor_frames=3
    )

    print("\n" + "=" * 80)
    print("\n🔍 Testing REPLACE mode with default gap (should fail):")
    visualize_join_clips_alignment(
        replace_mode=True,
        gap_frame_count=53,  # Default from join_clips.py
        blend_frames=3
    )
