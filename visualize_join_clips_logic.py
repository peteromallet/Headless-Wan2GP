#!/usr/bin/env python3
"""
Join Clips Logic Visualizer

This script visualizes the complete flow of join_clips frame processing:
1. Context extraction from input videos
2. Guide video construction
3. Mask video construction
4. VACE generation
5. Video trimming
6. Crossfade stitching

Run this to understand exactly what happens to each frame.
"""

from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class FrameMapping:
    """Tracks where a frame comes from and where it goes"""
    source_video: str  # "clip1", "transition", "clip2"
    source_index: int  # Index in source video
    stage: str  # "context", "guide", "generated", "trimmed", "final"
    final_index: int | None  # Index in final output (None if not in final)
    is_blended: bool = False  # Whether this frame is crossfaded
    blend_partner: Tuple[str, int] | None = None  # What it's blended with


def visualize_join_clips(
    clip1_frames: int = 416,
    clip2_frames: int = 416,
    context_frame_count: int = 20,
    gap_frame_count: int = 20,
    regenerate_anchors: bool = True,
    num_anchor_frames: int = 3,
    blend_frames: int = 3,
):
    """
    Visualize the complete join_clips logic with frame-level detail.

    Args:
        clip1_frames: Number of frames in starting clip
        clip2_frames: Number of frames in ending clip
        context_frame_count: Context frames for VACE
        gap_frame_count: Gap frames to generate
        regenerate_anchors: Whether to regenerate anchor frames
        num_anchor_frames: Number of anchor frames to regenerate
        blend_frames: Number of frames to blend at boundaries
    """

    print("=" * 100)
    print("JOIN CLIPS LOGIC VISUALIZATION")
    print("=" * 100)
    print(f"\nInput Parameters:")
    print(f"  clip1_frames: {clip1_frames}")
    print(f"  clip2_frames: {clip2_frames}")
    print(f"  context_frame_count: {context_frame_count}")
    print(f"  gap_frame_count: {gap_frame_count}")
    print(f"  regenerate_anchors: {regenerate_anchors}")
    print(f"  num_anchor_frames: {num_anchor_frames}")
    print(f"  blend_frames: {blend_frames}")

    # ========================================
    # PHASE 1: CONTEXT EXTRACTION
    # ========================================
    print("\n" + "=" * 100)
    print("PHASE 1: CONTEXT EXTRACTION")
    print("=" * 100)

    # Extract last N frames from clip1
    clip1_context_start = clip1_frames - context_frame_count
    clip1_context_frames = list(range(clip1_context_start, clip1_frames))

    print(f"\nClip1 (total: {clip1_frames} frames):")
    print(f"  Context: last {context_frame_count} frames")
    print(f"  Indices: [{clip1_context_start}:{clip1_frames}]")
    print(f"  Frames: {clip1_context_frames[:3]}...{clip1_context_frames[-3:]}")

    # Extract first N frames from clip2
    clip2_context_frames = list(range(0, context_frame_count))

    print(f"\nClip2 (total: {clip2_frames} frames):")
    print(f"  Context: first {context_frame_count} frames")
    print(f"  Indices: [0:{context_frame_count}]")
    print(f"  Frames: {clip2_context_frames[:3]}...{clip2_context_frames[-3:]}")

    # ========================================
    # PHASE 2: GUIDE VIDEO CONSTRUCTION
    # ========================================
    print("\n" + "=" * 100)
    print("PHASE 2: GUIDE VIDEO CONSTRUCTION")
    print("=" * 100)

    guide_frames = []
    guide_frame_sources = []  # Track what each guide frame represents

    if regenerate_anchors:
        # Preserved frames from clip1 (exclude last N anchors)
        num_preserved_before = context_frame_count - num_anchor_frames
        for i in range(num_preserved_before):
            frame_idx = clip1_context_frames[i]
            guide_frames.append(f"C1[{frame_idx}]")
            guide_frame_sources.append(("clip1", frame_idx, "preserved"))

        print(f"\n1. Preserved from Clip1: {num_preserved_before} frames")
        print(f"   Frames: {guide_frames[:3]}...{guide_frames[-3:] if len(guide_frames) >= 3 else guide_frames}")

        # Regenerated anchor frames (gray placeholders)
        for i in range(num_anchor_frames):
            guide_frames.append(f"GRAY-A1[{i}]")
            guide_frame_sources.append(("anchor1", i, "regenerated"))

        print(f"\n2. Regenerated Anchors (end of clip1): {num_anchor_frames} frames")
        print(f"   Frames: {guide_frames[-num_anchor_frames:]}")

        # Gap frames (gray placeholders)
        gap_start = len(guide_frames)
        for i in range(gap_frame_count):
            guide_frames.append(f"GRAY-GAP[{i}]")
            guide_frame_sources.append(("gap", i, "generated"))

        print(f"\n3. Gap: {gap_frame_count} frames")
        print(f"   Frames: {guide_frames[gap_start:gap_start+3]}...{guide_frames[-3:]}")

        # Regenerated anchor frames (gray placeholders)
        for i in range(num_anchor_frames):
            guide_frames.append(f"GRAY-A2[{i}]")
            guide_frame_sources.append(("anchor2", i, "regenerated"))

        print(f"\n4. Regenerated Anchors (start of clip2): {num_anchor_frames} frames")
        print(f"   Frames: {guide_frames[-num_anchor_frames:]}")

        # Preserved frames from clip2 (skip first N anchors)
        preserved_start = len(guide_frames)
        for i in range(num_anchor_frames, context_frame_count):
            frame_idx = clip2_context_frames[i]
            guide_frames.append(f"C2[{frame_idx}]")
            guide_frame_sources.append(("clip2", frame_idx, "preserved"))

        print(f"\n5. Preserved from Clip2: {context_frame_count - num_anchor_frames} frames")
        print(f"   Frames: {guide_frames[preserved_start:preserved_start+3]}...{guide_frames[-3:]}")
    else:
        # Without regenerate_anchors: all context frames preserved
        for frame_idx in clip1_context_frames:
            guide_frames.append(f"C1[{frame_idx}]")
            guide_frame_sources.append(("clip1", frame_idx, "preserved"))

        for i in range(gap_frame_count):
            guide_frames.append(f"GRAY-GAP[{i}]")
            guide_frame_sources.append(("gap", i, "generated"))

        for frame_idx in clip2_context_frames:
            guide_frames.append(f"C2[{frame_idx}]")
            guide_frame_sources.append(("clip2", frame_idx, "preserved"))

    total_guide_frames = len(guide_frames)
    print(f"\n{'─' * 100}")
    print(f"GUIDE VIDEO TOTAL: {total_guide_frames} frames")
    print(f"Structure: {guide_frames[0]} ... {guide_frames[total_guide_frames//2]} ... {guide_frames[-1]}")

    # ========================================
    # PHASE 3: MASK VIDEO CONSTRUCTION
    # ========================================
    print("\n" + "=" * 100)
    print("PHASE 3: MASK VIDEO CONSTRUCTION")
    print("=" * 100)

    mask_frames = []

    if regenerate_anchors:
        # Mark preserved frames as black (inactive/keep)
        num_inactive_before = context_frame_count - num_anchor_frames
        for i in range(num_inactive_before):
            mask_frames.append("BLACK")

        print(f"\n1. BLACK (keep): first {num_inactive_before} frames (preserved clip1 context)")

        # Mark regenerated + gap + regenerated as white (active/generate)
        num_active = num_anchor_frames + gap_frame_count + num_anchor_frames
        for i in range(num_active):
            mask_frames.append("WHITE")

        print(f"2. WHITE (generate): next {num_active} frames ({num_anchor_frames} anchor + {gap_frame_count} gap + {num_anchor_frames} anchor)")

        # Mark preserved frames as black
        num_inactive_after = context_frame_count - num_anchor_frames
        for i in range(num_inactive_after):
            mask_frames.append("BLACK")

        print(f"3. BLACK (keep): last {num_inactive_after} frames (preserved clip2 context)")
    else:
        # Without regenerate_anchors: preserve all context, generate only gap
        for i in range(context_frame_count):
            mask_frames.append("BLACK")

        for i in range(gap_frame_count):
            mask_frames.append("WHITE")

        for i in range(context_frame_count):
            mask_frames.append("BLACK")

    # Visualize mask
    print(f"\n{'─' * 100}")
    print(f"MASK VIDEO ({len(mask_frames)} frames):")

    # Find boundaries
    black_ranges = []
    white_ranges = []
    current_color = mask_frames[0]
    start_idx = 0

    for i in range(1, len(mask_frames) + 1):
        if i == len(mask_frames) or mask_frames[i] != current_color:
            if current_color == "BLACK":
                black_ranges.append((start_idx, i - 1))
            else:
                white_ranges.append((start_idx, i - 1))

            if i < len(mask_frames):
                current_color = mask_frames[i]
                start_idx = i

    print(f"  BLACK (keep) ranges: {black_ranges}")
    print(f"  WHITE (generate) ranges: {white_ranges}")

    # ========================================
    # PHASE 4: VACE GENERATION (Simulated)
    # ========================================
    print("\n" + "=" * 100)
    print("PHASE 4: VACE GENERATION (Simulated)")
    print("=" * 100)

    transition_frames = []

    for i, (frame_label, (source, idx, status)) in enumerate(zip(guide_frames, guide_frame_sources)):
        if mask_frames[i] == "BLACK":
            # Preserved (but with VACE artifacts)
            transition_frames.append(f"{frame_label}'")
        else:
            # Generated
            if status == "regenerated":
                transition_frames.append(f"GEN-A[{idx}]")
            else:
                transition_frames.append(f"GEN-GAP[{idx}]")

    print(f"\nTransition video: {len(transition_frames)} frames")
    print(f"  First 5: {transition_frames[:5]}")
    print(f"  Middle 5: {transition_frames[len(transition_frames)//2-2:len(transition_frames)//2+3]}")
    print(f"  Last 5: {transition_frames[-5:]}")

    # ========================================
    # PHASE 5: TRIMMING
    # ========================================
    print("\n" + "=" * 100)
    print("PHASE 5: TRIMMING FOR STITCHING")
    print("=" * 100)

    # Calculate trimming
    frames_to_keep_clip1 = clip1_frames - (context_frame_count - blend_frames)
    frames_to_skip_clip2 = context_frame_count - blend_frames

    print(f"\nTrimming calculations:")
    print(f"  Clip1: keep {frames_to_keep_clip1}/{clip1_frames} frames")
    print(f"    = {clip1_frames} - ({context_frame_count} - {blend_frames})")
    print(f"    = {clip1_frames} - {context_frame_count - blend_frames}")
    print(f"    Frames: [0:{frames_to_keep_clip1}]")

    print(f"\n  Transition: keep all {len(transition_frames)} frames")
    print(f"    Frames: [0:{len(transition_frames)}]")

    print(f"\n  Clip2: skip first {frames_to_skip_clip2}/{clip2_frames} frames, keep {clip2_frames - frames_to_skip_clip2}")
    print(f"    = skip ({context_frame_count} - {blend_frames})")
    print(f"    = skip {frames_to_skip_clip2}")
    print(f"    Frames: [{frames_to_skip_clip2}:{clip2_frames}]")

    clip1_trimmed = [f"C1[{i}]" for i in range(frames_to_keep_clip1)]
    transition_trimmed = transition_frames[:]
    clip2_trimmed = [f"C2[{i}]" for i in range(frames_to_skip_clip2, clip2_frames)]

    print(f"\nTrimmed videos:")
    print(f"  Clip1: {len(clip1_trimmed)} frames")
    print(f"    Last 5: {clip1_trimmed[-5:]}")
    print(f"  Transition: {len(transition_trimmed)} frames")
    print(f"    First 5: {transition_trimmed[:5]}")
    print(f"    Last 5: {transition_trimmed[-5:]}")
    print(f"  Clip2: {len(clip2_trimmed)} frames")
    print(f"    First 5: {clip2_trimmed[:5]}")

    # ========================================
    # PHASE 6: CROSSFADE STITCHING (CORRECTED)
    # ========================================
    print("\n" + "=" * 100)
    print("PHASE 6: CROSSFADE STITCHING (CORRECTED)")
    print("=" * 100)

    final_frames = []
    overlap_frames_for_next_blend = []  # Track overlap frames from previous video
    stitch_log = []

    # Video 0: Clip1 trimmed
    print(f"\n1. Processing Clip1 (video 0, {len(clip1_trimmed)} frames)")
    blend_with_next = blend_frames
    if blend_with_next > 0:
        frames_to_add = clip1_trimmed[:-blend_with_next]
        overlap_frames_for_next_blend = clip1_trimmed[-blend_with_next:]
    else:
        frames_to_add = clip1_trimmed
        overlap_frames_for_next_blend = []

    final_frames.extend(frames_to_add)
    print(f"   Added {len(frames_to_add)} frames (keeping {blend_with_next} for blend)")
    print(f"   Last 3 added: {frames_to_add[-3:]}")
    print(f"   Kept for blend: {overlap_frames_for_next_blend}")
    stitch_log.append(f"Clip1: added {len(frames_to_add)} frames, total={len(final_frames)}")

    # Video 1: Transition
    print(f"\n2. Processing Transition (video 1, {len(transition_trimmed)} frames)")
    blend_count = blend_frames

    if blend_count > 0 and overlap_frames_for_next_blend:
        # Use the overlap frames we saved from previous video (CORRECTED)
        frames_prev_for_fade = overlap_frames_for_next_blend
        print(f"   Using saved overlap frames for fade: {frames_prev_for_fade}")

        # Get frames for crossfade from current
        frames_curr_for_fade = transition_trimmed[:blend_count]
        print(f"   Current frames for fade: {frames_curr_for_fade}")

        # Crossfade (simulated)
        faded_frames = [f"BLEND({pf}⊕{cf})" for pf, cf in zip(frames_prev_for_fade, frames_curr_for_fade)]
        final_frames.extend(faded_frames)
        print(f"   Added {len(faded_frames)} crossfaded frames")
        print(f"   Blended: {faded_frames}")
        stitch_log.append(f"Blend1: added {len(faded_frames)} blended, total={len(final_frames)}")

        start_idx = blend_count
    else:
        start_idx = 0

    # Add remaining from transition
    blend_with_next = blend_frames
    end_idx = len(transition_trimmed) - blend_with_next if blend_with_next > 0 else len(transition_trimmed)
    frames_to_add = transition_trimmed[start_idx:end_idx]
    final_frames.extend(frames_to_add)

    # Save overlap frames for next blend
    if blend_with_next > 0:
        overlap_frames_for_next_blend = transition_trimmed[-blend_with_next:]
    else:
        overlap_frames_for_next_blend = []

    print(f"   Added {len(frames_to_add)} non-overlapping frames (keeping {blend_with_next} for next blend)")
    if len(frames_to_add) > 0:
        print(f"   First 3: {frames_to_add[:3]}")
        print(f"   Last 3: {frames_to_add[-3:]}")
    print(f"   Kept for blend: {overlap_frames_for_next_blend}")
    stitch_log.append(f"Transition: added {len(frames_to_add)} frames, total={len(final_frames)}")

    # Video 2: Clip2
    print(f"\n3. Processing Clip2 (video 2, {len(clip2_trimmed)} frames)")
    blend_count = blend_frames

    if blend_count > 0 and overlap_frames_for_next_blend:
        # Use the overlap frames we saved from previous video (CORRECTED)
        frames_prev_for_fade = overlap_frames_for_next_blend
        print(f"   Using saved overlap frames for fade: {frames_prev_for_fade}")

        # Get frames for crossfade from current
        frames_curr_for_fade = clip2_trimmed[:blend_count]
        print(f"   Current frames for fade: {frames_curr_for_fade}")

        # Crossfade (simulated)
        faded_frames = [f"BLEND({pf}⊕{cf})" for pf, cf in zip(frames_prev_for_fade, frames_curr_for_fade)]
        final_frames.extend(faded_frames)
        print(f"   Added {len(faded_frames)} crossfaded frames")
        print(f"   Blended: {faded_frames}")
        stitch_log.append(f"Blend2: added {len(faded_frames)} blended, total={len(final_frames)}")

        start_idx = blend_count
    else:
        start_idx = 0

    # Add remaining from clip2
    frames_to_add = clip2_trimmed[start_idx:]
    final_frames.extend(frames_to_add)
    print(f"   Added {len(frames_to_add)} remaining frames")
    print(f"   First 3: {frames_to_add[:3]}")
    stitch_log.append(f"Clip2: added {len(frames_to_add)} frames, total={len(final_frames)}")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    print(f"\nFrame counts:")
    print(f"  Input clip1: {clip1_frames} frames")
    print(f"  Input clip2: {clip2_frames} frames")
    print(f"  Guide video: {len(guide_frames)} frames")
    print(f"  Transition video: {len(transition_frames)} frames")
    print(f"  Clip1 trimmed: {len(clip1_trimmed)} frames")
    print(f"  Transition trimmed: {len(transition_trimmed)} frames")
    print(f"  Clip2 trimmed: {len(clip2_trimmed)} frames")
    print(f"  Final output: {len(final_frames)} frames")

    # With the corrected crossfade logic, we keep all frames but use overlap frames for blending
    # The blended frames represent the overlapping regions, so they're included in the count
    expected_trimmed_sum = (clip1_frames - context_frame_count) + len(transition_frames) + (clip2_frames - context_frame_count)
    print(f"\nSum of trimmed videos: {clip1_frames - context_frame_count} + {len(transition_frames)} + {clip2_frames - context_frame_count} = {expected_trimmed_sum}")
    print(f"Expected with corrected crossfade: {expected_trimmed_sum} frames")
    print(f"  (Crossfaded frames are included, representing the blended overlapping regions)")
    print(f"Actual: {len(final_frames)}")

    expected = expected_trimmed_sum
    if len(final_frames) == expected:
        print(f"\n✅ Frame count matches expected!")
    else:
        print(f"\n❌ Frame count mismatch! Expected {expected}, got {len(final_frames)}")

    print(f"\nStitching log:")
    for log in stitch_log:
        print(f"  {log}")

    print(f"\nFinal video structure:")
    print(f"  Frames 0-9: {final_frames[:10]}")
    print(f"  Frames {len(final_frames)//2-5}-{len(final_frames)//2+4}: {final_frames[len(final_frames)//2-5:len(final_frames)//2+5]}")
    print(f"  Frames {len(final_frames)-10}-{len(final_frames)-1}: {final_frames[-10:]}")

    # ========================================
    # BOUNDARY ANALYSIS
    # ========================================
    print("\n" + "=" * 100)
    print("BOUNDARY ANALYSIS - WHERE JUMPS MIGHT OCCUR")
    print("=" * 100)

    # Find blend boundaries in final output
    blend_indices = []
    for i, frame in enumerate(final_frames):
        if "BLEND" in frame:
            blend_indices.append(i)

    if blend_indices:
        print(f"\nBlended frames in final output:")
        for idx in blend_indices:
            context_start = max(0, idx - 2)
            context_end = min(len(final_frames), idx + 3)
            print(f"  Frame {idx}: {final_frames[context_start:context_end]}")

    print(f"\nPotential jump locations:")
    print(f"1. Boundary 1 (Clip1 → Transition): around frame {len(clip1_trimmed) - blend_frames}")
    context_idx = len(frames_to_add)
    print(f"   Before blend: {final_frames[max(0, context_idx-3):context_idx]}")
    print(f"   Blend zone: {final_frames[context_idx:context_idx+blend_frames]}")
    print(f"   After blend: {final_frames[context_idx+blend_frames:context_idx+blend_frames+3]}")

    print(f"\n2. Boundary 2 (Transition → Clip2): around frame {len(final_frames) - len(clip2_trimmed) + blend_frames}")
    context_idx = len(final_frames) - len(clip2_trimmed) + blend_frames
    print(f"   Before blend: {final_frames[max(0, context_idx-3):context_idx]}")
    print(f"   Blend zone: {final_frames[max(0, context_idx-blend_frames):context_idx]}")
    print(f"   After blend: {final_frames[context_idx:min(len(final_frames), context_idx+3)]}")


if __name__ == "__main__":
    import sys

    # Default parameters matching the logs
    params = {
        "clip1_frames": 416,
        "clip2_frames": 416,
        "context_frame_count": 20,
        "gap_frame_count": 20,
        "regenerate_anchors": True,
        "num_anchor_frames": 3,
        "blend_frames": 3,
    }

    # Allow command-line overrides
    if len(sys.argv) > 1:
        print("Usage: python visualize_join_clips_logic.py")
        print("\nEdit the 'params' dict in the script to change parameters")
        sys.exit(0)

    visualize_join_clips(**params)
