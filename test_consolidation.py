#!/usr/bin/env python3
"""
Test suite for group-based frame consolidation logic.
Tests the core algorithm before integration into travel_between_images.py
"""

def find_consolidatable_groups(prompts, neg_prompts, loras):
    """
    Find consecutive segments with identical parameters that can be consolidated.

    Args:
        prompts: List of prompts for each segment
        neg_prompts: List of negative prompts for each segment
        loras: List of LoRA configurations for each segment

    Returns:
        List of tuples: (start_idx, end_idx, can_consolidate)
        where can_consolidate=True means 2+ consecutive identical segments
    """
    if not prompts:
        return []

    groups = []
    current_group_start = 0

    for i in range(1, len(prompts)):
        # Check if segment i matches segment i-1
        params_match = (
            prompts[i] == prompts[i-1] and
            neg_prompts[i] == neg_prompts[i-1] and
            loras[i] == loras[i-1]
        )

        if not params_match:
            # End current group
            group_size = i - current_group_start
            can_consolidate = group_size > 1
            groups.append((current_group_start, i-1, can_consolidate))
            current_group_start = i

    # Handle last group
    group_size = len(prompts) - current_group_start
    can_consolidate = group_size > 1
    groups.append((current_group_start, len(prompts)-1, can_consolidate))

    return groups


def consolidate_group_frames(segment_frames, max_frames=81):
    """
    Consolidate a group of segments into fewer segments respecting frame limits.
    Replicates the real keyframe-based algorithm from optimize_frame_allocation_for_identical_params.

    Key concept: Each segment represents travel between keyframe images.
    With segments [20, 15, 25], we have 4 keyframe images at positions [0, 20, 35, 60].

    NOTE: Quantization is NOT applied here - it happens during actual video generation.
    This consolidation preserves exact frame counts.

    Args:
        segment_frames: List of frame counts for segments in this group
        max_frames: Maximum frames per consolidated segment

    Returns:
        List of consolidated frame counts (exact, no quantization)
    """
    if not segment_frames:
        return []

    if len(segment_frames) == 1:
        # Single segment - just return it (no consolidation needed)
        return segment_frames

    # Calculate keyframe positions (cumulative frame positions where images appear)
    keyframe_positions = [0]
    cumulative_pos = 0

    for frames in segment_frames:
        cumulative_pos += frames
        keyframe_positions.append(cumulative_pos)

    # Now consolidate: group keyframes into videos respecting frame limit
    consolidated_segments = []
    video_start = 0
    video_keyframes = [0]  # Always include first keyframe

    for i in range(1, len(keyframe_positions)):
        kf_pos = keyframe_positions[i]
        # Calculate how long the video would be if we include this keyframe
        # Note: This is the raw frame count from video_start to this keyframe
        video_length_if_included = kf_pos - video_start

        if video_length_if_included <= max_frames:
            # Keyframe fits in current video
            video_keyframes.append(kf_pos)
        else:
            # Current video is full, finalize it
            final_frame = video_keyframes[-1]
            video_length = final_frame - video_start
            consolidated_segments.append(video_length)

            # Start new video from the last keyframe
            video_start = video_keyframes[-1]
            video_keyframes = [video_start, kf_pos]

    # Finalize the last video
    final_frame = video_keyframes[-1]
    video_length = final_frame - video_start
    consolidated_segments.append(video_length)

    return consolidated_segments


def apply_group_consolidation(segment_frames, prompts, neg_prompts, loras, max_frames=81):
    """
    Apply group-based consolidation to a full segment list.

    Args:
        segment_frames: List of frame counts for all segments
        prompts: List of prompts for all segments
        neg_prompts: List of negative prompts for all segments
        loras: List of LoRA configs for all segments
        max_frames: Maximum frames per consolidated segment

    Returns:
        Dictionary with:
            - consolidated_frames: New frame allocation
            - consolidated_prompts: New prompt list
            - consolidated_neg_prompts: New negative prompt list
            - consolidated_loras: New LoRA list
            - group_info: Debug info about consolidation groups
    """
    # Find consolidatable groups
    groups = find_consolidatable_groups(prompts, neg_prompts, loras)

    # Build new consolidated lists
    new_frames = []
    new_prompts = []
    new_neg_prompts = []
    new_loras = []
    group_info = []

    for start_idx, end_idx, can_consolidate in groups:
        group_size = end_idx - start_idx + 1

        if can_consolidate:
            # Consolidate this group
            group_frames = segment_frames[start_idx:end_idx+1]
            consolidated = consolidate_group_frames(group_frames, max_frames)

            # Add consolidated segments
            new_frames.extend(consolidated)
            for _ in consolidated:
                new_prompts.append(prompts[start_idx])
                new_neg_prompts.append(neg_prompts[start_idx])
                new_loras.append(loras[start_idx])

            prompt_display = prompts[start_idx]
            if prompt_display is None:
                prompt_display = "None"
            elif len(prompt_display) > 50:
                prompt_display = prompt_display[:50] + "..."

            group_info.append({
                "original_range": f"{start_idx}-{end_idx}",
                "original_segments": group_size,
                "original_frames": group_frames,
                "consolidated_segments": len(consolidated),
                "consolidated_frames": consolidated,
                "saved_segments": group_size - len(consolidated),
                "prompt": prompt_display
            })
        else:
            # Keep single segment as-is
            new_frames.append(segment_frames[start_idx])
            new_prompts.append(prompts[start_idx])
            new_neg_prompts.append(neg_prompts[start_idx])
            new_loras.append(loras[start_idx])

            prompt_display = prompts[start_idx]
            if prompt_display is None:
                prompt_display = "None"
            elif len(prompt_display) > 50:
                prompt_display = prompt_display[:50] + "..."

            group_info.append({
                "original_range": f"{start_idx}",
                "original_segments": 1,
                "original_frames": [segment_frames[start_idx]],
                "consolidated_segments": 1,
                "consolidated_frames": [segment_frames[start_idx]],
                "saved_segments": 0,
                "prompt": prompt_display
            })

    return {
        "consolidated_frames": new_frames,
        "consolidated_prompts": new_prompts,
        "consolidated_neg_prompts": new_neg_prompts,
        "consolidated_loras": new_loras,
        "group_info": group_info
    }


def print_test_result(test_name, result):
    """Pretty print test results"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")

    original_total = sum([sum(g["original_frames"]) for g in result["group_info"]])
    consolidated_total = sum(result["consolidated_frames"])
    original_count = sum([g["original_segments"] for g in result["group_info"]])
    consolidated_count = len(result["consolidated_frames"])
    saved = original_count - consolidated_count

    print(f"\n📊 SUMMARY:")
    print(f"   Original: {original_count} segments, {original_total} total frames")
    print(f"   Consolidated: {consolidated_count} segments, {consolidated_total} total frames")
    print(f"   ⚡ Saved: {saved} segments ({saved} fewer model loads)")

    print(f"\n📹 FRAME ALLOCATION:")
    print(f"   Before: {[sum(g['original_frames']) for g in result['group_info']]}")
    print(f"   After:  {result['consolidated_frames']}")

    print(f"\n🔍 GROUP DETAILS:")
    for i, group in enumerate(result["group_info"]):
        status = "✅ CONSOLIDATED" if group["saved_segments"] > 0 else "⏭️  KEPT AS-IS"
        print(f"\n   Group {i+1} [{status}]:")
        print(f"      Range: segments {group['original_range']}")
        print(f"      Prompt: \"{group['prompt']}\"")
        print(f"      Original: {group['original_segments']} segments → {group['original_frames']}")
        print(f"      Result: {group['consolidated_segments']} segments → {group['consolidated_frames']}")
        if group["saved_segments"] > 0:
            print(f"      💾 Saved: {group['saved_segments']} segments")

    print(f"\n{'='*80}\n")


# =============================================================================
# TEST CASES
# =============================================================================

def test_all_identical():
    """All segments have identical parameters - should consolidate into fewer segments"""
    segment_frames = [20, 15, 25, 10, 15]
    prompts = ["A"] * 5
    neg_prompts = ["neg"] * 5
    loras = ["lora1"] * 5

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("All Identical (should consolidate all)", result)


def test_all_different():
    """All segments have different parameters - should keep all as-is"""
    segment_frames = [20, 15, 25, 10, 15]
    prompts = ["A", "B", "C", "D", "E"]
    neg_prompts = ["neg"] * 5
    loras = ["lora1"] * 5

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("All Different (should keep all as-is)", result)


def test_mixed_groups():
    """Mixed: some identical groups, some different - realistic scenario"""
    segment_frames = [20, 15, 25, 30, 10, 20, 15]
    prompts = ["A", "A", "A", "B", "C", "C", "D"]
    neg_prompts = ["neg"] * 7
    loras = ["lora1"] * 7

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("Mixed Groups (segments 0-2=A, 3=B, 4-5=C, 6=D)", result)


def test_exceeds_frame_limit():
    """Group that would exceed 81-frame limit - should split into multiple"""
    segment_frames = [50, 20, 30]  # Total = 100 frames, exceeds 81
    prompts = ["A"] * 3
    neg_prompts = ["neg"] * 3
    loras = ["lora1"] * 3

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras, max_frames=81)
    print_test_result("Exceeds Frame Limit (100 frames total, max 81)", result)


def test_alternating_prompts():
    """Alternating A-B-A-B pattern - no consolidation possible"""
    segment_frames = [20, 15, 20, 15]
    prompts = ["A", "B", "A", "B"]
    neg_prompts = ["neg"] * 4
    loras = ["lora1"] * 4

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("Alternating A-B-A-B (no consolidation possible)", result)


def test_single_segment():
    """Edge case: only one segment - should keep as-is"""
    segment_frames = [50]
    prompts = ["A"]
    neg_prompts = ["neg"]
    loras = ["lora1"]

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("Single Segment (edge case)", result)


def test_two_identical():
    """Edge case: exactly two identical segments"""
    segment_frames = [40, 35]
    prompts = ["A", "A"]
    neg_prompts = ["neg", "neg"]
    loras = ["lora1", "lora1"]

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("Two Identical Segments", result)


def test_large_many_small():
    """Many small segments with same prompt - should consolidate efficiently"""
    segment_frames = [10, 8, 12, 9, 11, 10, 8, 13]  # 8 segments, 81 total frames
    prompts = ["A"] * 8
    neg_prompts = ["neg"] * 8
    loras = ["lora1"] * 8

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras, max_frames=81)
    print_test_result("Many Small Segments (8 segments, 81 frames total)", result)


def test_neg_prompt_differs():
    """Prompts same but negative prompts differ - should NOT consolidate"""
    segment_frames = [20, 15, 25]
    prompts = ["A", "A", "A"]
    neg_prompts = ["neg1", "neg2", "neg1"]
    loras = ["lora1"] * 3

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("Negative Prompts Differ (should NOT consolidate)", result)


def test_lora_differs():
    """Prompts same but LoRAs differ - should NOT consolidate"""
    segment_frames = [20, 15, 25]
    prompts = ["A", "A", "A"]
    neg_prompts = ["neg"] * 3
    loras = ["lora1", "lora2", "lora1"]

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("LoRAs Differ (should NOT consolidate)", result)


def test_realistic_timeline():
    """Realistic timeline: intro (A), main content (B), outro (C)"""
    segment_frames = [15, 12, 10, 40, 35, 30, 20, 15, 10]
    prompts = ["intro scene"] * 3 + ["main action"] * 3 + ["outro scene"] * 3
    neg_prompts = ["blurry"] * 9
    loras = ["standard"] * 9

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("Realistic Timeline (intro/main/outro)", result)


def test_empty_input():
    """Edge case: empty arrays - should handle gracefully"""
    segment_frames = []
    prompts = []
    neg_prompts = []
    loras = []

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("Empty Input (should handle gracefully)", result)


def test_exactly_81_frames():
    """Edge case: exactly 81 frames in one group - should not split"""
    segment_frames = [40, 41]
    prompts = ["A", "A"]
    neg_prompts = ["neg", "neg"]
    loras = ["lora1", "lora1"]

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras, max_frames=81)
    print_test_result("Exactly 81 Frames (should fit in one segment)", result)


def test_82_frames():
    """Edge case: 82 frames (just over limit) - should split"""
    segment_frames = [40, 42]
    prompts = ["A", "A"]
    neg_prompts = ["neg", "neg"]
    loras = ["lora1", "lora1"]

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras, max_frames=81)
    print_test_result("82 Frames (just over limit, should split)", result)


def test_many_tiny_segments():
    """Stress test: 20 tiny segments - should consolidate efficiently"""
    segment_frames = [4, 3, 5, 4, 3, 6, 4, 5, 3, 4, 5, 3, 4, 6, 3, 5, 4, 3, 4, 5]
    prompts = ["A"] * 20
    neg_prompts = ["neg"] * 20
    loras = ["lora1"] * 20

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras, max_frames=81)
    print_test_result("20 Tiny Segments (stress test)", result)


def test_one_frame_segments():
    """Edge case: 1-frame segments - should still consolidate"""
    segment_frames = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    prompts = ["A"] * 10
    neg_prompts = ["neg"] * 10
    loras = ["lora1"] * 10

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("Ten 1-Frame Segments (edge case)", result)


def test_huge_single_segments():
    """Edge case: Each segment is already 80 frames - shouldn't consolidate"""
    segment_frames = [80, 80, 80]
    prompts = ["A", "A", "A"]
    neg_prompts = ["neg", "neg", "neg"]
    loras = ["lora1", "lora1", "lora1"]

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras, max_frames=81)
    print_test_result("Three 80-Frame Segments (too large to merge)", result)


def test_exact_frame_preservation():
    """Verify exact frame counts are preserved (no quantization in consolidation)"""
    segment_frames = [10, 11, 12, 13]  # Total = 46
    prompts = ["A"] * 4
    neg_prompts = ["neg"] * 4
    loras = ["lora1"] * 4

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("Exact Frame Preservation (46 frames → should stay 46)", result)


def test_multiple_groups_with_splits():
    """Complex: Multiple groups where some need splitting"""
    segment_frames = [30, 30, 30, 20, 20, 20, 20, 30, 30]  # Group1=90, Group2=80, Group3=60
    prompts = ["A", "A", "A", "B", "B", "B", "B", "C", "C"]
    neg_prompts = ["neg"] * 9
    loras = ["lora1"] * 9

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras, max_frames=81)
    print_test_result("Multiple Groups With Splits", result)


def test_whitespace_in_prompts():
    """Check if whitespace differences are handled correctly"""
    segment_frames = [20, 20, 20]
    prompts = ["A", " A", "A "]  # Different whitespace
    neg_prompts = ["neg"] * 3
    loras = ["lora1"] * 3

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("Whitespace in Prompts (should NOT consolidate)", result)


def test_empty_string_prompts():
    """Edge case: empty string prompts - should consolidate if all empty"""
    segment_frames = [20, 15, 25]
    prompts = ["", "", ""]
    neg_prompts = ["", "", ""]
    loras = ["", "", ""]

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("Empty String Prompts (should consolidate)", result)


def test_very_long_group():
    """Stress test: 50 segments with same prompt - should handle efficiently"""
    segment_frames = [10] * 50  # 500 total frames
    prompts = ["A"] * 50
    neg_prompts = ["neg"] * 50
    loras = ["lora1"] * 50

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras, max_frames=81)
    print_test_result("50 Segments Same Prompt (500 frames total)", result)


def test_sandwich_pattern():
    """Pattern: A-B-B-B-A - middle group should consolidate"""
    segment_frames = [20, 15, 20, 25, 20]
    prompts = ["A", "B", "B", "B", "A"]
    neg_prompts = ["neg"] * 5
    loras = ["lora1"] * 5

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("Sandwich Pattern A-B-B-B-A", result)


def test_off_by_one_frame_limit():
    """Test boundary: 80, 81, 82 frame groups"""
    # Group 1: 80 frames (under limit)
    # Group 2: 81 frames (exactly at limit)
    # Group 3: 82 frames (over limit)
    segment_frames = [40, 40, 40, 41, 41, 41]
    prompts = ["A", "A", "B", "B", "C", "C"]
    neg_prompts = ["neg"] * 6
    loras = ["lora1"] * 6

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras, max_frames=81)
    print_test_result("Boundary Test: 80, 81, 82 frame groups", result)


def test_repeated_group_pattern():
    """Pattern: AA-B-AA - same prompt appears in different positions"""
    segment_frames = [20, 20, 30, 20, 20]
    prompts = ["A", "A", "B", "A", "A"]
    neg_prompts = ["neg"] * 5
    loras = ["lora1"] * 5

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("Repeated Group Pattern AA-B-AA", result)


def test_many_tiny_groups():
    """Pattern: A-B-C-D-E-F-G-H - many single-segment groups"""
    segment_frames = [10, 12, 8, 15, 9, 11, 13, 10]
    prompts = ["A", "B", "C", "D", "E", "F", "G", "H"]
    neg_prompts = ["neg"] * 8
    loras = ["lora1"] * 8

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("Many Tiny Groups (8 different prompts)", result)


def test_case_sensitivity():
    """Check if 'A' and 'a' are treated as different"""
    segment_frames = [20, 20, 20]
    prompts = ["A", "a", "A"]
    neg_prompts = ["neg"] * 3
    loras = ["lora1"] * 3

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("Case Sensitivity Test (A vs a)", result)


def test_none_vs_empty_string():
    """Check None vs empty string handling"""
    segment_frames = [20, 20, 20, 20]
    prompts = [None, None, "", ""]
    neg_prompts = ["", None, None, ""]
    loras = ["lora1"] * 4

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("None vs Empty String Handling", result)


def test_multiple_params_differ():
    """When BOTH prompt and LoRA change at same time"""
    segment_frames = [20, 20, 20, 20]
    prompts = ["A", "A", "B", "B"]
    neg_prompts = ["neg"] * 4
    loras = ["lora1", "lora2", "lora2", "lora3"]

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("Multiple Params Differ Simultaneously", result)


def test_production_max_frames_65():
    """Production code uses max_frames=65, not 81"""
    segment_frames = [30, 30, 30]  # 90 total
    prompts = ["A"] * 3
    neg_prompts = ["neg"] * 3
    loras = ["lora1"] * 3

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras, max_frames=65)
    print_test_result("Production Max Frames = 65 (not 81)", result)


def test_pathological_100_one_frame():
    """Pathological: 100 segments of 1 frame each"""
    segment_frames = [1] * 100
    prompts = ["A"] * 100
    neg_prompts = ["neg"] * 100
    loras = ["lora1"] * 100

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras, max_frames=81)
    print_test_result("Pathological: 100 × 1-frame segments", result)


def test_real_prompt_punctuation():
    """Real-world prompt with subtle punctuation difference"""
    segment_frames = [25, 25, 25]
    prompts = [
        "a photorealistic portrait, 8k, highly detailed, cinematic lighting",
        "a photorealistic portrait, 8k, highly detailed, cinematic lighting",
        "a photorealistic portrait, 8k, highly detailed, cinematic lighting."  # period added
    ]
    neg_prompts = ["blurry, low quality"] * 3
    loras = ["realism_lora"] * 3

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("Real Prompt with Punctuation Difference", result)


def test_chunk_single_chunk_pattern():
    """Pattern: AAA-B-CCC (consolidatable chunk, single segment, consolidatable chunk)"""
    segment_frames = [15, 20, 15, 30, 20, 15, 25]
    prompts = ["A", "A", "A", "B", "C", "C", "C"]
    neg_prompts = ["neg"] * 7
    loras = ["lora1"] * 7

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("Chunk-Single-Chunk Pattern AAA-B-CCC", result)


def test_multiple_singles_between_chunks():
    """Pattern: AAA-B-C-D-EEE (chunk, multiple singles, chunk)"""
    segment_frames = [10, 15, 20, 25, 30, 35, 15, 20, 25]
    prompts = ["A", "A", "A", "B", "C", "D", "E", "E", "E"]
    neg_prompts = ["neg"] * 9
    loras = ["lora1"] * 9

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("Multiple Singles Between Chunks AAA-B-C-D-EEE", result)


def test_alternating_chunks_and_singles():
    """Pattern: AA-B-CC-D-EE (alternating chunks and singles)"""
    segment_frames = [20, 20, 30, 25, 25, 30, 20, 20]
    prompts = ["A", "A", "B", "C", "C", "D", "E", "E"]
    neg_prompts = ["neg"] * 8
    loras = ["lora1"] * 8

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("Alternating Chunks and Singles AA-B-CC-D-EE", result)


def test_large_chunk_small_single_large_chunk():
    """Pattern: AAAAAAA-B-CCCCCCC (large chunks with tiny single in between)"""
    segment_frames = [10, 10, 10, 10, 10, 10, 10, 5, 10, 10, 10, 10, 10, 10, 10]
    prompts = ["A"] * 7 + ["B"] + ["C"] * 7
    neg_prompts = ["neg"] * 15
    loras = ["lora1"] * 15

    result = apply_group_consolidation(segment_frames, prompts, neg_prompts, loras)
    print_test_result("Large Chunk-Tiny Single-Large Chunk", result)


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" GROUP-BASED CONSOLIDATION TEST SUITE - EXTENDED")
    print("="*80)

    # Original tests
    test_all_identical()
    test_all_different()
    test_mixed_groups()
    test_exceeds_frame_limit()
    test_alternating_prompts()
    test_single_segment()
    test_two_identical()
    test_large_many_small()
    test_neg_prompt_differs()
    test_lora_differs()
    test_realistic_timeline()

    # New edge case tests
    test_empty_input()
    test_exactly_81_frames()
    test_82_frames()
    test_many_tiny_segments()
    test_one_frame_segments()
    test_huge_single_segments()
    test_exact_frame_preservation()
    test_multiple_groups_with_splits()
    test_whitespace_in_prompts()
    test_empty_string_prompts()
    test_very_long_group()
    test_sandwich_pattern()
    test_off_by_one_frame_limit()

    # Critical production-readiness tests
    print("\n" + "="*80)
    print(" PRODUCTION-READINESS TESTS")
    print("="*80)
    test_repeated_group_pattern()
    test_many_tiny_groups()
    test_case_sensitivity()
    test_none_vs_empty_string()
    test_multiple_params_differ()
    test_production_max_frames_65()
    test_pathological_100_one_frame()
    test_real_prompt_punctuation()
    test_chunk_single_chunk_pattern()
    test_multiple_singles_between_chunks()
    test_alternating_chunks_and_singles()
    test_large_chunk_small_single_large_chunk()

    print("\n" + "="*80)
    print(" ALL TESTS COMPLETED")
    print("="*80 + "\n")
