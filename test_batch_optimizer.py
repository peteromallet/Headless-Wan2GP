#!/usr/bin/env python3
"""
Production-like test suite for batch optimizer.

This comprehensive test suite validates the batching system against realistic scenarios
and edge cases. It runs iteratively until all tests pass, providing detailed analysis
of mask application, frame mapping, and efficiency gains.

Usage:
    python test_batch_optimizer.py --loop-until-perfect --verbose
"""

import sys
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse
from dataclasses import asdict

# Add source directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "source"))

# Mock the logging utils to avoid import issues in test environment
class MockLogger:
    def debug(self, msg, task_id="unknown"): print(f"[DEBUG {task_id}] {msg}")
    def error(self, msg, task_id="unknown"): print(f"[ERROR {task_id}] {msg}")
    def warning(self, msg, task_id="unknown"): print(f"[WARN {task_id}] {msg}")
    def essential(self, msg, task_id="unknown"): print(f"[INFO {task_id}] {msg}")

# Patch the logger before importing
import sys
sys.modules['source.logging_utils'] = type('MockModule', (), {'travel_logger': MockLogger()})()

from source.batch_optimizer import (
    calculate_optimal_batching, 
    create_batch_mask_analysis,
    validate_batching_integrity,
    BatchGroup,
    BatchingAnalysis
)


class BatchTestScenario:
    """Represents a test scenario with expected outcomes."""
    
    def __init__(self, name: str, segment_frames: List[int], overlaps: List[int], 
                 prompts: List[str], negative_prompts: List[str],
                 expected_batches: int = None, should_batch: bool = None):
        self.name = name
        self.segment_frames = segment_frames
        self.overlaps = overlaps
        self.prompts = prompts
        self.negative_prompts = negative_prompts
        self.expected_batches = expected_batches
        self.should_batch = should_batch
    
    def total_original_frames(self) -> int:
        return sum(self.segment_frames) - sum(self.overlaps)


def create_test_scenarios() -> List[BatchTestScenario]:
    """Create comprehensive test scenarios covering edge cases and typical usage."""
    
    scenarios = [
        # Scenario 1: Your exact example (0→25→54 frames)
        BatchTestScenario(
            name="User Example: 0→25→54",
            segment_frames=[25, 29, 25],  # 25, 29 (25+4), 25 frames
            overlaps=[4, 4],
            prompts=["travel from start to image 1", "continue to image 2", "reach final destination"],
            negative_prompts=["blurry", "blurry", "blurry"],
            expected_batches=1,  # Should fit in single batch
            should_batch=True
        ),
        
        # Scenario 2: Small segments that should batch well
        BatchTestScenario(
            name="Small Multi-Segment",
            segment_frames=[17, 17, 17, 17, 17],  # 5 segments of 17 frames each
            overlaps=[2, 2, 2, 2],
            prompts=["seg1", "seg2", "seg3", "seg4", "seg5"],
            negative_prompts=["neg1", "neg2", "neg3", "neg4", "neg5"],
            expected_batches=1,  # Total: 85-8=77 frames, should fit in one batch
            should_batch=True
        ),
        
        # Scenario 3: Mixed sizes requiring intelligent splitting
        BatchTestScenario(
            name="Mixed Sizes",
            segment_frames=[45, 25, 33, 21],  # Different sized segments
            overlaps=[8, 6, 4],
            prompts=["large seg", "medium seg", "medium seg", "small seg"],
            negative_prompts=["neg"] * 4,
            expected_batches=2,  # Should split intelligently
            should_batch=True
        ),
        
        # Scenario 4: Too few segments (shouldn't batch)
        BatchTestScenario(
            name="Too Few Segments",
            segment_frames=[40, 35],  # Only 2 segments
            overlaps=[5],
            prompts=["start", "end"],
            negative_prompts=["neg1", "neg2"],
            expected_batches=2,  # Same as original
            should_batch=False
        ),
        
        # Scenario 5: Large segments (shouldn't batch effectively)
        BatchTestScenario(
            name="Large Segments",
            segment_frames=[73, 73, 73],  # Each segment near max limit
            overlaps=[10, 10],
            prompts=["large1", "large2", "large3"],
            negative_prompts=["neg"] * 3,
            expected_batches=3,  # Each should be individual
            should_batch=False
        ),
        
        # Scenario 6: Edge case - exactly at limit
        BatchTestScenario(
            name="Exactly At Limit",
            segment_frames=[41, 45],  # 41+45-5 = 81 frames exactly
            overlaps=[5],
            prompts=["part1", "part2"],
            negative_prompts=["neg1", "neg2"],
            expected_batches=1,
            should_batch=True
        ),
        
        # Scenario 7: Many tiny segments
        BatchTestScenario(
            name="Many Tiny Segments",
            segment_frames=[9] * 8,  # 8 segments of 9 frames each
            overlaps=[1] * 7,
            prompts=[f"tiny{i}" for i in range(8)],
            negative_prompts=["neg"] * 8,
            expected_batches=1,  # 72-7=65 frames, should fit in one
            should_batch=True
        ),
        
        # Scenario 8: Single segment (edge case)
        BatchTestScenario(
            name="Single Segment",
            segment_frames=[45],
            overlaps=[],
            prompts=["only segment"],
            negative_prompts=["neg"],
            expected_batches=1,
            should_batch=False
        ),
        
        # Scenario 9: Zero overlaps
        BatchTestScenario(
            name="Zero Overlaps",
            segment_frames=[20, 20, 20, 20],
            overlaps=[0, 0, 0],
            prompts=["seg1", "seg2", "seg3", "seg4"],
            negative_prompts=["neg"] * 4,
            expected_batches=1,  # 80 frames total, should fit
            should_batch=True
        ),
        
        # Scenario 10: Realistic production case
        BatchTestScenario(
            name="Realistic Production",
            segment_frames=[25, 29, 25, 29, 25],  # Typical travel journey
            overlaps=[6, 6, 6, 6],
            prompts=[
                "start at sunset beach",
                "travel through forest path", 
                "arrive at mountain peak",
                "descend to valley",
                "end at peaceful lake"
            ],
            negative_prompts=["blurry, low quality"] * 5,
            expected_batches=1,  # Should batch well
            should_batch=True
        )
    ]
    
    return scenarios


def test_scenario(scenario: BatchTestScenario, verbose: bool = False) -> Dict[str, Any]:
    """Test a single scenario and return detailed results."""
    
    print(f"\n{'='*60}")
    print(f"Testing: {scenario.name}")
    print(f"{'='*60}")
    print(f"Segments: {scenario.segment_frames}")
    print(f"Overlaps: {scenario.overlaps}")
    print(f"Total original frames: {scenario.total_original_frames()}")
    
    # Run batching analysis
    start_time = time.time()
    analysis = calculate_optimal_batching(
        segment_frames_expanded=scenario.segment_frames,
        frame_overlap_expanded=scenario.overlaps,
        base_prompts_expanded=scenario.prompts,
        negative_prompts_expanded=scenario.negative_prompts,
        task_id=f"test_{scenario.name.replace(' ', '_')}"
    )
    analysis_time = time.time() - start_time
    
    # Validate integrity
    validation = validate_batching_integrity(
        analysis, 
        scenario.segment_frames, 
        scenario.overlaps,
        task_id=f"validation_{scenario.name}"
    )
    
    # Analyze masks for each batch
    mask_analyses = []
    for batch_group in analysis.batch_groups:
        mask_analysis = create_batch_mask_analysis(
            batch_group,
            scenario.segment_frames,
            scenario.overlaps,
            task_id=f"mask_{scenario.name}_{batch_group.batch_index}"
        )
        mask_analyses.append(mask_analysis)
    
    # Check expectations
    results = {
        "scenario_name": scenario.name,
        "analysis_time_ms": analysis_time * 1000,
        "batching_analysis": asdict(analysis),
        "validation_result": validation,
        "mask_analyses": mask_analyses,
        "expectations_met": True,
        "expectation_failures": []
    }
    
    # Validate expectations
    if scenario.expected_batches is not None:
        if len(analysis.batch_groups) != scenario.expected_batches:
            results["expectations_met"] = False
            results["expectation_failures"].append(
                f"Expected {scenario.expected_batches} batches, got {len(analysis.batch_groups)}"
            )
    
    if scenario.should_batch is not None:
        if analysis.should_use_batching != scenario.should_batch:
            results["expectations_met"] = False
            results["expectation_failures"].append(
                f"Expected should_batch={scenario.should_batch}, got {analysis.should_use_batching}"
            )
    
    # Display results
    print(f"Analysis time: {analysis_time*1000:.2f}ms")
    print(f"Should use batching: {analysis.should_use_batching}")
    print(f"Efficiency gain: {analysis.efficiency_gain:.2f}x")
    print(f"Batches created: {len(analysis.batch_groups)}")
    print(f"Reason: {analysis.reason}")
    
    if verbose:
        print(f"\nDetailed batch breakdown:")
        for i, batch in enumerate(analysis.batch_groups):
            print(f"  Batch {i}: segments {batch.segment_indices} -> {batch.total_frames} frames")
            print(f"    Overlaps: {batch.internal_overlaps}")
            print(f"    Prompt: {batch.combined_prompt[:50]}...")
    
    print(f"\nMask Analysis Summary:")
    for i, mask_analysis in enumerate(mask_analyses):
        ma = mask_analysis
        print(f"  Batch {i}: {ma['anchored_frames']} anchored, {ma['transition_frames']} transition, {ma['free_frames']} free")
        print(f"    Anchor %: {ma['anchor_percentage']:.1f}%")
        
        if verbose:
            print(f"    Frame-by-frame mask map:")
            for frame_idx in sorted(ma['frame_mask_map'].keys())[:10]:  # Show first 10 frames
                mask_val = ma['frame_mask_map'][frame_idx]
                source = ma['frame_source_map'][frame_idx]
                print(f"      Frame {frame_idx}: mask={mask_val:.1f} ({source})")
            if len(ma['frame_mask_map']) > 10:
                print(f"      ... and {len(ma['frame_mask_map'])-10} more frames")
    
    # Validation results
    print(f"\nValidation: {'✓ PASSED' if validation['is_valid'] else '✗ FAILED'}")
    if validation['issues']:
        print(f"Issues: {validation['issues']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    
    # Expectation results
    if results["expectations_met"]:
        print(f"Expectations: ✓ ALL MET")
    else:
        print(f"Expectations: ✗ FAILED")
        for failure in results["expectation_failures"]:
            print(f"  - {failure}")
    
    return results


def run_stress_test(num_iterations: int = 100, verbose: bool = False) -> Dict[str, Any]:
    """Run stress test with random scenarios."""
    
    print(f"\n{'='*60}")
    print(f"STRESS TEST: {num_iterations} random scenarios")
    print(f"{'='*60}")
    
    stress_results = {
        "total_iterations": num_iterations,
        "successful_analyses": 0,
        "failed_analyses": 0,
        "validation_failures": 0,
        "total_time": 0,
        "avg_time_per_analysis": 0,
        "efficiency_gains": []
    }
    
    start_time = time.time()
    
    for i in range(num_iterations):
        try:
            # Generate random scenario
            num_segments = random.randint(2, 8)
            segment_frames = [random.randint(9, 73) for _ in range(num_segments)]
            overlaps = [random.randint(0, min(segment_frames[j], segment_frames[j+1])//2) 
                       for j in range(num_segments-1)]
            prompts = [f"random_prompt_{j}" for j in range(num_segments)]
            negative_prompts = ["random_negative"] * num_segments
            
            scenario = BatchTestScenario(
                name=f"Stress_{i}",
                segment_frames=segment_frames,
                overlaps=overlaps,
                prompts=prompts,
                negative_prompts=negative_prompts
            )
            
            # Run analysis
            analysis = calculate_optimal_batching(
                segment_frames_expanded=scenario.segment_frames,
                frame_overlap_expanded=scenario.overlaps,
                base_prompts_expanded=scenario.prompts,
                negative_prompts_expanded=scenario.negative_prompts,
                task_id=f"stress_{i}"
            )
            
            # Validate
            validation = validate_batching_integrity(
                analysis, 
                scenario.segment_frames, 
                scenario.overlaps,
                task_id=f"stress_validation_{i}"
            )
            
            stress_results["successful_analyses"] += 1
            stress_results["efficiency_gains"].append(analysis.efficiency_gain)
            
            if not validation["is_valid"]:
                stress_results["validation_failures"] += 1
                if verbose:
                    print(f"Stress {i}: Validation failed - {validation['issues']}")
            
        except Exception as e:
            stress_results["failed_analyses"] += 1
            if verbose:
                print(f"Stress {i}: Analysis failed - {e}")
    
    total_time = time.time() - start_time
    stress_results["total_time"] = total_time
    stress_results["avg_time_per_analysis"] = total_time / num_iterations
    
    # Calculate statistics
    if stress_results["efficiency_gains"]:
        gains = stress_results["efficiency_gains"]
        stress_results["avg_efficiency_gain"] = sum(gains) / len(gains)
        stress_results["max_efficiency_gain"] = max(gains)
        stress_results["min_efficiency_gain"] = min(gains)
    
    print(f"Stress test completed:")
    print(f"  Successful: {stress_results['successful_analyses']}/{num_iterations}")
    print(f"  Failed: {stress_results['failed_analyses']}/{num_iterations}")
    print(f"  Validation failures: {stress_results['validation_failures']}")
    print(f"  Avg time per analysis: {stress_results['avg_time_per_analysis']*1000:.2f}ms")
    if stress_results["efficiency_gains"]:
        print(f"  Avg efficiency gain: {stress_results['avg_efficiency_gain']:.2f}x")
        print(f"  Max efficiency gain: {stress_results['max_efficiency_gain']:.2f}x")
    
    return stress_results


def main():
    parser = argparse.ArgumentParser(description="Test batch optimizer in production-like environment")
    parser.add_argument("--loop-until-perfect", action="store_true", 
                       help="Keep running tests until all pass perfectly")
    parser.add_argument("--verbose", action="store_true", 
                       help="Show detailed output")
    parser.add_argument("--stress-test", type=int, default=0,
                       help="Run stress test with N random scenarios")
    parser.add_argument("--max-iterations", type=int, default=10,
                       help="Maximum iterations for loop-until-perfect")
    
    args = parser.parse_args()
    
    scenarios = create_test_scenarios()
    iteration = 1
    all_passed = False
    
    while not all_passed and iteration <= args.max_iterations:
        print(f"\n{'#'*80}")
        print(f"TEST ITERATION {iteration}")
        print(f"{'#'*80}")
        
        all_results = []
        all_expectations_met = True
        all_validations_passed = True
        
        for scenario in scenarios:
            result = test_scenario(scenario, verbose=args.verbose)
            all_results.append(result)
            
            if not result["expectations_met"]:
                all_expectations_met = False
            
            if not result["validation_result"]["is_valid"]:
                all_validations_passed = False
        
        # Summary
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration} SUMMARY")
        print(f"{'='*60}")
        
        total_scenarios = len(scenarios)
        passed_expectations = sum(1 for r in all_results if r["expectations_met"])
        passed_validations = sum(1 for r in all_results if r["validation_result"]["is_valid"])
        
        print(f"Scenarios tested: {total_scenarios}")
        print(f"Expectations met: {passed_expectations}/{total_scenarios}")
        print(f"Validations passed: {passed_validations}/{total_scenarios}")
        
        avg_analysis_time = sum(r["analysis_time_ms"] for r in all_results) / len(all_results)
        print(f"Average analysis time: {avg_analysis_time:.2f}ms")
        
        # Calculate efficiency statistics
        batching_scenarios = [r for r in all_results if r["batching_analysis"]["should_use_batching"]]
        if batching_scenarios:
            efficiency_gains = [r["batching_analysis"]["efficiency_gain"] for r in batching_scenarios]
            avg_gain = sum(efficiency_gains) / len(efficiency_gains)
            max_gain = max(efficiency_gains)
            print(f"Batching efficiency: {len(batching_scenarios)} scenarios, avg {avg_gain:.2f}x gain, max {max_gain:.2f}x")
        
        all_passed = all_expectations_met and all_validations_passed
        
        if all_passed:
            print(f"\n🎉 ALL TESTS PASSED PERFECTLY! 🎉")
            break
        elif not args.loop_until_perfect:
            break
        else:
            print(f"\n⚠️  Some tests failed. Iteration {iteration+1} starting in 2 seconds...")
            time.sleep(2)
            iteration += 1
    
    # Stress test if requested
    if args.stress_test > 0:
        stress_results = run_stress_test(args.stress_test, args.verbose)
        
        # Check stress test results
        success_rate = stress_results["successful_analyses"] / stress_results["total_iterations"]
        validation_rate = (stress_results["successful_analyses"] - stress_results["validation_failures"]) / stress_results["total_iterations"]
        
        print(f"\nStress test results:")
        print(f"  Success rate: {success_rate*100:.1f}%")
        print(f"  Validation rate: {validation_rate*100:.1f}%")
        
        if success_rate >= 0.95 and validation_rate >= 0.95:
            print(f"✅ Stress test PASSED (≥95% success and validation rates)")
        else:
            print(f"❌ Stress test FAILED (<95% success or validation rate)")
    
    # Final verdict
    if all_passed:
        print(f"\n🟢 FINAL VERDICT: BATCH OPTIMIZER IS PRODUCTION READY! 🟢")
        return 0
    else:
        print(f"\n🔴 FINAL VERDICT: BATCH OPTIMIZER NEEDS FIXES 🔴")
        return 1


if __name__ == "__main__":
    sys.exit(main())
