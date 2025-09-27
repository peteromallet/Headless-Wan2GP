#!/usr/bin/env python3
"""
Phase Distribution Predictor for WanGP
=====================================

This script predicts the actual phase distribution for WanGP experiments
based on the real timestep generation and switching logic found in the codebase.

Key Implementation Details from WanGP:
- Timesteps: np.linspace(1000, 1, sampling_steps) + [0]
- Phase switching: if t <= switch_threshold: switch_phase()
- Switch happens at the step where timestep first becomes <= threshold
"""

import numpy as np
import argparse
from typing import List, Tuple, Dict

def generate_wangp_timesteps(num_steps: int) -> List[float]:
    """
    Generate timesteps exactly as WanGP does in any2video.py:417-422

    WanGP Code:
    timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
    timesteps.append(0.)
    """
    timesteps = list(np.linspace(1000, 1, num_steps, dtype=np.float32))
    timesteps.append(0.0)
    return timesteps

def predict_phase_distribution(num_steps: int, switch_threshold: float, switch_threshold2: float = None) -> Dict:
    """
    Predict phase distribution based on WanGP's actual switching logic

    Args:
        num_steps: Number of inference steps
        switch_threshold: First phase switch threshold (0-1000)
        switch_threshold2: Second phase switch threshold (0-1000), for 3-phase models

    Returns:
        Dictionary containing timesteps, switch points, and phase distribution
    """
    timesteps = generate_wangp_timesteps(num_steps)

    # Find switch points (following loras_multipliers.py:114-115 logic)
    switch_step_1 = None
    switch_step_2 = None

    for i, t in enumerate(timesteps[:-1]):  # Exclude the final 0.0 timestep from switching logic
        if switch_step_1 is None and t <= switch_threshold:
            switch_step_1 = i
        if switch_threshold2 is not None and switch_step_2 is None and t <= switch_threshold2:
            switch_step_2 = i

    # Calculate phase distribution
    if switch_threshold2 is not None:
        # 3-phase model
        phase_1_steps = switch_step_1 if switch_step_1 is not None else num_steps
        phase_2_steps = (switch_step_2 - switch_step_1) if (switch_step_2 is not None and switch_step_1 is not None) else (num_steps - phase_1_steps)
        phase_3_steps = num_steps - phase_1_steps - phase_2_steps
        distribution = [phase_1_steps, phase_2_steps, phase_3_steps]
        phases = 3
    else:
        # 2-phase model
        phase_1_steps = switch_step_1 if switch_step_1 is not None else num_steps
        phase_2_steps = num_steps - phase_1_steps
        distribution = [phase_1_steps, phase_2_steps]
        phases = 2

    return {
        'num_steps': num_steps,
        'timesteps': timesteps,
        'switch_threshold': switch_threshold,
        'switch_threshold2': switch_threshold2,
        'switch_step_1': switch_step_1,
        'switch_step_2': switch_step_2,
        'distribution': distribution,
        'phases': phases
    }

def analyze_experiment_config(config: Dict) -> Dict:
    """
    Analyze a specific experiment configuration

    Args:
        config: Dictionary containing experiment parameters

    Returns:
        Analysis results
    """
    return predict_phase_distribution(
        config['num_inference_steps'],
        config['switch_threshold'],
        config.get('switch_threshold2')
    )

def print_analysis(result: Dict):
    """Print detailed analysis of phase distribution"""
    print(f"\n=== Phase Distribution Analysis ===")
    print(f"Steps: {result['num_steps']}")
    print(f"Thresholds: {result['switch_threshold']}" +
          (f", {result['switch_threshold2']}" if result['switch_threshold2'] else ""))
    print(f"Phases: {result['phases']}")

    print(f"\nTimesteps:")
    for i, t in enumerate(result['timesteps']):
        marker = ""
        if result['switch_step_1'] is not None and i == result['switch_step_1']:
            marker += " ← Phase 1→2 switch"
        if result['switch_step_2'] is not None and i == result['switch_step_2']:
            marker += " ← Phase 2→3 switch"
        print(f"  Step {i:2d}: {t:6.1f}{marker}")

    print(f"\nPhase Distribution: {'-'.join(map(str, result['distribution']))}")
    for i, steps in enumerate(result['distribution'], 1):
        print(f"  Phase {i}: {steps} steps")

    return result

def verify_against_log_data():
    """Verify predictions against known log behavior"""
    print("=== Verification Against Log Data ===")

    # Test 1: 6-step experiments (currently running)
    # Log shows: 2-2-2 distribution for 6 steps
    # Phase switches at steps 2 and 4
    print("\n--- Test 1: 6-Step Experiment ---")
    config_6step = {
        'num_inference_steps': 6,
        'switch_threshold': 667,  # Should switch at step 2
        'switch_threshold2': 333  # Should switch at step 4
    }
    result_6 = analyze_experiment_config(config_6step)
    print_analysis(result_6)
    expected_6 = [2, 2, 2]
    success_6 = result_6['distribution'] == expected_6
    if success_6:
        print("✅ 6-STEP VERIFICATION PASSED!")
    else:
        print(f"❌ 6-STEP VERIFICATION FAILED: Expected {expected_6}, got {result_6['distribution']}")

    # Test 2: 9-step experiments from earlier logs
    # Log showed: Phase switches at steps 3 and 7 -> 3-4-2 distribution
    print("\n--- Test 2: 9-Step Experiment ---")
    config_9step = {
        'num_inference_steps': 9,
        'switch_threshold': 625,  # Should switch at step 3
        'switch_threshold2': 250  # Should switch at step 7
    }
    result_9 = analyze_experiment_config(config_9step)
    print_analysis(result_9)
    expected_9 = [3, 4, 2]
    success_9 = result_9['distribution'] == expected_9
    if success_9:
        print("✅ 9-STEP VERIFICATION PASSED!")
    else:
        print(f"❌ 9-STEP VERIFICATION FAILED: Expected {expected_9}, got {result_9['distribution']}")

    return success_6 and success_9

def recommend_thresholds(target_distribution: List[int], num_steps: int = None) -> Dict:
    """
    Recommend thresholds to achieve target phase distribution

    Args:
        target_distribution: Desired steps per phase [phase1, phase2, ...]
        num_steps: Number of inference steps (auto-inferred from distribution if not provided)

    Returns:
        Recommended thresholds and analysis
    """
    # Auto-infer num_steps if not provided
    if num_steps is None:
        num_steps = sum(target_distribution)
        print(f"Auto-inferred {num_steps} steps from distribution {target_distribution}")

    # Validate that the distribution matches num_steps
    if sum(target_distribution) != num_steps:
        raise ValueError(f"Target distribution {target_distribution} sums to {sum(target_distribution)}, but {num_steps} steps specified")

    timesteps = generate_wangp_timesteps(num_steps)
    recommendations = {}
    recommendations['num_steps'] = num_steps

    if len(target_distribution) == 2:
        # 2-phase: find threshold for phase 1 → 2 switch
        target_step = target_distribution[0]
        if target_step < num_steps:
            recommended_threshold = timesteps[target_step] + 1  # Just above the timestep
            recommendations['switch_threshold'] = recommended_threshold

    elif len(target_distribution) == 3:
        # 3-phase: find thresholds for both switches
        phase1_steps = target_distribution[0]
        phase2_steps = target_distribution[1]

        # First switch (phase 1 → 2)
        if phase1_steps < num_steps:
            threshold1 = timesteps[phase1_steps] + 1
            recommendations['switch_threshold'] = threshold1

        # Second switch (phase 2 → 3)
        switch2_step = phase1_steps + phase2_steps
        if switch2_step < num_steps:
            threshold2 = timesteps[switch2_step] + 1
            recommendations['switch_threshold2'] = threshold2

    return recommendations

def main():
    parser = argparse.ArgumentParser(description='Predict WanGP phase distributions')
    parser.add_argument('--steps', type=int, default=9, help='Number of inference steps')
    parser.add_argument('--threshold1', type=float, default=900, help='First switch threshold')
    parser.add_argument('--threshold2', type=float, help='Second switch threshold (for 3-phase)')
    parser.add_argument('--verify', action='store_true', help='Run verification against log data')
    parser.add_argument('--recommend', help='Recommend thresholds for target distribution (e.g., "3,3,3")')

    args = parser.parse_args()

    if args.verify:
        verify_against_log_data()
        return

    if args.recommend:
        target = [int(x) for x in args.recommend.split(',')]
        # Use args.steps only if explicitly provided, otherwise auto-infer
        explicit_steps = args.steps if 'steps' in vars(args) and args.steps != 9 else None  # 9 is the default
        recommendations = recommend_thresholds(target, explicit_steps)

        actual_steps = recommendations['num_steps']
        print(f"\n=== Threshold Recommendations for {'-'.join(map(str, target))} Distribution ===")
        print(f"For {actual_steps} steps:")
        for key, value in recommendations.items():
            if key != 'num_steps':
                if isinstance(value, float):
                    print(f"  {key}: {value:.1f}")
                else:
                    print(f"  {key}: {value}")

        # Test the recommendations
        if 'switch_threshold' in recommendations:
            test_result = predict_phase_distribution(
                actual_steps,
                recommendations['switch_threshold'],
                recommendations.get('switch_threshold2')
            )
            print(f"\nPredicted result with these thresholds:")
            print_analysis(test_result)
    else:
        # Standard analysis
        result = predict_phase_distribution(args.steps, args.threshold1, args.threshold2)
        print_analysis(result)

if __name__ == '__main__':
    main()