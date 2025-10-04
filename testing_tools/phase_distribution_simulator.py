#!/usr/bin/env python3
"""
Phase Distribution Simulator for WanGP with Flow Shift Support
==============================================================

This script accurately simulates phase distribution accounting for the
timestep_transform function with flow_shift parameter used in euler solver.

Key Implementation Details from WanGP any2video.py:
- Line 474: timesteps = list(np.linspace(1000, 1, sampling_steps))
- Line 475: timesteps.append(0.)
- Line 478: timesteps = [timestep_transform(t, shift=shift) for t in timesteps][:-1]
- Line 879-880: Phase switches when t <= threshold

The flow_shift dramatically changes timestep values, which affects when
phase switches occur. This script accounts for that transformation.
"""

import numpy as np
import argparse
from typing import List, Tuple, Dict
import json


def timestep_transform(t: float, shift: float = 5.0, num_timesteps: int = 1000) -> float:
    """
    Exact timestep transform function from any2video.py line 52-57

    Args:
        t: Original timestep value (0-1000)
        shift: Flow shift parameter (default 5.0)
        num_timesteps: Maximum timestep value (default 1000)

    Returns:
        Transformed timestep value
    """
    t = t / num_timesteps
    # shift the timestep based on ratio
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t


def generate_timesteps(num_steps: int, flow_shift: float = 5.0, use_transform: bool = True) -> List[float]:
    """
    Generate timesteps exactly as WanGP does for euler solver

    Args:
        num_steps: Number of inference steps
        flow_shift: Flow shift parameter (only used if use_transform=True)
        use_transform: Whether to apply timestep_transform (euler uses this)

    Returns:
        List of timestep values (length = num_steps, excludes final 0.0)
    """
    # Line 474-475: Generate base timesteps
    timesteps = list(np.linspace(1000, 1, num_steps, dtype=np.float32))
    timesteps.append(0.)

    if use_transform:
        # Line 478: Transform and exclude final 0.0
        timesteps = [timestep_transform(t, shift=flow_shift, num_timesteps=1000)
                     for t in timesteps][:-1]
    else:
        # For non-euler solvers (unipc, dpm++), timesteps may be different
        # but for this simulator we'll just use the base timesteps
        timesteps = timesteps[:-1]  # Exclude final 0.0

    return timesteps


def simulate_phase_distribution(
    num_steps: int,
    switch_threshold: float,
    switch_threshold2: float = None,
    flow_shift: float = 5.0,
    sample_solver: str = "euler"
) -> Dict:
    """
    Simulate phase distribution based on WanGP's actual switching logic

    Args:
        num_steps: Number of inference steps
        switch_threshold: First phase switch threshold (0-1000)
        switch_threshold2: Second phase switch threshold (0-1000), for 3-phase
        flow_shift: Flow shift parameter (default 5.0)
        sample_solver: Solver type ("euler", "unipc", "dpm++")

    Returns:
        Dictionary containing timesteps, switch points, and phase distribution
    """
    # Generate timesteps (euler uses transform, others may not)
    use_transform = (sample_solver == "euler")
    timesteps = generate_timesteps(num_steps, flow_shift, use_transform)

    # Find switch points (following any2video.py line 879-880 logic)
    switch_step_1 = None
    switch_step_2 = None

    for i, t in enumerate(timesteps):
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
        'phases': phases,
        'flow_shift': flow_shift,
        'sample_solver': sample_solver,
        'use_transform': use_transform
    }


def print_analysis(result: Dict, verbose: bool = True):
    """Print detailed analysis of phase distribution"""
    print(f"\n{'='*70}")
    print(f"PHASE DISTRIBUTION ANALYSIS")
    print(f"{'='*70}")
    print(f"Steps: {result['num_steps']}")
    print(f"Sample Solver: {result['sample_solver']}")
    print(f"Flow Shift: {result['flow_shift']}")
    print(f"Timestep Transform: {'Yes' if result['use_transform'] else 'No'}")
    print(f"Thresholds: {result['switch_threshold']}" +
          (f", {result['switch_threshold2']}" if result['switch_threshold2'] else ""))
    print(f"Phases: {result['phases']}")

    if verbose:
        print(f"\nTimesteps:")
        for i, t in enumerate(result['timesteps']):
            marker = ""
            if result['switch_step_1'] is not None and i == result['switch_step_1']:
                marker += " ← Phase 1→2 switch"
            if result['switch_step_2'] is not None and i == result['switch_step_2']:
                marker += " ← Phase 2→3 switch"
            print(f"  Step {i:2d}: {t:7.1f}{marker}")

    print(f"\n{'='*70}")
    print(f"DISTRIBUTION: {'-'.join(map(str, result['distribution']))}")
    print(f"{'='*70}")
    for i, steps in enumerate(result['distribution'], 1):
        phase_range_start = sum(result['distribution'][:i-1])
        phase_range_end = phase_range_start + steps - 1
        print(f"  Phase {i}: steps {phase_range_start}-{phase_range_end} ({steps} steps)")
    print(f"{'='*70}")


def recommend_thresholds(
    target_distribution: List[int],
    flow_shift: float = 5.0,
    sample_solver: str = "euler"
) -> Dict:
    """
    Recommend thresholds to achieve target phase distribution

    Args:
        target_distribution: Desired steps per phase [phase1, phase2, ...]
        flow_shift: Flow shift parameter
        sample_solver: Solver type

    Returns:
        Recommended thresholds and verification
    """
    num_steps = sum(target_distribution)

    # Generate timesteps
    use_transform = (sample_solver == "euler")
    timesteps = generate_timesteps(num_steps, flow_shift, use_transform)

    recommendations = {
        'num_steps': num_steps,
        'flow_shift': flow_shift,
        'sample_solver': sample_solver
    }

    if len(target_distribution) >= 2:
        # First switch (phase 1 → 2)
        target_step = target_distribution[0]
        if target_step < num_steps:
            # Threshold should be just above the timestep at this step
            recommended_threshold = int(timesteps[target_step] + 1)
            recommendations['switch_threshold'] = recommended_threshold

    if len(target_distribution) >= 3:
        # Second switch (phase 2 → 3)
        target_step = target_distribution[0] + target_distribution[1]
        if target_step < num_steps:
            recommended_threshold = int(timesteps[target_step] + 1)
            recommendations['switch_threshold2'] = recommended_threshold

    return recommendations


def analyze_config_file(config_path: str):
    """Analyze a WanGP config file"""
    with open(config_path, 'r') as f:
        config = json.load(f)

    num_steps = config.get('num_inference_steps', 6)
    switch_threshold = config.get('switch_threshold', 0)
    switch_threshold2 = config.get('switch_threshold2', None)
    flow_shift = config.get('flow_shift', 5.0)
    sample_solver = config.get('sample_solver', 'euler')

    print(f"\nAnalyzing config: {config_path}")
    print(f"Model: {config.get('model', {}).get('name', 'Unknown')}")

    result = simulate_phase_distribution(
        num_steps, switch_threshold, switch_threshold2,
        flow_shift, sample_solver
    )

    print_analysis(result, verbose=True)

    # Check if name matches actual distribution
    model_name = config.get('model', {}).get('name', '')
    if '2-2-2' in model_name:
        expected = [2, 2, 2]
        if result['distribution'] == expected:
            print(f"\n✅ Config matches name: 2-2-2 distribution verified!")
        else:
            actual = '-'.join(map(str, result['distribution']))
            print(f"\n❌ WARNING: Config name says '2-2-2' but actual distribution is {actual}!")
            print(f"\nTo fix, update thresholds to:")
            recs = recommend_thresholds(expected, flow_shift, sample_solver)
            print(f"  switch_threshold: {recs.get('switch_threshold')}")
            print(f"  switch_threshold2: {recs.get('switch_threshold2')}")


def main():
    parser = argparse.ArgumentParser(
        description='Simulate WanGP phase distributions with flow_shift support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simulate with specific parameters
  python phase_distribution_simulator.py --steps 6 --threshold1 883 --threshold2 558 --shift 5

  # Recommend thresholds for target distribution
  python phase_distribution_simulator.py --recommend 2,2,2 --shift 5

  # Analyze a config file
  python phase_distribution_simulator.py --config ../Wan2GP/defaults/lightning_baseline_2_2_2.json
        """
    )
    parser.add_argument('--steps', type=int, default=6, help='Number of inference steps')
    parser.add_argument('--threshold1', type=float, help='First switch threshold')
    parser.add_argument('--threshold2', type=float, help='Second switch threshold (for 3-phase)')
    parser.add_argument('--shift', type=float, default=5.0, help='Flow shift parameter (default: 5.0)')
    parser.add_argument('--solver', type=str, default='euler', choices=['euler', 'unipc', 'dpm++'],
                        help='Sample solver type (default: euler)')
    parser.add_argument('--recommend', help='Recommend thresholds for target distribution (e.g., "2,2,2")')
    parser.add_argument('--config', help='Path to WanGP config JSON file to analyze')
    parser.add_argument('--verbose', action='store_true', help='Show detailed timestep breakdown')

    args = parser.parse_args()

    if args.config:
        analyze_config_file(args.config)
        return

    if args.recommend:
        target = [int(x) for x in args.recommend.split(',')]
        recommendations = recommend_thresholds(target, args.shift, args.solver)

        print(f"\n{'='*70}")
        print(f"THRESHOLD RECOMMENDATIONS FOR {'-'.join(map(str, target))} DISTRIBUTION")
        print(f"{'='*70}")
        print(f"Steps: {recommendations['num_steps']}")
        print(f"Flow Shift: {recommendations['flow_shift']}")
        print(f"Sample Solver: {recommendations['sample_solver']}")
        print(f"\nRecommended thresholds:")
        if 'switch_threshold' in recommendations:
            print(f"  switch_threshold: {recommendations['switch_threshold']}")
        if 'switch_threshold2' in recommendations:
            print(f"  switch_threshold2: {recommendations['switch_threshold2']}")

        # Verify recommendations
        print(f"\nVerification:")
        result = simulate_phase_distribution(
            recommendations['num_steps'],
            recommendations.get('switch_threshold', 0),
            recommendations.get('switch_threshold2'),
            recommendations['flow_shift'],
            recommendations['sample_solver']
        )
        print_analysis(result, verbose=args.verbose)

    else:
        # Standard simulation
        if args.threshold1 is None:
            print("Error: Must specify --threshold1, --recommend, or --config")
            parser.print_help()
            return

        result = simulate_phase_distribution(
            args.steps, args.threshold1, args.threshold2,
            args.shift, args.solver
        )
        print_analysis(result, verbose=args.verbose)


if __name__ == '__main__':
    main()
