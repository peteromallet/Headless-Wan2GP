#!/usr/bin/env python3
"""
Debug Detailed Phase Switching - Trace the EXACT execution flow

This script simulates the exact execution from any2video.py to see
why we get 4-1-1 instead of 2-2-2 phase distribution.
"""

import numpy as np

def debug_exact_execution_flow():
    """Debug the exact execution flow from any2video.py lines 875-877."""

    print("=== DEBUGGING EXACT EXECUTION FLOW ===")

    # Test parameters
    num_timesteps = 1000
    sampling_steps = 6
    switch_threshold = 667
    switch2_threshold = 333
    guide_phases = 3
    model_switch_phase = 2

    # Generate timesteps exactly as in any2video.py
    timesteps = list(np.linspace(num_timesteps, 1, sampling_steps, dtype=np.float32))
    timesteps.append(0.)

    print("Parameters:")
    print(f"  switch_threshold: {switch_threshold}")
    print(f"  switch2_threshold: {switch2_threshold}")
    print(f"  guide_phases: {guide_phases}")
    print(f"  model_switch_phase: {model_switch_phase}")
    print()

    print("Generated timesteps:")
    for i, t in enumerate(timesteps):
        print(f"  Step {i}: {t:.1f}")
    print()

    # Simulate the exact denoising loop from any2video.py
    print("=== SIMULATING EXACT DENOISING LOOP ===")

    guidance_switch_done = False
    guidance_switch2_done = False
    current_trans = "model1"

    for i, t in enumerate(timesteps[:-1]):  # Skip final 0.0
        print(f"\nStep {i}: timestep = {t:.1f}")

        # Store initial state
        initial_switch_done = guidance_switch_done
        initial_switch2_done = guidance_switch2_done
        initial_trans = current_trans

        # First update_guidance call - line 876
        # guide_scale, guidance_switch_done, trans, denoising_extra =
        #   update_guidance(i, t, guide_scale, guide2_scale, guidance_switch_done, switch_threshold, trans, 2, denoising_extra)
        print(f"  First update_guidance call (phase 2):")
        print(f"    Checking: guide_phases({guide_phases}) >= 2 and not guidance_switch_done({guidance_switch_done}) and t({t:.1f}) <= switch_threshold({switch_threshold})")

        if guide_phases >= 2 and not guidance_switch_done and t <= switch_threshold:
            guidance_switch_done = True
            print(f"    ✅ PHASE 2 SWITCH TRIGGERED!")
            # Check model switch: model_switch_phase == phase_no-1 (2-1=1)
            if model_switch_phase == 1:  # phase_no-1 where phase_no=2
                current_trans = "model2"
                print(f"    📋 Model switched to model2")
        else:
            print(f"    ❌ No switch")

        # Second update_guidance call - line 877
        # guide_scale, guidance_switch2_done, trans, denoising_extra =
        #   update_guidance(i, t, guide_scale, guide3_scale, guidance_switch2_done, switch2_threshold, trans, 3, denoising_extra)
        print(f"  Second update_guidance call (phase 3):")
        print(f"    Checking: guide_phases({guide_phases}) >= 3 and not guidance_switch2_done({guidance_switch2_done}) and t({t:.1f}) <= switch2_threshold({switch2_threshold})")

        if guide_phases >= 3 and not guidance_switch2_done and t <= switch2_threshold:
            guidance_switch2_done = True
            print(f"    ✅ PHASE 3 SWITCH TRIGGERED!")
            # Check model switch: model_switch_phase == phase_no-1 (3-1=2)
            if model_switch_phase == 2:  # phase_no-1 where phase_no=3
                current_trans = "model2"
                print(f"    📋 Model switched to model2")
        else:
            print(f"    ❌ No switch")

        # Determine current phase based on switch states
        if guidance_switch2_done:
            current_phase = "3/3"
            noise_type = "Low Noise" if current_trans == "model2" else "High Noise"
        elif guidance_switch_done:
            current_phase = "2/3"
            noise_type = "Low Noise" if current_trans == "model2" else "High Noise"
        else:
            current_phase = "1/3"
            noise_type = "Low Noise" if current_trans == "model2" else "High Noise"

        print(f"  📊 Result: Phase {current_phase} {noise_type} (using {current_trans})")

        # Show changes
        if initial_switch_done != guidance_switch_done:
            print(f"  🔄 guidance_switch_done: {initial_switch_done} → {guidance_switch_done}")
        if initial_switch2_done != guidance_switch2_done:
            print(f"  🔄 guidance_switch2_done: {initial_switch2_done} → {guidance_switch2_done}")
        if initial_trans != current_trans:
            print(f"  🔄 model: {initial_trans} → {current_trans}")

    print(f"\n=== FINAL ANALYSIS ===")
    print("Based on the simulation above:")

    # Count phases
    guidance_switch_done = False
    guidance_switch2_done = False
    phase_counts = {"1/3": 0, "2/3": 0, "3/3": 0}

    for i, t in enumerate(timesteps[:-1]):
        # Apply same logic
        if guide_phases >= 2 and not guidance_switch_done and t <= switch_threshold:
            guidance_switch_done = True
        if guide_phases >= 3 and not guidance_switch2_done and t <= switch2_threshold:
            guidance_switch2_done = True

        # Count phase
        if guidance_switch2_done:
            phase_counts["3/3"] += 1
        elif guidance_switch_done:
            phase_counts["2/3"] += 1
        else:
            phase_counts["1/3"] += 1

    print(f"Phase distribution:")
    print(f"  Phase 1/3: {phase_counts['1/3']} steps")
    print(f"  Phase 2/3: {phase_counts['2/3']} steps")
    print(f"  Phase 3/3: {phase_counts['3/3']} steps")
    print(f"  Pattern: {phase_counts['1/3']}-{phase_counts['2/3']}-{phase_counts['3/3']}")

if __name__ == "__main__":
    debug_exact_execution_flow()