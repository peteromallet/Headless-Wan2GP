#!/usr/bin/env python3
"""
Timestep Analysis Script - Understand why 2-2-2 distribution becomes 4-1-1

This script simulates the exact timestep calculation logic used by the Euler scheduler
to understand the phase switching behavior.
"""

import numpy as np

def analyze_timestep_distribution(num_inference_steps=6, num_timesteps=1000, switch_threshold=667, switch2_threshold=333):
    """Simulate the exact timestep calculation and phase switching logic."""

    print(f"=== TIMESTEP DISTRIBUTION ANALYSIS ===")
    print(f"num_inference_steps: {num_inference_steps}")
    print(f"num_timesteps: {num_timesteps}")
    print(f"switch_threshold: {switch_threshold}")
    print(f"switch2_threshold: {switch2_threshold}")
    print()

    # This is the exact line from any2video.py for Euler scheduler:
    # timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
    timesteps = list(np.linspace(num_timesteps, 1, num_inference_steps, dtype=np.float32))
    timesteps.append(0.)

    print("Generated timesteps:")
    for i, t in enumerate(timesteps):
        print(f"  Step {i}: {t:.1f}")
    print()

    # Apply the phase switching logic from get_model_switch_steps()
    model_switch_step = model_switch_step2 = None
    guide_phases = 3

    for i, t in enumerate(timesteps):
        if guide_phases >= 2 and model_switch_step is None and t <= switch_threshold:
            model_switch_step = i
        if guide_phases >= 3 and model_switch_step2 is None and t <= switch2_threshold:
            model_switch_step2 = i

    if model_switch_step is None:
        model_switch_step = num_inference_steps
    if model_switch_step2 is None:
        model_switch_step2 = num_inference_steps

    print("Phase switching analysis:")
    print(f"  model_switch_step (Phase 1→2): {model_switch_step}")
    print(f"  model_switch_step2 (Phase 2→3): {model_switch_step2}")
    print()

    # Calculate phase distribution
    phase1_steps = model_switch_step
    phase2_steps = model_switch_step2 - model_switch_step
    phase3_steps = num_inference_steps - model_switch_step2

    print("Phase distribution:")
    print(f"  Phase 1 steps: 0-{model_switch_step-1} = {phase1_steps} steps")
    print(f"  Phase 2 steps: {model_switch_step}-{model_switch_step2-1} = {phase2_steps} steps")
    print(f"  Phase 3 steps: {model_switch_step2}-{num_inference_steps-1} = {phase3_steps} steps")
    print(f"  Distribution: {phase1_steps}-{phase2_steps}-{phase3_steps}")
    print()

    # Show which timesteps trigger switches AND simulate the actual execution
    print("Switch trigger analysis:")
    print("Simulating actual execution with update_guidance calls:")

    # Simulate the actual execution loop
    guidance_switch_done = guidance_switch2_done = False
    current_phase = "1/3 High Noise"

    for i, t in enumerate(timesteps[:-1]):  # Exclude the final 0.0
        # This simulates the update_guidance calls in any2video.py
        step_started_with_phase = current_phase

        # First update_guidance call (for phase 2)
        if guide_phases >= 2 and not guidance_switch_done and t <= switch_threshold:
            guidance_switch_done = True
            current_phase = "2/3 High Noise"

        # Second update_guidance call (for phase 3)
        if guide_phases >= 3 and not guidance_switch2_done and t <= switch2_threshold:
            guidance_switch2_done = True
            current_phase = "3/3 Low Noise"

        switch_info = ""
        if step_started_with_phase != current_phase:
            switch_info = f" ← SWITCHED from {step_started_with_phase} to {current_phase}"

        print(f"  Step {i}: t={t:.1f} → Phase {current_phase}{switch_info}")

    return phase1_steps, phase2_steps, phase3_steps

def find_correct_thresholds_for_222():
    """Find thresholds that would give us 2-2-2 distribution."""
    print("\n=== FINDING CORRECT THRESHOLDS FOR 2-2-2 ===")

    num_inference_steps = 6
    num_timesteps = 1000

    # Generate timesteps
    timesteps = list(np.linspace(num_timesteps, 1, num_inference_steps, dtype=np.float32))
    timesteps.append(0.)

    print("Timesteps:", [f"{t:.1f}" for t in timesteps[:-1]])

    # For 2-2-2 distribution:
    # Phase 1: steps 0-1 (switch after step 1, so at step 2)
    # Phase 2: steps 2-3 (switch after step 3, so at step 4)
    # Phase 3: steps 4-5

    target_switch_step = 2  # Switch to phase 2 at step 2
    target_switch_step2 = 4  # Switch to phase 3 at step 4

    switch_threshold = timesteps[target_switch_step] + 0.1  # Just above timestep at step 2
    switch2_threshold = timesteps[target_switch_step2] + 0.1  # Just above timestep at step 4

    print(f"\nFor 2-2-2 distribution, we need:")
    print(f"  switch_threshold > {timesteps[target_switch_step]:.1f} (step 2 timestep)")
    print(f"  switch2_threshold > {timesteps[target_switch_step2]:.1f} (step 4 timestep)")
    print(f"  Suggested switch_threshold: {switch_threshold:.0f}")
    print(f"  Suggested switch2_threshold: {switch2_threshold:.0f}")

    return int(switch_threshold), int(switch2_threshold)

if __name__ == "__main__":
    # Analyze current configuration
    print("Current configuration (667, 333):")
    analyze_timestep_distribution(6, 1000, 667, 333)

    # Find correct thresholds
    correct_thresh, correct_thresh2 = find_correct_thresholds_for_222()

    print(f"\nTesting corrected thresholds ({correct_thresh}, {correct_thresh2}):")
    analyze_timestep_distribution(6, 1000, correct_thresh, correct_thresh2)