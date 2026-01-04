#!/usr/bin/env python3
"""
Test script for phase_multiplier_utils

Validates parsing, conversion, and filtering of phase-based LoRA multipliers.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from source.phase_multiplier_utils import (
    is_lightning_lora,
    parse_phase_multiplier,
    convert_to_phase_format,
    format_phase_multipliers,
    get_phase_loras
)


def test_lightning_detection():
    """Test Lightning LoRA auto-detection."""
    print("=" * 70)
    print("Test 1: Lightning LoRA Detection")
    print("=" * 70)

    test_cases = [
        ("Qwen-Image-Edit-Lightning-8steps.safetensors", True),
        ("style_transfer.safetensors", False),
        ("Wan2.2-Lightning_T2V-v1.1.safetensors", True),
        ("detail_enhancer.safetensors", False),
        ("distill_lora.safetensors", True),
        ("accelerator_v2.safetensors", True),
    ]

    for lora_name, expected in test_cases:
        result = is_lightning_lora(lora_name)
        status = "✅" if result == expected else "❌"
        print(f"{status} {lora_name}: {result} (expected {expected})")

    print()


def test_parse_phase_multiplier():
    """Test parsing of phase-based multiplier strings."""
    print("=" * 70)
    print("Test 2: Parse Phase Multipliers")
    print("=" * 70)

    test_cases = [
        ("1.0", 2, [1.0, 1.0]),           # Simple format
        ("1.0;0.5", 2, [1.0, 0.5]),       # Phase format
        ("1.0;0", 2, [1.0, 0.0]),         # Disable in pass 2
        ("0;0.8", 2, [0.0, 0.8]),         # Enable only in pass 2
        ("1.0;", 2, [1.0, 0.0]),          # Missing value
        ("1.0;0.5;0.3", 3, [1.0, 0.5, 0.3]),  # 3-phase
    ]

    for mult_str, num_phases, expected in test_cases:
        try:
            result, valid = parse_phase_multiplier(mult_str, num_phases)
            status = "✅" if result == expected else "❌"
            print(f"{status} '{mult_str}' ({num_phases} phases): {result}")
        except Exception as e:
            print(f"❌ '{mult_str}': ERROR - {e}")

    print()


def test_convert_to_phase_format():
    """Test conversion from simple to phase-based format."""
    print("=" * 70)
    print("Test 3: Convert to Phase Format")
    print("=" * 70)

    test_cases = [
        ("1.0", "style.safetensors", "1.0;1.0"),          # Standard LoRA
        ("1.0", "Lightning-8steps.safetensors", "1.0;0"), # Lightning LoRA
        ("1.1", "style_transfer.safetensors", "1.1;1.1"), # Standard with different strength
        ("1.0;0.5", "any.safetensors", "1.0;0.5"),        # Already in phase format
    ]

    for mult, lora_name, expected in test_cases:
        result = convert_to_phase_format(mult, lora_name, num_phases=2)
        status = "✅" if result == expected else "❌"
        print(f"{status} {lora_name} ({mult}): {result} (expected {expected})")

    print()


def test_format_phase_multipliers():
    """Test batch conversion of multiplier lists."""
    print("=" * 70)
    print("Test 4: Format Multiple Multipliers")
    print("=" * 70)

    lora_names = [
        "Qwen-Image-Edit-Lightning-8steps.safetensors",
        "style_transfer_qwen_edit_2.safetensors",
        "in_scene_different_object.safetensors"
    ]

    multipliers = ["1.0", "1.1", "0.5"]

    expected = ["1.0;0", "1.1;1.1", "0.5;0.5"]

    result = format_phase_multipliers(lora_names, multipliers, num_phases=2)

    print(f"Input LoRAs:")
    for name in lora_names:
        print(f"  - {name}")

    print(f"\nInput Multipliers: {multipliers}")
    print(f"Expected Output:   {expected}")
    print(f"Actual Output:     {result}")

    status = "✅" if result == expected else "❌"
    print(f"\n{status} Conversion {'PASSED' if result == expected else 'FAILED'}")

    print()


def test_get_phase_loras():
    """Test filtering LoRAs by phase."""
    print("=" * 70)
    print("Test 5: Filter LoRAs by Phase")
    print("=" * 70)

    lora_names = [
        "Lightning-8steps.safetensors",
        "style.safetensors",
        "detail.safetensors"
    ]

    multipliers = ["1.0;0", "1.1;1.2", "0;0.8"]

    # Test Pass 1 (index 0)
    print("Pass 1 (index 0):")
    pass1_loras, pass1_mults = get_phase_loras(lora_names, multipliers, phase_index=0, num_phases=2)
    print(f"  LoRAs:       {pass1_loras}")
    print(f"  Multipliers: {pass1_mults}")
    print(f"  Expected:    ['Lightning-8steps.safetensors', 'style.safetensors'] with ['1.0', '1.1']")

    # Test Pass 2 (index 1)
    print("\nPass 2 (index 1):")
    pass2_loras, pass2_mults = get_phase_loras(lora_names, multipliers, phase_index=1, num_phases=2)
    print(f"  LoRAs:       {pass2_loras}")
    print(f"  Multipliers: {pass2_mults}")
    print(f"  Expected:    ['style.safetensors', 'detail.safetensors'] with ['1.2', '0.8']")

    # Validate results
    pass1_ok = (
        pass1_loras == ["Lightning-8steps.safetensors", "style.safetensors"] and
        pass1_mults == ["1.0", "1.1"]
    )
    pass2_ok = (
        pass2_loras == ["style.safetensors", "detail.safetensors"] and
        pass2_mults == ["1.2", "0.8"]
    )

    status = "✅" if (pass1_ok and pass2_ok) else "❌"
    print(f"\n{status} Phase filtering {'PASSED' if (pass1_ok and pass2_ok) else 'FAILED'}")

    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Phase Multiplier Utilities Test Suite")
    print("=" * 70)
    print()

    test_lightning_detection()
    test_parse_phase_multiplier()
    test_convert_to_phase_format()
    test_format_phase_multipliers()
    test_get_phase_loras()

    print("=" * 70)
    print("All Tests Complete")
    print("=" * 70)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
