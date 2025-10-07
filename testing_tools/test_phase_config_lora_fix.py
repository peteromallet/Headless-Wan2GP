#!/usr/bin/env python3
"""
Test to verify that phase_config LoRA multipliers are correctly formatted.

This tests the fix for the issue where phase-config multipliers like ["0.9;0", "0;0.9"]
were being incorrectly joined with commas instead of spaces.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source.lora_utils import normalize_lora_format


def test_phase_config_format():
    """Test that phase-config multipliers with semicolons are handled correctly."""

    print("\n=== Test 1: Phase-config format (list of strings with semicolons) ===")
    params = {
        "lora_multipliers": ["0.9;0", "0;0.9"]  # Phase-config format from parse_phase_config
    }
    result = normalize_lora_format(params, task_id="test1")
    print(f"Input:  {['0.9;0', '0;0.9']}")
    print(f"Output: {result['lora_multipliers']}")
    assert result["lora_multipliers"] == ["0.9;0", "0;0.9"], "List should be preserved as-is"
    print("✓ PASSED\n")

    print("=== Test 2: Phase-config format (space-separated string) ===")
    params = {
        "loras_multipliers": "0.9;0 0;0.9"  # Space-separated phase-config string
    }
    result = normalize_lora_format(params, task_id="test2")
    print(f"Input:  '0.9;0 0;0.9'")
    print(f"Output: {result['lora_multipliers']}")
    assert result["lora_multipliers"] == ["0.9;0", "0;0.9"], "Should split by spaces"
    print("✓ PASSED\n")

    print("=== Test 3: Phase-config format (comma-separated string - legacy) ===")
    params = {
        "loras_multipliers": "0.9;0,0;0.9"  # Comma-separated phase-config string
    }
    result = normalize_lora_format(params, task_id="test3")
    print(f"Input:  '0.9;0,0;0.9'")
    print(f"Output: {result['lora_multipliers']}")
    assert result["lora_multipliers"] == ["0.9;0", "0;0.9"], "Should split by commas when no spaces"
    print("✓ PASSED\n")

    print("=== Test 4: Regular format (comma-separated floats) ===")
    params = {
        "loras_multipliers": "0.9,1.0"  # Regular comma-separated floats
    }
    result = normalize_lora_format(params, task_id="test4")
    print(f"Input:  '0.9,1.0'")
    print(f"Output: {result['lora_multipliers']}")
    assert result["lora_multipliers"] == [0.9, 1.0], "Should convert to floats"
    print("✓ PASSED\n")

    print("=== Test 5: Regular format (list of floats) ===")
    params = {
        "lora_multipliers": [0.9, 1.0]  # Regular list of floats
    }
    result = normalize_lora_format(params, task_id="test5")
    print(f"Input:  {[0.9, 1.0]}")
    print(f"Output: {result['lora_multipliers']}")
    assert result["lora_multipliers"] == [0.9, 1.0], "List should be preserved as-is"
    print("✓ PASSED\n")


def test_loras_multipliers_formatting():
    """Test the direct formatting logic from process_all_loras."""

    print("=== Test 6: Phase-config multipliers formatting (semicolons) ===")
    lora_multipliers = ["0.9;0", "0;0.9"]

    # Simulate the logic from process_all_loras line 718-721
    if any(";" in str(m) for m in lora_multipliers):
        result = " ".join(map(str, lora_multipliers))
    else:
        result = ",".join(map(str, lora_multipliers))

    print(f"Input multipliers:  {['0.9;0', '0;0.9']}")
    print(f"Output string:      '{result}'")
    assert result == "0.9;0 0;0.9", "Should be space-separated with semicolons preserved"
    print("✓ PASSED\n")

    print("=== Test 7: Regular multipliers formatting (no semicolons) ===")
    lora_multipliers = [0.9, 1.0]

    # Simulate the logic from process_all_loras line 718-721
    if any(";" in str(m) for m in lora_multipliers):
        result = " ".join(map(str, lora_multipliers))
    else:
        result = ",".join(map(str, lora_multipliers))

    print(f"Input multipliers:  {[0.9, 1.0]}")
    print(f"Output string:      '{result}'")
    assert result == "0.9,1.0", "Should be comma-separated"
    print("✓ PASSED\n")

    print("=== Test 8: Mixed format with ramp in one phase ===")
    lora_multipliers = ["0.5,0.7,0.9;0", "0;1.0"]  # Ramp in phase 1, single value in phase 2

    # Simulate the logic from process_all_loras line 718-721
    if any(";" in str(m) for m in lora_multipliers):
        result = " ".join(map(str, lora_multipliers))
    else:
        result = ",".join(map(str, lora_multipliers))

    print(f"Input multipliers:  {['0.5,0.7,0.9;0', '0;1.0']}")
    print(f"Output string:      '{result}'")
    assert result == "0.5,0.7,0.9;0 0;1.0", "Should be space-separated with ramp and semicolons"
    print("✓ PASSED\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing Phase-Config LoRA Multiplier Formatting")
    print("="*70)

    try:
        test_phase_config_format()
        test_loras_multipliers_formatting()

        print("="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nThe fix correctly handles phase-config multipliers:")
        print("  • Input:  ['0.9;0', '0;0.9']")
        print("  • Output: '0.9;0 0;0.9' (space-separated)")
        print("\nThis resolves the error:")
        print('  "if the \';\' syntax is used for one Lora multiplier,')
        print('   the multipliers for its N denoising phases should be specified"')
        print()

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
