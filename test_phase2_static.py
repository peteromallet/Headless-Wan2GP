#!/usr/bin/env python3
"""
Phase 2 Static Validation Test

Validates code structure without running anything.
This test can run before implementation to guide development.
"""

import ast
import sys
from pathlib import Path
import inspect

def validate_task_type_mapping():
    """Check if _get_task_type_directory() function exists and has correct structure."""
    print("\n1. Validating task type mapping function...")

    # Check if function exists
    try:
        sys.path.insert(0, str(Path(__file__).parent / "source"))
        from common_utils import _get_task_type_directory
        print("   ✓ _get_task_type_directory() function found")
    except ImportError:
        print("   ✗ _get_task_type_directory() function NOT found")
        return False

    # Check function signature
    sig = inspect.signature(_get_task_type_directory)
    params = list(sig.parameters.keys())

    if 'task_type' in params:
        print("   ✓ Function accepts task_type parameter")
    else:
        print("   ✗ Function missing task_type parameter")
        return False

    # Test with some known task types
    expected_mappings = {
        'vace': 'generation/vace',
        't2v': 'generation/text_to_video',
        'flux': 'generation/flux',
        'join_clips_orchestrator': 'orchestrator_runs/join_clips',
        'travel_orchestrator': 'orchestrator_runs/travel',
        'image_inpaint': 'editing/inpaint',
        'extract_frame': 'specialized/frame_extraction',
    }

    all_correct = True
    for task_type, expected_dir in expected_mappings.items():
        try:
            actual_dir = _get_task_type_directory(task_type)
            if actual_dir == expected_dir:
                print(f"   ✓ {task_type} → {expected_dir}")
            else:
                print(f"   ✗ {task_type}: expected {expected_dir}, got {actual_dir}")
                all_correct = False
        except Exception as e:
            print(f"   ✗ {task_type}: error - {e}")
            all_correct = False

    # Test default/unknown task type
    try:
        default_dir = _get_task_type_directory('unknown_task_xyz')
        if default_dir in ['misc', 'default', '']:
            print(f"   ✓ Unknown task type → {default_dir} (default)")
        else:
            print(f"   ⚠ Unknown task type → {default_dir} (unexpected default)")
    except Exception as e:
        print(f"   ✗ Default handling error: {e}")
        all_correct = False

    return all_correct


def validate_prepare_output_path():
    """Check if prepare_output_path() has task_type parameter."""
    print("\n2. Validating prepare_output_path() function...")

    try:
        from source.common_utils import prepare_output_path
        print("   ✓ prepare_output_path() function found")
    except ImportError:
        print("   ✗ prepare_output_path() function NOT found")
        return False

    # Check function signature
    sig = inspect.signature(prepare_output_path)
    params = list(sig.parameters.keys())

    if 'task_type' in params:
        print("   ✓ Function has task_type parameter")

        # Check if it's optional (has default value)
        task_type_param = sig.parameters['task_type']
        if task_type_param.default != inspect.Parameter.empty:
            print(f"   ✓ task_type is optional (default: {task_type_param.default})")
        else:
            print("   ✗ task_type is required (should be optional for backwards compatibility)")
            return False
    else:
        print("   ✗ Function missing task_type parameter")
        return False

    return True


def validate_code_structure():
    """Validate overall code structure for Phase 2."""
    print("\n3. Validating code structure...")

    # Check common_utils.py exists
    common_utils_path = Path(__file__).parent / "source" / "common_utils.py"
    if common_utils_path.exists():
        print("   ✓ source/common_utils.py exists")
    else:
        print("   ✗ source/common_utils.py NOT found")
        return False

    # Parse the file to check structure
    try:
        with open(common_utils_path, 'r') as f:
            tree = ast.parse(f.read())

        # Find function definitions
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        required_functions = ['prepare_output_path', '_get_task_type_directory']
        for func_name in required_functions:
            if func_name in functions:
                print(f"   ✓ {func_name}() defined in common_utils.py")
            else:
                print(f"   ✗ {func_name}() NOT defined in common_utils.py")
                return False

    except Exception as e:
        print(f"   ✗ Error parsing common_utils.py: {e}")
        return False

    return True


def main():
    print("="*60)
    print("Phase 2 Static Validation Test")
    print("="*60)

    results = []

    # Run validation checks
    results.append(("Code Structure", validate_code_structure()))
    results.append(("Task Type Mapping", validate_task_type_mapping()))
    results.append(("prepare_output_path()", validate_prepare_output_path()))

    # Summary
    print("\n" + "="*60)
    print("Validation Results")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n✅ Phase 2 Static Validation: PASSED")
        print("\nAll code structure checks passed!")
        print("Ready to proceed with implementation.")
        return 0
    else:
        print("\n❌ Phase 2 Static Validation: FAILED")
        print("\nSome checks failed. Fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
