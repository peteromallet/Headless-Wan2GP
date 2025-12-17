#!/usr/bin/env python3
"""
Phase 1 Static Validation

This validates the Phase 1 changes without requiring WGP dependencies.
It checks that the parameter flow is correct through the call chain.
"""

import ast
import sys
from pathlib import Path

def validate_parameter_flow():
    """
    Validate that main_output_dir parameter flows correctly through:
    worker.py → HeadlessTaskQueue → WanOrchestrator → wgp.server_config
    """
    print("\n" + "="*60)
    print("Phase 1 Static Validation (No Dependencies Required)")
    print("="*60 + "\n")

    errors = []
    warnings = []

    # 1. Check worker.py passes main_output_dir to HeadlessTaskQueue
    print("1. Checking worker.py → HeadlessTaskQueue parameter passing...")
    worker_file = Path("worker.py")
    if worker_file.exists():
        content = worker_file.read_text()
        if 'main_output_dir=str(main_output_dir)' in content:
            print("   ✓ worker.py passes main_output_dir to HeadlessTaskQueue")
        else:
            errors.append("worker.py does not pass main_output_dir to HeadlessTaskQueue")
            print("   ✗ Missing parameter passing")
    else:
        errors.append("worker.py not found")

    # 2. Check HeadlessTaskQueue accepts and stores main_output_dir
    print("\n2. Checking HeadlessTaskQueue parameter handling...")
    hmm_file = Path("headless_model_management.py")
    if hmm_file.exists():
        content = hmm_file.read_text()

        # Check __init__ signature
        if 'def __init__(self, wan_dir: str, max_workers: int = 1, debug_mode: bool = False, main_output_dir: Optional[str] = None):' in content:
            print("   ✓ HeadlessTaskQueue.__init__ accepts main_output_dir parameter")
        else:
            errors.append("HeadlessTaskQueue.__init__ signature missing main_output_dir")
            print("   ✗ Missing parameter in __init__ signature")

        # Check storage
        if 'self.main_output_dir = main_output_dir' in content:
            print("   ✓ HeadlessTaskQueue stores main_output_dir as instance variable")
        else:
            warnings.append("HeadlessTaskQueue may not store main_output_dir")
            print("   ⚠ Parameter storage unclear")

        # Check passing to WanOrchestrator
        if 'self.orchestrator = WanOrchestrator(self.wan_dir, main_output_dir=self.main_output_dir)' in content:
            print("   ✓ HeadlessTaskQueue passes main_output_dir to WanOrchestrator")
        else:
            errors.append("HeadlessTaskQueue does not pass main_output_dir to WanOrchestrator")
            print("   ✗ Missing parameter passing to WanOrchestrator")
    else:
        errors.append("headless_model_management.py not found")

    # 3. Check WanOrchestrator accepts and uses main_output_dir
    print("\n3. Checking WanOrchestrator parameter handling...")
    wgp_file = Path("headless_wgp.py")
    if wgp_file.exists():
        content = wgp_file.read_text()

        # Check __init__ signature
        if 'def __init__(self, wan_root: str, main_output_dir: Optional[str] = None):' in content:
            print("   ✓ WanOrchestrator.__init__ accepts main_output_dir parameter")
        else:
            errors.append("WanOrchestrator.__init__ signature missing main_output_dir")
            print("   ✗ Missing parameter in __init__ signature")

        # Check save_path configuration
        if "'save_path': absolute_outputs_path" in content:
            print("   ✓ WanOrchestrator sets wgp.server_config['save_path']")
        else:
            errors.append("WanOrchestrator does not set save_path")
            print("   ✗ Missing save_path configuration")

        # Check image_save_path configuration (CRITICAL FIX)
        if "'image_save_path': absolute_outputs_path" in content:
            print("   ✓ WanOrchestrator sets wgp.server_config['image_save_path'] (CRITICAL FIX)")
        else:
            errors.append("WanOrchestrator does not set image_save_path (videos will work, images won't!)")
            print("   ✗ CRITICAL: Missing image_save_path configuration")
    else:
        errors.append("headless_wgp.py not found")

    # 4. Check backwards compatibility
    print("\n4. Checking backwards compatibility...")
    all_files_check = all([
        'main_output_dir: Optional[str] = None' in Path(f).read_text()
        for f in ['headless_model_management.py', 'headless_wgp.py']
        if Path(f).exists()
    ])
    if all_files_check:
        print("   ✓ All parameters have Optional with None defaults (backwards compatible)")
    else:
        warnings.append("Backwards compatibility may be affected")
        print("   ⚠ Parameter defaults unclear")

    # Report results
    print("\n" + "="*60)
    print("Validation Results")
    print("="*60)

    if not errors and not warnings:
        print("\n✅ ALL CHECKS PASSED")
        print("\nPhase 1 implementation is correct:")
        print("  • Parameter flow: worker.py → HeadlessTaskQueue → WanOrchestrator ✓")
        print("  • Both save_path (videos) and image_save_path (images) configured ✓")
        print("  • Backwards compatible (optional parameters with None defaults) ✓")
        print("\n🎯 Ready for production validation tomorrow!")
        return True
    else:
        if errors:
            print(f"\n❌ {len(errors)} ERROR(S) FOUND:")
            for error in errors:
                print(f"  • {error}")
        if warnings:
            print(f"\n⚠️  {len(warnings)} WARNING(S):")
            for warning in warnings:
                print(f"  • {warning}")
        print("\n🔴 Phase 1 has issues - review before production deployment")
        return False

if __name__ == "__main__":
    success = validate_parameter_flow()
    sys.exit(0 if success else 1)
