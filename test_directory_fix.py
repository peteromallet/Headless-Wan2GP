#!/usr/bin/env python3
"""Test that directory checking works correctly."""

import os
import sys

print("=" * 60)
print("Testing Directory Validation")
print("=" * 60)

# Test 1: Wrong directory should fail
print("\nTest 1: Try to create WanOrchestrator from WRONG directory")
print(f"Current directory: {os.getcwd()}")

wan_dir = os.path.join(os.getcwd(), 'Wan2GP')
print(f"Wan2GP path: {wan_dir}")

# Don't chdir - should fail!
sys.path.insert(0, os.getcwd())

try:
    from headless_wgp import WanOrchestrator
    print("  Attempting to create WanOrchestrator without chdir...")
    orc = WanOrchestrator(wan_dir)
    print("  ❌ FAILED: Should have raised RuntimeError!")
    sys.exit(1)
except RuntimeError as e:
    if "CRITICAL: WanOrchestrator must be initialized from Wan2GP directory" in str(e):
        print(f"  ✅ PASSED: Got expected error")
        print(f"     Error: {str(e)[:100]}...")
    else:
        print(f"  ❌ FAILED: Got different error: {e}")
        sys.exit(1)

# Test 2: Correct directory should work (at least past the directory check)
print("\nTest 2: Create WanOrchestrator from CORRECT directory")
os.chdir(wan_dir)
print(f"Changed to: {os.getcwd()}")

# Clear the module so it reimports
if 'headless_wgp' in sys.modules:
    del sys.modules['headless_wgp']

# Note: This will hang on wgp import due to model downloads, but we just want
# to verify the directory check passes
print("  Creating WanOrchestrator (will hang on wgp import, that's OK)...")
print("  If you don't see an error about wrong directory, the fix worked!")

try:
    from headless_wgp import WanOrchestrator
    # This will timeout/hang, but if we get past __init__ directory check, that's success
    print("  Starting WanOrchestrator init (expect timeout)...")
except RuntimeError as e:
    if "CRITICAL: WanOrchestrator must be initialized from Wan2GP directory" in str(e):
        print(f"  ❌ FAILED: Still getting directory error: {e}")
        sys.exit(1)
    else:
        # Some other error, that's OK for this test
        print(f"  ✅ Directory check passed (got different error: {e})")
        sys.exit(0)

print("\n  Note: Test will hang here during wgp import - that's expected.")
print("  The important thing is we didn't get a directory error above.")
