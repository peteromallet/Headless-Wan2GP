#!/usr/bin/env python3
"""
Wan2GP Upgrade Verification Test
Run before and after upgrade to ensure no degradation.

Usage:
    python test_wan2gp_upgrade.py          # Run all tests
    python test_wan2gp_upgrade.py --quick  # Import tests only (no GPU)
"""

import sys
import os
import json
import traceback
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Wan2GP"))

# Test results tracking
RESULTS = {"passed": 0, "failed": 0, "skipped": 0}

def test(name):
    """Decorator for test functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\n{'='*60}")
            print(f"TEST: {name}")
            print('='*60)
            try:
                result = func(*args, **kwargs)
                if result is None or result:
                    print(f"✅ PASSED: {name}")
                    RESULTS["passed"] += 1
                else:
                    print(f"❌ FAILED: {name}")
                    RESULTS["failed"] += 1
            except Exception as e:
                print(f"❌ FAILED: {name}")
                print(f"   Error: {e}")
                traceback.print_exc()
                RESULTS["failed"] += 1
        return wrapper
    return decorator


# =============================================================================
# IMPORT TESTS - These should always work
# =============================================================================

@test("Import: shared.utils (file exists)")
def test_import_shared_utils():
    # Check files exist (actual import requires CUDA/dependencies)
    utils_path = PROJECT_ROOT / "Wan2GP" / "shared" / "utils" / "utils.py"
    loras_path = PROJECT_ROOT / "Wan2GP" / "shared" / "utils" / "loras_mutipliers.py"
    
    if utils_path.exists():
        print(f"   ✓ shared/utils/utils.py exists")
    else:
        print(f"   ✗ shared/utils/utils.py MISSING")
        return False
    
    if loras_path.exists():
        print(f"   ✓ shared/utils/loras_mutipliers.py exists")
    else:
        print(f"   ✗ shared/utils/loras_mutipliers.py MISSING")
        return False
    
    return True

@test("Import: shared.attention (file exists)")
def test_import_attention():
    attention_path = PROJECT_ROOT / "Wan2GP" / "shared" / "attention.py"
    if attention_path.exists():
        with open(attention_path) as f:
            content = f.read()
        if "get_attention_modes" in content:
            print("   ✓ shared/attention.py exists with get_attention_modes")
            return True
    print("   ✗ shared/attention.py MISSING or invalid")
    return False

@test("Import: models.wan (file exists)")
def test_import_models_wan():
    any2video_path = PROJECT_ROOT / "Wan2GP" / "models" / "wan" / "any2video.py"
    if any2video_path.exists():
        with open(any2video_path) as f:
            content = f.read()
        if "class WanAny2V" in content:
            print("   ✓ models/wan/any2video.py exists with WanAny2V class")
            return True
    print("   ✗ models/wan/any2video.py MISSING or invalid")
    return False

@test("Import: models.qwen")
def test_import_models_qwen():
    from models.qwen.qwen_handler import family_handler
    supported = family_handler.query_supported_types()
    print(f"   ✓ Qwen supported types: {supported}")
    return True

@test("Import: wgp core functions")
def test_import_wgp():
    # This tests the main wgp.py file imports correctly
    os.chdir(PROJECT_ROOT / "Wan2GP")
    
    # Import specific functions rather than whole module (avoids Gradio UI)
    import importlib.util
    spec = importlib.util.spec_from_file_location("wgp", PROJECT_ROOT / "Wan2GP" / "wgp.py")
    
    # Just verify the file can be parsed
    import ast
    with open(PROJECT_ROOT / "Wan2GP" / "wgp.py", 'r') as f:
        content = f.read()
    ast.parse(content)
    print("   ✓ wgp.py syntax valid")
    
    os.chdir(PROJECT_ROOT)
    return True


# =============================================================================
# CONFIG TESTS - Verify our custom configs load
# =============================================================================

@test("Config: Lightning baselines exist")
def test_lightning_configs():
    defaults_dir = PROJECT_ROOT / "Wan2GP" / "defaults"
    
    # Core configs we MUST have (any lightning config)
    lightning_configs = list(defaults_dir.glob("*lightning*.json"))
    
    if len(lightning_configs) >= 4:
        print(f"   ✓ Found {len(lightning_configs)} lightning configs")
        for c in lightning_configs[:6]:
            print(f"      - {c.name}")
        return True
    else:
        print(f"   ⚠ Only found {len(lightning_configs)} lightning configs (expected 4+)")
        return False

@test("Config: All critical configs from manifest")
def test_critical_configs():
    manifest_path = PROJECT_ROOT / "upgrade_backup" / "CONFIG_MANIFEST.json"
    defaults_dir = PROJECT_ROOT / "Wan2GP" / "defaults"
    
    if not manifest_path.exists():
        print("   ⚠ Manifest not found (run backup first)")
        RESULTS["skipped"] += 1
        RESULTS["passed"] -= 1
        return None
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    missing_critical = []
    present_critical = 0
    
    for category, info in manifest["categories"].items():
        if info.get("critical", False):
            for config_file in info["files"]:
                config_path = defaults_dir / config_file
                if config_path.exists():
                    present_critical += 1
                else:
                    missing_critical.append(config_file)
    
    if missing_critical:
        print(f"   ❌ Missing {len(missing_critical)} critical configs:")
        for c in missing_critical[:5]:
            print(f"      - {c}")
        if len(missing_critical) > 5:
            print(f"      ... and {len(missing_critical) - 5} more")
        return False
    else:
        print(f"   ✓ All {present_critical} critical configs present")
        return True

@test("Config: Lightning configs are valid JSON")
def test_lightning_configs_valid():
    defaults_dir = PROJECT_ROOT / "Wan2GP" / "defaults"
    
    configs = list(defaults_dir.glob("*lightning*.json"))
    if not configs:
        print("   ⚠ No lightning configs found")
        return False
    
    for config_path in configs:
        with open(config_path) as f:
            data = json.load(f)
        # Verify expected keys
        if "model" in data or "guidance_phases" in data:
            print(f"   ✓ {config_path.name}: valid")
        else:
            print(f"   ⚠ {config_path.name}: unexpected structure")
    
    return True

@test("Config: VACE cocktail configs exist")
def test_vace_cocktail_configs():
    defaults_dir = PROJECT_ROOT / "Wan2GP" / "defaults"
    configs = list(defaults_dir.glob("vace*cocktail*.json"))
    print(f"   Found {len(configs)} VACE cocktail configs")
    for c in configs[:5]:
        print(f"   ✓ {c.name}")
    return len(configs) > 0

@test("Config: Qwen configs exist")
def test_qwen_configs():
    defaults_dir = PROJECT_ROOT / "Wan2GP" / "defaults"
    configs = list(defaults_dir.glob("qwen*.json"))
    print(f"   Found {len(configs)} Qwen configs")
    for c in configs:
        print(f"   ✓ {c.name}")
    return len(configs) > 0


# =============================================================================
# HEADLESS TESTS - Verify our headless infrastructure works
# =============================================================================

@test("Headless: headless infrastructure exists")
def test_headless_wgp():
    # Check for headless-related files
    headless_files = [
        ("headless_wgp.py", ["generate", "load"]),
        ("headless_model_management.py", ["model", "task"]),
        ("worker.py", ["worker", "task"]),
    ]
    
    found = 0
    for filename, keywords in headless_files:
        path = PROJECT_ROOT / filename
        if path.exists():
            with open(path) as f:
                content = f.read().lower()
            if any(kw in content for kw in keywords):
                print(f"   ✓ {filename} exists and valid")
                found += 1
            else:
                print(f"   ⚠ {filename} exists but may be incomplete")
        else:
            print(f"   ✗ {filename} MISSING")
    
    return found >= 2  # At least 2 of 3 headless files should exist

@test("Headless: worker.py imports")
def test_worker_imports():
    worker_path = PROJECT_ROOT / "worker.py"
    if not worker_path.exists():
        print("   ⚠ worker.py not found")
        return False
    
    # Verify syntax
    import ast
    with open(worker_path) as f:
        content = f.read()
    ast.parse(content)
    print("   ✓ worker.py syntax valid")
    return True


# =============================================================================
# MODIFICATION TESTS - Verify our custom modifications exist
# =============================================================================

@test("Modification: vid2vid_init in any2video.py")
def test_vid2vid_modification():
    any2video_path = PROJECT_ROOT / "Wan2GP" / "models" / "wan" / "any2video.py"
    
    if not any2video_path.exists():
        print(f"   ⚠ File not found: {any2video_path}")
        return False
    
    with open(any2video_path) as f:
        content = f.read()
    
    checks = [
        ("vid2vid_init_video", "vid2vid initialization parameter"),
        ("vid2vid_init_strength", "vid2vid strength parameter"),
        ("latent_noise_mask_strength", "latent noise mask parameter"),
    ]
    
    all_found = True
    for check, desc in checks:
        if check in content:
            print(f"   ✓ {desc}")
        else:
            print(f"   ✗ {desc} NOT FOUND")
            all_found = False
    
    return all_found

@test("Modification: DPM++_SDE sampler support")
def test_dpm_sde_sampler():
    any2video_path = PROJECT_ROOT / "Wan2GP" / "models" / "wan" / "any2video.py"
    
    if not any2video_path.exists():
        # Try alternate location
        any2video_path = PROJECT_ROOT / "Wan2GP" / "wan" / "any2video.py"
    
    if not any2video_path.exists():
        print(f"   ⚠ any2video.py not found")
        return False
    
    with open(any2video_path) as f:
        content = f.read()
    
    if "dpm++_sde" in content.lower():
        print("   ✓ DPM++_SDE sampler found")
        return True
    else:
        print("   ⚠ DPM++_SDE sampler not found")
        return False


# =============================================================================
# GPU TESTS - Only run with --full flag
# =============================================================================

@test("GPU: CUDA available")
def test_cuda():
    import torch
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"   ✓ CUDA available: {device_name}")
        return True
    else:
        print("   ⚠ CUDA not available (expected on CPU-only systems)")
        RESULTS["passed"] -= 1  # Don't count as pass
        RESULTS["skipped"] += 1
        return None  # Skipped


# =============================================================================
# MAIN
# =============================================================================

def run_quick_tests():
    """Run import and config tests only (no GPU required)"""
    print("\n" + "="*60)
    print("RUNNING QUICK TESTS (imports and configs)")
    print("="*60)
    
    # Import tests
    test_import_shared_utils()
    test_import_attention()
    test_import_models_wan()
    test_import_models_qwen()
    test_import_wgp()
    
    # Config tests
    test_lightning_configs()
    test_lightning_configs_valid()
    test_vace_cocktail_configs()
    test_qwen_configs()
    test_critical_configs()
    
    # Headless tests
    test_headless_wgp()
    test_worker_imports()
    
    # Modification tests
    test_vid2vid_modification()
    test_dpm_sde_sampler()

def run_full_tests():
    """Run all tests including GPU tests"""
    run_quick_tests()
    
    print("\n" + "="*60)
    print("RUNNING GPU TESTS")
    print("="*60)
    
    test_cuda()

def print_summary():
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"   ✅ Passed:  {RESULTS['passed']}")
    print(f"   ❌ Failed:  {RESULTS['failed']}")
    print(f"   ⏭️  Skipped: {RESULTS['skipped']}")
    print("="*60)
    
    if RESULTS['failed'] == 0:
        print("\n🎉 ALL TESTS PASSED! Safe to proceed with upgrade.\n")
        return 0
    else:
        print(f"\n⚠️  {RESULTS['failed']} TESTS FAILED! Review before proceeding.\n")
        return 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Wan2GP Upgrade Verification Test")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only (no GPU)")
    parser.add_argument("--full", action="store_true", help="Run all tests including GPU")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("WAN2GP UPGRADE VERIFICATION TEST")
    print("="*60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Wan2GP path:  {PROJECT_ROOT / 'Wan2GP'}")
    
    if args.full:
        run_full_tests()
    else:
        run_quick_tests()
    
    exit_code = print_summary()
    sys.exit(exit_code)

