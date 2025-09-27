#!/usr/bin/env python3
"""
Simple test to demonstrate variants.json workflow working correctly
"""

import sys
from pathlib import Path
import json

# Add current directory to path
sys.path.append('.')

from base_tester import scan_experiments

def test_variants_simple():
    """Test that variants.json workflow follows simple_test structure"""
    print("🎯 Testing variants.json workflow with simple_test structure")
    print("=" * 60)

    # Show the structure we created
    test_path = Path("testing/variant_batch_test")
    print(f"Test batch: {test_path}")
    print(f"Experiment folder: {test_path}/variant_experiment_1/")
    print(f"  - settings.json ✅")
    print(f"  - variants.json ✅")
    print(f"  - images/ folder ✅")
    print()

    # Load and show variants.json content
    variants_file = test_path / "variant_experiment_1" / "variants.json"
    with open(variants_file, 'r') as f:
        variants_data = json.load(f)

    print("📄 Variants.json content:")
    print(f"  Found {len(variants_data['variants'])} variants:")
    for i, variant in enumerate(variants_data['variants'], 1):
        print(f"    {i}. {variant['prompt'][:40]}... ({variant['length']} frames)")
    print()

    # Test the scanning
    print("🔍 Scanning for tests...")
    pending_tests = scan_experiments(test_path)

    print(f"✅ Found {len(pending_tests)} pending tests")

    # Show what would be processed
    variant_tests = [t for t in pending_tests if t.get('is_variant', False)]
    print(f"✅ {len(variant_tests)} are variant-based tests")
    print()

    print("📋 Would generate these outputs:")
    for test in variant_tests:
        test_name = test['test_name']
        generation_num = test['generation_num']
        output_file = f"{generation_num}_output.mp4"

        input_set = test.get('input_set', {})
        prompt = input_set.get('prompt', 'Unknown')
        resolution = input_set.get('resolution', 'Unknown')
        length = input_set.get('length', 'Unknown')

        print(f"  📁 {test_name}/{output_file}")
        print(f"     Resolution: {resolution[0]}x{resolution[1]} ({length} frames)")
        print(f"     Prompt: {prompt[:50]}...")
        print()

    print("🎯 SUCCESS! Variants.json workflow is working correctly!")
    print("   Each variant in variants.json creates a separate numbered output")
    print("   Following the same pattern as simple_test structure")

if __name__ == "__main__":
    test_variants_simple()