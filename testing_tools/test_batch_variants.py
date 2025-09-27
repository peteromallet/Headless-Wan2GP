#!/usr/bin/env python3
"""
Test the new batch-level variants.json workflow
"""

import sys
from pathlib import Path
import json

# Add current directory to path
sys.path.append('.')

from base_tester import load_settings
from variant_utils import load_variants_json, process_variant

def scan_experiments_with_batch_variants(experiments_path):
    """Scan for experiments with batch-level variants.json"""
    experiments_dir = Path(experiments_path)

    if not experiments_dir.exists():
        return []

    # Look for batch-level variants.json
    batch_variants_file = experiments_dir / "variants.json"
    batch_variants_data = None

    if batch_variants_file.exists():
        print(f"✅ Found batch-level variants.json in {experiments_dir.name}")
        batch_variants_data = load_variants_json(batch_variants_file)
    else:
        print(f"❌ No batch-level variants.json found in {experiments_dir.name}")
        return []

    if not batch_variants_data or 'variants' not in batch_variants_data:
        print("❌ Invalid batch variants data")
        return []

    print(f"📋 Batch has {len(batch_variants_data['variants'])} variants")

    pending_tests = []

    # Scan each experiment folder
    for test_folder in experiments_dir.iterdir():
        if test_folder.is_dir() and (test_folder / "settings.json").exists():
            print(f"📁 Found experiment: {test_folder.name}")

            # Process batch variants for this experiment
            temp_dir = test_folder / "temp_variants"
            temp_dir.mkdir(exist_ok=True)

            # Process each variant with this experiment's settings
            for i, variant_data in enumerate(batch_variants_data['variants'], 1):
                output_file = test_folder / f"{i}_output.mp4"
                error_file = test_folder / f"{i}_error.json"

                if not output_file.exists() and not error_file.exists():
                    print(f"  Variant {i}: {variant_data['prompt'][:50]}...")

                    # Process the variant
                    processed_files = process_variant(variant_data, i, experiments_dir, temp_dir)

                    if processed_files:
                        print(f"    ✅ Ready for generation")
                        pending_tests.append({
                            "folder": test_folder,
                            "settings_file": test_folder / "settings.json",
                            "test_name": test_folder.name,
                            "generation_num": i,
                            "input_set": {
                                'video': Path(processed_files['video']),
                                'mask': Path(processed_files['mask']),
                                'prompt_file': Path(processed_files['prompt_file']),
                                'resolution': processed_files['resolution'],
                                'length': processed_files['length'],
                                'prompt': processed_files['prompt']
                            },
                            "is_variant": True
                        })
                    else:
                        print(f"    ❌ Variant processing failed")
                else:
                    if output_file.exists():
                        print(f"  Variant {i}: Already completed")
                    if error_file.exists():
                        print(f"  Variant {i}: Has error - skipping")

    return pending_tests

def test_batch_variants():
    """Test the batch variants workflow"""
    print("🎯 Testing Batch-Level Variants.json Workflow")
    print("=" * 60)

    # Test the new structure
    pending_tests = scan_experiments_with_batch_variants('testing/variant_demo')

    print(f"\n✅ Found {len(pending_tests)} pending variant tests")

    # Group by experiment
    by_experiment = {}
    for test in pending_tests:
        exp_name = test['test_name']
        if exp_name not in by_experiment:
            by_experiment[exp_name] = []
        by_experiment[exp_name].append(test)

    print(f"📊 Tests grouped by experiment:")
    for exp_name, tests in by_experiment.items():
        print(f"  📁 {exp_name}/")
        for test in tests:
            variant_num = test['generation_num']
            input_set = test['input_set']
            print(f"    {variant_num}_output.mp4 - {input_set['prompt'][:40]}...")

    print("\n🎯 This demonstrates the correct workflow:")
    print("   • variants.json at batch level defines 3 variants")
    print("   • Each experiment uses ALL 3 variants with its own settings")
    print("   • Results: 3 experiments × 3 variants = 9 total outputs")

if __name__ == "__main__":
    test_batch_variants()