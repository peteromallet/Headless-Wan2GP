#!/usr/bin/env python3
"""Minimal test to debug batch variant detection"""

import sys
from pathlib import Path
sys.path.append('.')

from variant_utils import load_variants_json

def test_simple_detection():
    """Test just the basic detection without processing"""
    experiments_dir = Path('testing/variant_demo')

    print(f"🔍 Testing basic variant detection in {experiments_dir}")

    # Check batch-level variants.json
    batch_variants_file = experiments_dir / "variants.json"
    if batch_variants_file.exists():
        print(f"✅ Found batch-level variants.json")
        try:
            batch_variants_data = load_variants_json(batch_variants_file)
            if batch_variants_data and 'variants' in batch_variants_data:
                print(f"✅ Loaded {len(batch_variants_data['variants'])} variants")
                for i, variant in enumerate(batch_variants_data['variants'], 1):
                    print(f"   Variant {i}: {variant.get('prompt', 'No prompt')[:50]}...")
            else:
                print("❌ Invalid variant data")
        except Exception as e:
            print(f"❌ Error loading variants: {e}")
    else:
        print("❌ No batch-level variants.json found")

    # Check experiments
    experiments = []
    for test_folder in experiments_dir.iterdir():
        if test_folder.is_dir() and (test_folder / "settings.json").exists():
            experiments.append(test_folder.name)
            print(f"✅ Found experiment: {test_folder.name}")

    print(f"\n📊 Summary:")
    print(f"   Experiments: {len(experiments)}")
    print(f"   Expected total outputs: {len(experiments)} × 3 = {len(experiments) * 3}")

if __name__ == "__main__":
    test_simple_detection()