#!/usr/bin/env python3
"""
LoRA File Integrity Checker

This script checks LoRA files in common directories for corruption or size issues.
It can also clean up corrupted files if requested.

Usage:
    python check_loras.py                    # Check all LoRA directories
    python check_loras.py --fix              # Check and remove corrupted files
    python check_loras.py --dir /path/to/loras  # Check specific directory
"""

import argparse
import sys
from pathlib import Path

# Add source directory to path to import common_utils
sys.path.append(str(Path(__file__).parent / "source"))
from common_utils import check_loras_in_directory

def main():
    parser = argparse.ArgumentParser(description="Check LoRA file integrity")
    parser.add_argument("--dir", type=str, help="Specific directory to check")
    parser.add_argument("--fix", action="store_true", help="Remove corrupted files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Default LoRA directories to check
    default_dirs = [
        "Wan2GP/loras",
        "Wan2GP/loras_hunyuan", 
        "Wan2GP/loras_hunyuan_i2v",
        "Wan2GP/loras_i2v",
        "Wan2GP/loras_ltxv"
    ]
    
    if args.dir:
        dirs_to_check = [args.dir]
    else:
        dirs_to_check = default_dirs
    
    total_checked = 0
    total_valid = 0
    total_invalid = 0
    
    print("🔍 LoRA File Integrity Check")
    print("=" * 50)
    
    for lora_dir in dirs_to_check:
        dir_path = Path(lora_dir)
        print(f"\n📁 Checking directory: {dir_path}")
        
        if not dir_path.exists():
            print(f"   ⚠️  Directory not found: {dir_path}")
            continue
            
        results = check_loras_in_directory(dir_path, fix_issues=args.fix)
        
        if "error" in results:
            print(f"   ❌ Error: {results['error']}")
            continue
        
        total_checked += results["total_files"]
        total_valid += results["valid_files"]
        total_invalid += results["invalid_files"]
        
        print(f"   📊 Files found: {results['total_files']}")
        print(f"   ✅ Valid: {results['valid_files']}")
        print(f"   ❌ Invalid: {results['invalid_files']}")
        
        if results["invalid_files"] > 0:
            print(f"   🚨 Issues found:")
            for issue in results["issues"]:
                print(f"      {issue}")
        
        if args.verbose:
            print(f"   📋 Detailed results:")
            for summary_line in results["summary"]:
                print(f"      {summary_line}")
    
    print("\n" + "=" * 50)
    print(f"📈 Summary:")
    print(f"   Total files checked: {total_checked}")
    print(f"   Valid files: {total_valid}")
    print(f"   Invalid files: {total_invalid}")
    
    if total_invalid > 0:
        print(f"\n💡 Found {total_invalid} corrupted files.")
        if not args.fix:
            print("   Run with --fix to automatically remove corrupted files.")
        else:
            print("   Corrupted files have been removed.")
        sys.exit(1)
    else:
        print("\n🎉 All LoRA files are valid!")
        sys.exit(0)

if __name__ == "__main__":
    main() 