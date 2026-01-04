#!/usr/bin/env python3
"""
Restore custom Wan2GP default configurations after upgrade.

Usage:
    python upgrade_backup/restore_configs.py          # Restore all configs
    python upgrade_backup/restore_configs.py --check  # Check which are missing
    python upgrade_backup/restore_configs.py --critical  # Restore only critical configs
"""

import json
import shutil
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKUP_DIR = SCRIPT_DIR / "defaults"
TARGET_DIR = PROJECT_ROOT / "Wan2GP" / "defaults"
MANIFEST_FILE = SCRIPT_DIR / "CONFIG_MANIFEST.json"

def load_manifest():
    with open(MANIFEST_FILE) as f:
        return json.load(f)

def get_critical_configs(manifest):
    """Get list of critical config files"""
    critical = []
    for category, info in manifest["categories"].items():
        if info.get("critical", False):
            critical.extend(info["files"])
    return critical

def get_all_configs():
    """Get all backed up config files"""
    return [f.name for f in BACKUP_DIR.glob("*.json")]

def check_configs():
    """Check which configs are missing from target"""
    manifest = load_manifest()
    all_configs = get_all_configs()
    critical_configs = get_critical_configs(manifest)
    
    missing_critical = []
    missing_other = []
    present = []
    
    for config in all_configs:
        target_path = TARGET_DIR / config
        if target_path.exists():
            present.append(config)
        elif config in critical_configs:
            missing_critical.append(config)
        else:
            missing_other.append(config)
    
    print(f"\n{'='*60}")
    print("CONFIG STATUS CHECK")
    print(f"{'='*60}")
    print(f"✅ Present:          {len(present)}")
    print(f"❌ Missing Critical: {len(missing_critical)}")
    print(f"⚠️  Missing Other:    {len(missing_other)}")
    
    if missing_critical:
        print(f"\n❌ CRITICAL CONFIGS MISSING:")
        for c in missing_critical:
            print(f"   - {c}")
    
    if missing_other:
        print(f"\n⚠️  Other configs missing:")
        for c in missing_other[:10]:
            print(f"   - {c}")
        if len(missing_other) > 10:
            print(f"   ... and {len(missing_other) - 10} more")
    
    return missing_critical, missing_other

def restore_configs(configs_to_restore, dry_run=False):
    """Restore specified configs"""
    restored = 0
    skipped = 0
    errors = 0
    
    for config in configs_to_restore:
        source = BACKUP_DIR / config
        target = TARGET_DIR / config
        
        if not source.exists():
            print(f"⚠️  Source not found: {config}")
            errors += 1
            continue
        
        if target.exists():
            print(f"⏭️  Skipping (exists): {config}")
            skipped += 1
            continue
        
        if dry_run:
            print(f"🔄 Would restore: {config}")
            restored += 1
        else:
            try:
                shutil.copy2(source, target)
                print(f"✅ Restored: {config}")
                restored += 1
            except Exception as e:
                print(f"❌ Error restoring {config}: {e}")
                errors += 1
    
    return restored, skipped, errors

def main():
    parser = argparse.ArgumentParser(description="Restore Wan2GP configs")
    parser.add_argument("--check", action="store_true", help="Check which configs are missing")
    parser.add_argument("--critical", action="store_true", help="Restore only critical configs")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be restored")
    parser.add_argument("--force", action="store_true", help="Overwrite existing configs")
    args = parser.parse_args()
    
    if not BACKUP_DIR.exists():
        print(f"❌ Backup directory not found: {BACKUP_DIR}")
        return 1
    
    if not TARGET_DIR.exists():
        print(f"❌ Target directory not found: {TARGET_DIR}")
        print("   Is Wan2GP installed?")
        return 1
    
    manifest = load_manifest()
    
    if args.check:
        missing_critical, missing_other = check_configs()
        if missing_critical:
            print(f"\n💡 Run 'python {__file__} --critical' to restore critical configs")
        return 0 if not missing_critical else 1
    
    # Determine which configs to restore
    if args.critical:
        configs = get_critical_configs(manifest)
        print(f"\n🔧 Restoring {len(configs)} CRITICAL configs...")
    else:
        configs = get_all_configs()
        print(f"\n🔧 Restoring ALL {len(configs)} configs...")
    
    if args.force:
        print("⚠️  Force mode: will overwrite existing configs")
        # Remove existing to force overwrite
        for config in configs:
            target = TARGET_DIR / config
            if target.exists():
                target.unlink()
    
    restored, skipped, errors = restore_configs(configs, dry_run=args.dry_run)
    
    print(f"\n{'='*60}")
    print("RESTORE SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Restored: {restored}")
    print(f"⏭️  Skipped:  {skipped}")
    print(f"❌ Errors:   {errors}")
    
    if args.dry_run:
        print(f"\n💡 Run without --dry-run to actually restore")
    
    return 0 if errors == 0 else 1

if __name__ == "__main__":
    exit(main())

