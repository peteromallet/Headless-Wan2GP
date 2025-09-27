#!/usr/bin/env python3
"""
Monitor script to track base_tester.py progress and update consolidated_logs.json
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime

def update_consolidated_log():
    """Update the consolidated log with latest summaries and run files"""

    base_dir = Path("/workspace/agent_tasks/Headless-Wan2GP/testing_tools/testing/simple_test")
    consolidated_file = base_dir / "consolidated_logs.json"

    # Load existing consolidated log
    try:
        with open(consolidated_file, 'r') as f:
            consolidated = json.load(f)
    except:
        consolidated = {
            "collection_timestamp": datetime.now().isoformat(),
            "directory": str(base_dir),
            "summaries_data": [],
            "individual_run_files": []
        }

    # Update collection timestamp
    consolidated["collection_timestamp"] = datetime.now().isoformat()

    # Re-read summaries.json if it exists
    summaries_file = base_dir / "summaries.json"
    if summaries_file.exists():
        try:
            with open(summaries_file, 'r') as f:
                consolidated["summaries_data"] = json.load(f)
        except:
            pass

    # Find all new run files
    run_files = []

    # Find all run_details.json and metadata.json files
    for pattern in ["**/run_details.json", "**/metadata.json", "**/*_run_details.json", "**/*_metadata.json", "**/*_error.json"]:
        for file_path in base_dir.glob(pattern):
            if file_path.name not in ["consolidated_logs.json", "summaries.json"]:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    relative_path = str(file_path.relative_to(base_dir))
                    run_files.append({
                        "source_file": relative_path,
                        "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        "data": data
                    })
                except:
                    continue

    # Sort by modification time
    run_files.sort(key=lambda x: x["last_modified"])
    consolidated["individual_run_files"] = run_files

    # Add analysis
    successful = len([f for f in run_files if f["data"].get("status") in ["completed", "success"]])
    failed = len([f for f in run_files if f["data"].get("status") in ["failed", "error"]])

    consolidated["analysis"] = {
        "total_summaries_entries": len(consolidated["summaries_data"]),
        "total_individual_runs": len(run_files),
        "successful_generations": successful,
        "failed_generations": failed,
        "last_update": datetime.now().isoformat()
    }

    # Write back to consolidated log
    with open(consolidated_file, 'w') as f:
        json.dump(consolidated, f, indent=2)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Updated consolidated log - {len(run_files)} run files, {successful} successful, {failed} failed")

if __name__ == "__main__":
    print("Starting base_tester monitor...")

    while True:
        try:
            update_consolidated_log()
            time.sleep(30)  # Update every 30 seconds
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            break
        except Exception as e:
            print(f"Error updating log: {e}")
            time.sleep(30)