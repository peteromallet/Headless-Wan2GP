#!/usr/bin/env python3
"""
Task Database Inspector
======================

Simple utility to check what tasks are currently in the SQLite database.
"""

import sys
import json
import sqlite3
from pathlib import Path

def check_tasks(db_path="tasks.db"):
    """Check what tasks are in the database"""
    
    if not Path(db_path).exists():
        print(f"❌ Database file not found: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tasks
        cursor.execute("""
            SELECT id, task_type, status, created_at, project_id, 
                   substr(params, 1, 100) as params_preview
            FROM tasks 
            ORDER BY created_at DESC
        """)
        
        tasks = cursor.fetchall()
        
        if not tasks:
            print("📭 No tasks found in database")
            return
        
        print(f"📋 Found {len(tasks)} tasks in database:")
        print("=" * 100)
        
        for task in tasks:
            task_id, task_type, status, created_at, project_id, params_preview = task
            
            print(f"🔹 Task ID: {task_id}")
            print(f"   Type: {task_type}")
            print(f"   Status: {status}")
            print(f"   Created: {created_at}")
            print(f"   Project: {project_id}")
            print(f"   Params Preview: {params_preview}...")
            print("-" * 50)
        
        # Count by status
        cursor.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status")
        status_counts = cursor.fetchall()
        
        print("\n📊 Status Summary:")
        for status, count in status_counts:
            print(f"   {status}: {count}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")

def delete_task(task_id, db_path="tasks.db"):
    """Delete a specific task"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        
        if cursor.rowcount > 0:
            conn.commit()
            print(f"✅ Deleted task: {task_id}")
        else:
            print(f"❌ Task not found: {task_id}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error deleting task: {e}")

def clear_all_tasks(db_path="tasks.db"):
    """Clear all tasks from database"""
    response = input("⚠️  Are you sure you want to delete ALL tasks? (yes/no): ")
    if response.lower() != 'yes':
        print("Operation cancelled")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM tasks")
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        print(f"✅ Deleted {deleted_count} tasks")
        
    except Exception as e:
        print(f"❌ Error clearing tasks: {e}")

def main():
    """Main function with command line interface"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "clear":
            clear_all_tasks()
        elif command == "delete" and len(sys.argv) > 2:
            task_id = sys.argv[2]
            delete_task(task_id)
        else:
            print("Usage:")
            print("  python check_tasks.py           # Show all tasks")
            print("  python check_tasks.py clear     # Delete all tasks")
            print("  python check_tasks.py delete <task_id>  # Delete specific task")
    else:
        check_tasks()

if __name__ == "__main__":
    main() 