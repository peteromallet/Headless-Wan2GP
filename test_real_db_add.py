#!/usr/bin/env python3
"""
Test the actual add_task_to_db function with explicit UUID approach.
"""

import sys
import os
import uuid
import json
from pathlib import Path

# Add current directory to Python path so we can import from source
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our database operations
from source import db_operations as db_ops

def test_real_add_task_to_db():
    """Test our modified add_task_to_db function."""
    print("=== Testing Real add_task_to_db Function ===")
    
    try:
        # Ensure we're using SQLite for this test
        original_db_type = db_ops.DB_TYPE
        original_db_path = db_ops.SQLITE_DB_PATH
        
        # Set up test database
        test_db_path = "test_real_add_task.db"
        if Path(test_db_path).exists():
            Path(test_db_path).unlink()
            
        db_ops.DB_TYPE = "sqlite"
        db_ops.SQLITE_DB_PATH = test_db_path
        
        # Initialize the test database
        db_ops._init_db_sqlite(test_db_path)
        print(f"✓ Initialized test database: {test_db_path}")
        
        # Test 1: Add first task
        task_payload_1 = {
            "task_id": "logical_task_1",  # This should be ignored for DB row ID
            "prompt": "Test prompt 1",
            "model": "test_model",
            "project_id": "test_project"
        }
        
        actual_db_row_id_1 = db_ops.add_task_to_db(
            task_payload=task_payload_1,
            task_type_str="test_task",
            dependant_on=None
        )
        
        print(f"✓ Added first task:")
        print(f"  Logical ID (in params): {task_payload_1['task_id']}")
        print(f"  Actual DB row ID: {actual_db_row_id_1}")
        
        # Verify they're different
        if actual_db_row_id_1 != task_payload_1['task_id']:
            print("✓ Database row ID differs from logical task ID (as expected)")
        else:
            print("✗ Database row ID should differ from logical task ID!")
            return False
        
        # Test 2: Add dependent task
        task_payload_2 = {
            "task_id": "logical_task_2",
            "prompt": "Test prompt 2", 
            "model": "test_model",
            "project_id": "test_project"
        }
        
        actual_db_row_id_2 = db_ops.add_task_to_db(
            task_payload=task_payload_2,
            task_type_str="test_task_dependent",
            dependant_on=actual_db_row_id_1  # Use actual DB row ID for dependency
        )
        
        print(f"✓ Added dependent task:")
        print(f"  Logical ID (in params): {task_payload_2['task_id']}")
        print(f"  Actual DB row ID: {actual_db_row_id_2}")
        print(f"  Depends on: {actual_db_row_id_1}")
        
        # Test 3: Verify the tasks were stored correctly
        import sqlite3
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, task_type, dependant_on, params FROM tasks ORDER BY created_at")
        rows = cursor.fetchall()
        
        if len(rows) == 2:
            print("✓ Both tasks were stored in database")
            
            # Check first task
            task1_id, task1_type, task1_depends, task1_params = rows[0]
            task1_params_obj = json.loads(task1_params)
            
            print(f"  Task 1 - DB ID: {task1_id}, Type: {task1_type}, Depends: {task1_depends}")
            print(f"  Task 1 - Logical ID in params: {task1_params_obj.get('task_id')}")
            
            if task1_id == actual_db_row_id_1 and task1_params_obj.get('task_id') == 'logical_task_1':
                print("✓ Task 1 stored correctly with separate DB ID and logical ID")
            else:
                print("✗ Task 1 storage mismatch!")
                return False
            
            # Check second task
            task2_id, task2_type, task2_depends, task2_params = rows[1]
            task2_params_obj = json.loads(task2_params)
            
            print(f"  Task 2 - DB ID: {task2_id}, Type: {task2_type}, Depends: {task2_depends}")
            print(f"  Task 2 - Logical ID in params: {task2_params_obj.get('task_id')}")
            
            if (task2_id == actual_db_row_id_2 and 
                task2_params_obj.get('task_id') == 'logical_task_2' and
                task2_depends == actual_db_row_id_1):
                print("✓ Task 2 stored correctly with proper dependency relationship")
            else:
                print("✗ Task 2 storage or dependency mismatch!")
                return False
        else:
            print(f"✗ Expected 2 tasks, found {len(rows)}")
            return False
            
        conn.close()
        
        # Clean up
        Path(test_db_path).unlink()
        
        # Restore original settings
        db_ops.DB_TYPE = original_db_type
        db_ops.SQLITE_DB_PATH = original_db_path
        
        print("✓ Real add_task_to_db test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Real add_task_to_db test FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up on error
        if Path(test_db_path).exists():
            Path(test_db_path).unlink()
            
        # Restore original settings
        db_ops.DB_TYPE = original_db_type
        db_ops.SQLITE_DB_PATH = original_db_path
        
        return False

if __name__ == "__main__":
    success = test_real_add_task_to_db()
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
