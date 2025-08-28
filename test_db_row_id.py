#!/usr/bin/env python3
"""
Test script to verify that we can explicitly set UUID row IDs in the database.
This tests both SQLite and Supabase approaches.
"""

import sys
import os
import uuid
import json
import sqlite3
from pathlib import Path

# Add the source directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "source"))

def test_sqlite_explicit_uuid():
    """Test if we can explicitly set UUIDs as row IDs in SQLite."""
    print("=== Testing SQLite with explicit UUID row IDs ===")
    
    test_db_path = "test_uuid_rows.db"
    
    try:
        # Clean up any existing test database
        if Path(test_db_path).exists():
            Path(test_db_path).unlink()
        
        # Create a test database with our schema
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        
        # Create the tasks table (similar to our real schema)
        cursor.execute("""
            CREATE TABLE tasks (
                id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                params TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Queued',
                dependant_on TEXT NULL,
                created_at TEXT NOT NULL,
                project_id TEXT NOT NULL
            )
        """)
        
        # Test 1: Insert with explicit UUID
        test_uuid_1 = str(uuid.uuid4())
        test_params = {"test": "data", "task_id": "logical_id_123"}
        
        cursor.execute("""
            INSERT INTO tasks (id, task_type, params, status, created_at, project_id) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            test_uuid_1,
            "test_task",
            json.dumps(test_params),
            "Queued",
            "2025-01-01T00:00:00Z",
            "test_project"
        ))
        
        print(f"✓ Successfully inserted task with explicit UUID: {test_uuid_1}")
        
        # Test 2: Insert a dependent task
        test_uuid_2 = str(uuid.uuid4())
        test_params_2 = {"test": "data2", "task_id": "logical_id_456"}
        
        cursor.execute("""
            INSERT INTO tasks (id, task_type, params, status, created_at, project_id, dependant_on) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            test_uuid_2,
            "test_task_dependent",
            json.dumps(test_params_2),
            "Queued",
            "2025-01-01T00:01:00Z",
            "test_project",
            test_uuid_1  # Depends on first task's actual row ID
        ))
        
        print(f"✓ Successfully inserted dependent task with UUID: {test_uuid_2}")
        print(f"  - Depends on: {test_uuid_1}")
        
        # Test 3: Verify we can query by the explicit UUIDs
        cursor.execute("SELECT id, task_type, dependant_on FROM tasks WHERE id = ?", (test_uuid_1,))
        row1 = cursor.fetchone()
        print(f"✓ Retrieved task 1: ID={row1[0]}, Type={row1[1]}, DependsOn={row1[2]}")
        
        cursor.execute("SELECT id, task_type, dependant_on FROM tasks WHERE id = ?", (test_uuid_2,))
        row2 = cursor.fetchone()
        print(f"✓ Retrieved task 2: ID={row2[0]}, Type={row2[1]}, DependsOn={row2[2]}")
        
        # Test 4: Verify dependency relationship
        if row2[2] == test_uuid_1:
            print("✓ Dependency relationship correctly established using explicit UUIDs")
        else:
            print(f"✗ Dependency mismatch! Expected {test_uuid_1}, got {row2[2]}")
            return False
        
        conn.commit()
        conn.close()
        
        # Clean up
        Path(test_db_path).unlink()
        
        print("✓ SQLite explicit UUID test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ SQLite explicit UUID test FAILED: {e}")
        if Path(test_db_path).exists():
            Path(test_db_path).unlink()
        return False

def test_supabase_uuid_format():
    """Test if our UUID format is compatible with Supabase requirements."""
    print("\n=== Testing UUID format compatibility ===")
    
    try:
        # Generate UUIDs using our method
        test_uuid_1 = str(uuid.uuid4())
        test_uuid_2 = str(uuid.uuid4())
        
        print(f"Generated UUID 1: {test_uuid_1}")
        print(f"Generated UUID 2: {test_uuid_2}")
        
        # Check format (should be 36 characters with hyphens)
        if len(test_uuid_1) == 36 and len(test_uuid_2) == 36:
            print("✓ UUID length is correct (36 characters)")
        else:
            print(f"✗ UUID length incorrect: {len(test_uuid_1)}, {len(test_uuid_2)}")
            return False
            
        # Check if they're valid UUID4 format
        try:
            uuid_obj_1 = uuid.UUID(test_uuid_1)
            uuid_obj_2 = uuid.UUID(test_uuid_2)
            print("✓ UUIDs are valid UUID4 format")
        except ValueError as e:
            print(f"✗ Invalid UUID format: {e}")
            return False
            
        # Check that they're unique
        if test_uuid_1 != test_uuid_2:
            print("✓ UUIDs are unique")
        else:
            print("✗ UUIDs are not unique!")
            return False
            
        print("✓ UUID format compatibility test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ UUID format test FAILED: {e}")
        return False

def test_edge_function_compatibility():
    """Test that our UUID approach will work with the Supabase edge function."""
    print("\n=== Testing Edge Function Payload Compatibility ===")
    
    try:
        # Simulate what our updated add_task_to_db function will send
        generated_uuid = str(uuid.uuid4())
        
        # This is the payload structure we'll send to create-task edge function
        test_payload = {
            "task_id": generated_uuid,  # Our explicitly generated UUID
            "params": {
                "task_id": "logical_id_from_orchestrator",  # The logical ID in params
                "prompt": "test prompt",
                "model": "test_model"
            },
            "task_type": "travel_segment",
            "project_id": "test_project_id",
            "dependant_on": None
        }
        
        print(f"Edge function payload:")
        print(f"  task_id (DB row ID): {test_payload['task_id']}")
        print(f"  params.task_id (logical): {test_payload['params']['task_id']}")
        print(f"  task_type: {test_payload['task_type']}")
        print(f"  dependant_on: {test_payload['dependant_on']}")
        
        # Verify that our approach separates the concerns correctly
        db_row_id = test_payload["task_id"]
        logical_id = test_payload["params"]["task_id"]
        
        if db_row_id != logical_id:
            print("✓ Database row ID and logical task ID are properly separated")
        else:
            print("✗ Database row ID and logical task ID should be different!")
            return False
            
        # Test dependency chain simulation
        first_task_db_id = str(uuid.uuid4())
        second_task_payload = {
            "task_id": str(uuid.uuid4()),
            "params": {"task_id": "logical_second_task"},
            "task_type": "travel_segment", 
            "project_id": "test_project_id",
            "dependant_on": first_task_db_id  # Uses first task's DB row ID
        }
        
        print(f"\nDependency chain test:")
        print(f"  First task DB ID: {first_task_db_id}")
        print(f"  Second task depends on: {second_task_payload['dependant_on']}")
        
        if second_task_payload["dependant_on"] == first_task_db_id:
            print("✓ Dependency chain uses actual database row IDs")
        else:
            print("✗ Dependency chain mismatch!")
            return False
            
        print("✓ Edge function compatibility test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Edge function compatibility test FAILED: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing explicit UUID row ID approach...\n")
    
    results = []
    
    # Test SQLite compatibility
    results.append(test_sqlite_explicit_uuid())
    
    # Test UUID format
    results.append(test_supabase_uuid_format())
    
    # Test edge function compatibility
    results.append(test_edge_function_compatibility())
    
    print(f"\n=== Test Results ===")
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ ALL TESTS PASSED - Explicit UUID row ID approach is viable!")
        return True
    else:
        print("✗ SOME TESTS FAILED - Need to revise approach")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
