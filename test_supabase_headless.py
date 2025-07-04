#!/usr/bin/env python3
"""
Test script for Supabase headless implementation.
This script tests the basic functionality without actually running tasks.
"""

import sys
import os
import argparse
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from source import db_operations as db_ops
from supabase import create_client

def test_supabase_connection(url, token):
    """Test basic Supabase connection and authentication."""
    print("Testing Supabase connection...")
    try:
        client = create_client(url, token)
        # Try a simple query to test authentication
        response = client.table("tasks").select("count", count="exact").execute()
        print(f"✅ Successfully connected to Supabase")
        print(f"   Total tasks in database: {response.count}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        return False

def test_rpc_functions(url, token):
    """Test required RPC functions."""
    print("\nTesting RPC functions...")
    client = create_client(url, token)
    
    # Test func_claim_task
    try:
        response = client.rpc("func_claim_task", {
            "p_table_name": "tasks",
            "p_worker_id": "test_worker"
        }).execute()
        
        if response.data and len(response.data) > 0:
            task = response.data[0]
            print(f"✅ func_claim_task works")
            print(f"   Claimed task: {task.get('task_id_out')}")
            
            # Check if project_id_out is present
            if task.get('project_id_out') is not None:
                print(f"   ✅ project_id_out is present: {task.get('project_id_out')}")
            else:
                print(f"   ⚠️  project_id_out is missing from RPC response!")
                print(f"   Available fields: {list(task.keys())}")
        else:
            print(f"✅ func_claim_task works (no tasks available)")
    except Exception as e:
        print(f"❌ func_claim_task failed: {e}")

def test_db_operations_module(url, token):
    """Test the db_operations module with Supabase."""
    print("\nTesting db_operations module...")
    
    # Configure the module
    db_ops.DB_TYPE = "supabase"
    db_ops.PG_TABLE_NAME = "tasks"
    db_ops.SUPABASE_URL = url
    db_ops.SUPABASE_SERVICE_KEY = token
    db_ops.SUPABASE_CLIENT = create_client(url, token)
    
    # Test get_oldest_queued_task
    try:
        task = db_ops.get_oldest_queued_task_supabase()
        if task:
            print(f"✅ get_oldest_queued_task_supabase works")
            print(f"   Task ID: {task.get('task_id')}")
            print(f"   Task Type: {task.get('task_type')}")
            print(f"   Project ID: {task.get('project_id')}")
            if task.get('project_id') is None:
                print(f"   ⚠️  project_id is None - RPC needs update")
        else:
            print(f"✅ get_oldest_queued_task_supabase works (no tasks available)")
    except Exception as e:
        print(f"❌ get_oldest_queued_task_supabase failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test Supabase headless implementation")
    parser.add_argument("--supabase-url", required=True, help="Supabase project URL")
    parser.add_argument("--supabase-access-token", required=True, help="Supabase access token")
    
    args = parser.parse_args()
    
    print("=== Supabase Headless Test ===\n")
    
    # Run tests
    if test_supabase_connection(args.supabase_url, args.supabase_access_token):
        test_rpc_functions(args.supabase_url, args.supabase_access_token)
        test_db_operations_module(args.supabase_url, args.supabase_access_token)
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main() 