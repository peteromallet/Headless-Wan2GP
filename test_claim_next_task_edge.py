#!/usr/bin/env python3
"""
Test script for the claim-next-task Supabase Edge Function.
Tests both service-role and user token scenarios.
"""

import argparse
import json
import httpx
from datetime import datetime
import uuid

def test_edge_function(supabase_url: str, token: str, test_name: str = "Unknown"):
    """Test the claim-next-task edge function."""
    
    edge_url = f"{supabase_url.rstrip('/')}/functions/v1/claim-next-task"
    
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"URL: {edge_url}")
    print(f"Token: {token[:20]}...{token[-20:]}")
    print(f"{'='*60}")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    try:
        print(f"\nCalling edge function...")
        response = httpx.post(edge_url, json={}, headers=headers, timeout=30)
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ SUCCESS - Claimed task:")
            print(json.dumps(data, indent=2))
            
            # Verify required fields
            required_fields = ["task_id", "params", "task_type", "project_id"]
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                print(f"\n⚠️  WARNING: Missing required fields: {missing_fields}")
            
        elif response.status_code == 204:
            print(f"\n✅ SUCCESS - No tasks available (204 No Content)")
            
        elif response.status_code == 401:
            print(f"\n❌ UNAUTHORIZED - Check your token")
            print(f"Response: {response.text}")
            
        elif response.status_code == 403:
            print(f"\n❌ FORBIDDEN - Token not recognized or user not found")
            print(f"Response: {response.text}")
            
        else:
            print(f"\n❌ ERROR - Unexpected status code")
            print(f"Response: {response.text}")
            
    except httpx.TimeoutException:
        print(f"\n❌ TIMEOUT - Request took too long")
    except Exception as e:
        print(f"\n❌ EXCEPTION: {type(e).__name__}: {e}")

def create_test_task(supabase_url: str, service_key: str, user_id: str = None):
    """Create a test task using the service key."""
    from supabase import create_client
    
    print("\nCreating test task...")
    client = create_client(supabase_url, service_key)
    
    task_id = f"test_{uuid.uuid4().hex[:8]}"
    task_data = {
        "id": task_id,
        "task_type": "single_image",
        "params": json.dumps({
            "task_id": task_id,
            "prompt": "Test task created by edge function test",
            "model": "t2v",
            "seed": 42
        }),
        "status": "Queued",
        "project_id": user_id or "test_project",
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    
    try:
        response = client.table("tasks").insert(task_data).execute()
        print(f"✅ Created test task: {task_id} for project/user: {task_data['project_id']}")
        return task_id
    except Exception as e:
        print(f"❌ Failed to create test task: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Test claim-next-task edge function")
    parser.add_argument("--supabase-url", required=True, help="Supabase project URL")
    parser.add_argument("--user-token", help="User JWT or PAT token")
    parser.add_argument("--service-key", help="Service role key")
    parser.add_argument("--create-test-task", action="store_true", help="Create a test task first")
    parser.add_argument("--user-id", help="User ID for test task creation")
    
    args = parser.parse_args()
    
    if not args.user_token and not args.service_key:
        print("ERROR: Provide at least one of --user-token or --service-key")
        return
    
    # Create test task if requested
    if args.create_test_task and args.service_key:
        create_test_task(args.supabase_url, args.service_key, args.user_id)
    
    # Test with service key
    if args.service_key:
        test_edge_function(args.supabase_url, args.service_key, "Service Role Key")
    
    # Test with user token
    if args.user_token:
        test_edge_function(args.supabase_url, args.user_token, "User Token")

if __name__ == "__main__":
    main() 