"""RunPod sync command."""

import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv


def run(client, options: dict):
    """Handle 'debug.py runpod' command."""
    load_dotenv()
    
    try:
        import runpod
    except ImportError:
        print("❌ Error: runpod module not installed")
        print("   Install with: pip install runpod")
        sys.exit(1)
    
    runpod_api_key = os.getenv('RUNPOD_API_KEY')
    if not runpod_api_key:
        print("❌ Error: RUNPOD_API_KEY not set in environment")
        sys.exit(1)
    
    runpod.api_key = runpod_api_key
    
    try:
        # Get RunPod pods
        print("🔍 Analyzing RunPod vs Database state...")
        runpod_pods = runpod.get_pods()
        
        # Filter for active pods that are orchestrator-managed (start with gpu_ or gpu-)
        active_runpod_pods = [
            pod for pod in runpod_pods 
            if pod.get('desiredStatus') in ['RUNNING', 'PROVISIONING']
            and (pod.get('name', '').startswith('gpu_') or pod.get('name', '').startswith('gpu-'))
        ]
        
        # Get database workers (only active/spawning to avoid fetching thousands of terminated workers)
        active_result = client.db.supabase.table('workers').select('*').eq('status', 'active').execute()
        spawning_result = client.db.supabase.table('workers').select('*').eq('status', 'spawning').execute()
        
        active_db_workers = (active_result.data or []) + (spawning_result.data or [])
        
        # Create lookup
        db_runpod_ids = set()
        for worker in active_db_workers:
            runpod_id = worker.get('metadata', {}).get('runpod_id')
            if runpod_id:
                db_runpod_ids.add(runpod_id)
        
        # Find orphaned pods
        orphaned_pods = []
        runpod_pod_ids = set()
        
        for pod in active_runpod_pods:
            pod_id = pod.get('id')
            runpod_pod_ids.add(pod_id)
            
            if pod_id not in db_runpod_ids:
                orphaned_pods.append(pod)
        
        # Find database-only workers
        db_only_workers = []
        for worker in active_db_workers:
            runpod_id = worker.get('metadata', {}).get('runpod_id')
            if runpod_id and runpod_id not in runpod_pod_ids:
                db_only_workers.append(worker)
        
        # Report
        print("\n" + "=" * 80)
        print("☁️  RUNPOD SYNC STATUS")
        print("=" * 80)
        
        print(f"\n📊 Summary:")
        print(f"   RunPod active pods: {len(active_runpod_pods)}")
        print(f"   Database active workers: {len(active_db_workers)}")
        print(f"   Orphaned pods (RunPod only): {len(orphaned_pods)}")
        print(f"   Stale workers (DB only): {len(db_only_workers)}")
        
        if orphaned_pods:
            print(f"\n🚨 ORPHANED PODS (in RunPod but not tracked in database):")
            print("-" * 80)
            total_cost = 0
            
            for pod in orphaned_pods:
                name = pod.get('name', 'unnamed')
                pod_id = pod.get('id', 'no-id')
                status = pod.get('desiredStatus', 'unknown')
                cost_per_hr = pod.get('costPerHr', 0)
                created = pod.get('createdAt', 'unknown')
                
                print(f"\n   Pod: {name}")
                print(f"   ID: {pod_id}")
                print(f"   Status: {status}")
                print(f"   Cost: ${cost_per_hr}/hr")
                print(f"   Created: {created}")
                total_cost += cost_per_hr
            
            print(f"\n💰 Total hourly cost of orphaned pods: ${total_cost:.3f}/hr")
            print(f"   Daily waste: ${total_cost * 24:.2f}")
            print(f"   Monthly waste: ${total_cost * 24 * 30:.2f}")
            
            if options.get('terminate'):
                print(f"\n⚠️  TERMINATING {len(orphaned_pods)} ORPHANED PODS...")
                for pod in orphaned_pods:
                    pod_id = pod.get('id')
                    print(f"   Terminating {pod_id}...")
                    try:
                        runpod.terminate_pod(pod_id)
                        print(f"   ✅ Terminated {pod_id}")
                    except Exception as e:
                        print(f"   ❌ Failed to terminate {pod_id}: {e}")
            else:
                print(f"\n💡 To terminate orphaned pods, run:")
                print(f"   python scripts/debug.py runpod --terminate")
        else:
            print(f"\n✅ No orphaned pods found!")
        
        if db_only_workers:
            print(f"\n⚠️  STALE WORKERS (in database but not in RunPod):")
            print("-" * 80)
            
            for worker in db_only_workers:
                worker_id = worker['id']
                status = worker['status']
                created = worker.get('created_at', 'unknown')
                runpod_id = worker.get('metadata', {}).get('runpod_id', 'N/A')
                
                print(f"\n   Worker: {worker_id}")
                print(f"   Status: {status}")
                print(f"   RunPod ID: {runpod_id}")
                print(f"   Created: {created}")
            
            print(f"\n💡 These workers should be marked as terminated in the database.")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)





