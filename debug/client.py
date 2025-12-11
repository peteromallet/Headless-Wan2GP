"""Debug client for querying task data from Supabase."""

import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from collections import Counter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from supabase import create_client
from debug.models import TaskInfo, TasksSummary


class DebugClient:
    """Client for debugging task data."""
    
    def __init__(self):
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in environment")
        
        self.supabase = create_client(supabase_url, supabase_key)
    
    def get_task_info(self, task_id: str) -> TaskInfo:
        """Get complete task information from DB."""
        # Get task state from DB
        result = self.supabase.table('tasks').select('*').eq('id', task_id).execute()
        state = result.data[0] if result.data else None
        
        # Try to get logs if system_logs table exists
        logs = []
        try:
            logs_result = self.supabase.table('system_logs').select('*').eq('task_id', task_id).order('timestamp').execute()
            logs = logs_result.data or []
        except:
            pass  # system_logs table might not exist
        
        return TaskInfo(
            task_id=task_id,
            state=state,
            logs=logs
        )
    
    def get_recent_tasks(
        self,
        limit: int = 50,
        status: Optional[str] = None,
        task_type: Optional[str] = None,
        hours: Optional[int] = None
    ) -> TasksSummary:
        """Get recent tasks with analysis."""
        # Build query
        query = self.supabase.table('tasks').select('*')
        
        if status:
            query = query.eq('status', status)
        if task_type:
            query = query.eq('task_type', task_type)
        if hours:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            query = query.gte('created_at', cutoff.isoformat())
        
        query = query.order('created_at', desc=True).limit(limit)
        result = query.execute()
        tasks = result.data or []
        
        # Calculate statistics
        status_dist = Counter(t.get('status') for t in tasks)
        type_dist = Counter(t.get('task_type') for t in tasks)
        
        # Calculate timing statistics
        processing_times = []
        queue_times = []
        
        for task in tasks:
            if task.get('generation_started_at') and task.get('generation_processed_at'):
                try:
                    started = datetime.fromisoformat(task['generation_started_at'].replace('Z', '+00:00'))
                    processed = datetime.fromisoformat(task['generation_processed_at'].replace('Z', '+00:00'))
                    processing_times.append((processed - started).total_seconds())
                except:
                    pass
            
            if task.get('created_at') and task.get('generation_started_at'):
                try:
                    created = datetime.fromisoformat(task['created_at'].replace('Z', '+00:00'))
                    started = datetime.fromisoformat(task['generation_started_at'].replace('Z', '+00:00'))
                    queue_times.append((started - created).total_seconds())
                except:
                    pass
        
        timing_stats = {
            'avg_processing_seconds': sum(processing_times) / len(processing_times) if processing_times else None,
            'avg_queue_seconds': sum(queue_times) / len(queue_times) if queue_times else None,
            'total_with_timing': len(processing_times)
        }
        
        return TasksSummary(
            tasks=tasks,
            total_count=len(tasks),
            status_distribution=dict(status_dist),
            task_type_distribution=dict(type_dist),
            timing_stats=timing_stats
        )
