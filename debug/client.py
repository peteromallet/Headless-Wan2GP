"""Unified client for debugging data access."""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from collections import Counter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from supabase import create_client
from debug.models import (
    TaskInfo, WorkerInfo, TasksSummary, WorkersSummary,
    SystemHealth, OrchestratorStatus
)


class LogQueryClient:
    """Client for querying system logs from Supabase."""
    
    def __init__(self, supabase_client):
        self.supabase = supabase_client
    
    def get_logs(
        self,
        start_time: datetime = None,
        end_time: datetime = None,
        source_type: str = None,
        source_id: str = None,
        worker_id: str = None,
        task_id: str = None,
        log_level: str = None,
        cycle_number: int = None,
        search_term: str = None,
        limit: int = 1000,
        order_desc: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query logs with flexible filters.
        """
        # Default time range: last 24 hours
        if not start_time:
            start_time = datetime.now(timezone.utc) - timedelta(hours=24)
        if not end_time:
            end_time = datetime.now(timezone.utc)
        
        # Check if system_logs table exists
        try:
            query = self.supabase.table('system_logs').select('*')
            
            # Apply filters
            query = query.gte('timestamp', start_time.isoformat())
            query = query.lte('timestamp', end_time.isoformat())
            
            if source_type:
                query = query.eq('source_type', source_type)
            if source_id:
                query = query.eq('source_id', source_id)
            if worker_id:
                query = query.eq('worker_id', worker_id)
            if task_id:
                query = query.eq('task_id', task_id)
            if log_level:
                query = query.eq('log_level', log_level)
            if cycle_number is not None:
                query = query.eq('cycle_number', cycle_number)
            if search_term:
                query = query.ilike('message', f'%{search_term}%')
            
            # Order and limit
            query = query.order('timestamp', desc=order_desc)
            query = query.limit(limit)
            
            result = query.execute()
            return result.data or []
        except Exception as e:
            # system_logs table might not exist in this setup
            return []
    
    def get_task_timeline(self, task_id: str) -> List[Dict[str, Any]]:
        """Get timeline of logs for a specific task."""
        return self.get_logs(
            task_id=task_id,
            limit=1000,
            order_desc=False  # Chronological order
        )


class DebugClient:
    """Unified client for all debugging data."""
    
    def __init__(self):
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in environment")
        
        self.supabase = create_client(supabase_url, supabase_key)
        self.log_client = LogQueryClient(self.supabase)
    
    # ==================== TASK METHODS ====================
    
    def get_task_info(self, task_id: str, follow_cascade: bool = True) -> TaskInfo:
        """Get complete task information: logs + DB state.
        
        If follow_cascade is True and this is an orchestrator that failed due to
        a child task, also fetch the child task info.
        """
        # Get logs for this task
        logs = self.log_client.get_task_timeline(task_id)
        
        # Get task state from DB
        result = self.supabase.table('tasks').select('*').eq('id', task_id).execute()
        state = result.data[0] if result.data else None
        
        # Check for cascaded failure and fetch child task
        child_task_info = None
        if follow_cascade and state:
            error_msg = state.get('error_message', '') or ''
            output_loc = state.get('output_location', '') or ''
            
            # Look for "Cascaded failed from related task <uuid>" pattern
            import re
            cascade_pattern = r'[Cc]ascad(?:ed|e) (?:failed |failure )?(?:from )?(?:related )?task[:\s]+([a-f0-9-]{36})'
            
            for text in [error_msg, output_loc]:
                match = re.search(cascade_pattern, text)
                if match:
                    child_task_id = match.group(1)
                    if child_task_id != task_id:  # Avoid infinite recursion
                        child_task_info = self.get_task_info(child_task_id, follow_cascade=False)
                    break
        
        return TaskInfo(
            task_id=task_id,
            state=state,
            logs=logs,
            child_task_info=child_task_info
        )
    
    # ==================== WORKER METHODS ====================
    
    def get_worker_info(self, worker_id: str, hours: int = 24, startup: bool = False) -> WorkerInfo:
        """Get complete worker information: logs + DB state + tasks."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # Get logs for this worker
        logs = self.log_client.get_logs(
            start_time=start_time,
            worker_id=worker_id,
            limit=5000,
            order_desc=False
        )
        
        # Try to get worker state from DB (workers table may not exist)
        state = None
        try:
            result = self.supabase.table('workers').select('*').eq('id', worker_id).execute()
            state = result.data[0] if result.data else None
        except:
            pass
        
        # Get tasks assigned to this worker
        tasks = []
        try:
            tasks_result = self.supabase.table('tasks').select('*').eq('worker_id', worker_id).order('created_at', desc=True).limit(20).execute()
            tasks = tasks_result.data or []
        except:
            pass
        
        return WorkerInfo(
            worker_id=worker_id,
            state=state,
            logs=logs,
            tasks=tasks
        )
    
    def check_worker_logging(self, worker_id: str) -> Dict[str, Any]:
        """Check if worker has started logging."""
        logs = self.log_client.get_logs(
            worker_id=worker_id,
            limit=10,
            order_desc=True
        )
        
        return {
            'is_logging': len(logs) > 0,
            'log_count': len(logs),
            'recent_logs': logs[:5] if logs else []
        }
    
    def check_worker_disk_space(self, worker_id: str) -> Dict[str, Any]:
        """Check disk space on a live worker via SSH (not supported in this setup)."""
        return {'available': False, 'error': 'SSH disk check not supported in this setup'}
    
    # ==================== MULTI-TASK METHODS ====================
    
    def get_recent_tasks(
        self,
        limit: int = 50,
        status: Optional[str] = None,
        task_type: Optional[str] = None,
        worker_id: Optional[str] = None,
        hours: Optional[int] = None
    ) -> TasksSummary:
        """Get recent tasks with analysis."""
        # Build query
        query = self.supabase.table('tasks').select('*')
        
        if status:
            query = query.eq('status', status)
        if task_type:
            query = query.eq('task_type', task_type)
        if worker_id:
            query = query.eq('worker_id', worker_id)
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
        
        # Get recent failures from logs
        recent_failures = []
        if hours:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        else:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        
        error_logs = self.log_client.get_logs(
            start_time=cutoff,
            log_level='ERROR',
            limit=100
        )
        
        # Extract task failures
        for log in error_logs:
            if log.get('task_id'):
                recent_failures.append({
                    'task_id': log['task_id'],
                    'timestamp': log['timestamp'],
                    'message': log['message'],
                    'worker_id': log.get('worker_id')
                })
        
        return TasksSummary(
            tasks=tasks,
            total_count=len(tasks),
            status_distribution=dict(status_dist),
            task_type_distribution=dict(type_dist),
            timing_stats=timing_stats,
            recent_failures=recent_failures[:10]
        )
    
    # ==================== MULTI-WORKER METHODS ====================
    
    def get_workers_summary(self, hours: int = 2, detailed: bool = False) -> WorkersSummary:
        """Get workers summary with health status."""
        # Workers table may not exist in this setup
        workers = []
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            result = self.supabase.table('workers').select('*').gte('created_at', cutoff_time.isoformat()).order('created_at', desc=True).execute()
            workers = result.data or []
        except:
            pass
        
        # Calculate statistics
        now = datetime.now(timezone.utc)
        status_counts = Counter(w.get('status') for w in workers)
        
        active_healthy = 0
        active_stale = 0
        
        for worker in workers:
            if worker.get('status') == 'active':
                last_hb = worker.get('last_heartbeat')
                if last_hb:
                    try:
                        hb_time = datetime.fromisoformat(last_hb.replace('Z', '+00:00'))
                        age_seconds = (now - hb_time).total_seconds()
                        if age_seconds < 60:
                            active_healthy += 1
                        else:
                            active_stale += 1
                    except:
                        active_stale += 1
        
        # Get recent failures
        recent_failures = []
        for worker in workers:
            if worker.get('status') in ['error', 'terminated']:
                metadata = worker.get('metadata', {})
                recent_failures.append({
                    'worker_id': worker['id'],
                    'status': worker['status'],
                    'created_at': worker.get('created_at'),
                    'error_reason': metadata.get('error_reason', 'Unknown') if isinstance(metadata, dict) else 'Unknown'
                })
        
        # Calculate failure rate
        failure_rate = None
        if len(workers) >= 5:
            failed = len([w for w in workers if w['status'] in ['error', 'terminated']])
            failure_rate = failed / len(workers)
        
        return WorkersSummary(
            workers=workers,
            total_count=len(workers),
            status_counts=dict(status_counts),
            active_healthy=active_healthy,
            active_stale=active_stale,
            recent_failures=recent_failures[:10],
            failure_rate=failure_rate
        )
    
    # ==================== SYSTEM HEALTH ====================
    
    def get_system_health(self) -> SystemHealth:
        """Get overall system health."""
        now = datetime.now(timezone.utc)
        
        # Get workers (may not exist)
        workers = []
        try:
            workers_result = self.supabase.table('workers').select('*').neq('status', 'terminated').execute()
            workers = workers_result.data or []
        except:
            pass
        
        workers_active = len([w for w in workers if w.get('status') == 'active'])
        workers_spawning = len([w for w in workers if w.get('status') == 'spawning'])
        
        # Count healthy workers
        workers_healthy = 0
        for worker in workers:
            if worker.get('status') == 'active' and worker.get('last_heartbeat'):
                try:
                    hb_time = datetime.fromisoformat(worker['last_heartbeat'].replace('Z', '+00:00'))
                    if (now - hb_time).total_seconds() < 60:
                        workers_healthy += 1
                except:
                    pass
        
        # Get task counts
        tasks_result = self.supabase.table('tasks').select('status').execute()
        tasks = tasks_result.data or []
        
        tasks_queued = len([t for t in tasks if t['status'] == 'Queued'])
        tasks_in_progress = len([t for t in tasks if t['status'] == 'In Progress'])
        
        # Get recent errors
        error_cutoff = now - timedelta(hours=1)
        recent_errors = self.log_client.get_logs(
            start_time=error_cutoff,
            log_level='ERROR',
            limit=50
        )[:10]
        
        return SystemHealth(
            timestamp=now,
            workers_active=workers_active,
            workers_spawning=workers_spawning,
            workers_healthy=workers_healthy,
            tasks_queued=tasks_queued,
            tasks_in_progress=tasks_in_progress,
            recent_errors=recent_errors,
            failure_rate=None,
            failure_rate_status='OK'
        )
    
    # ==================== ORCHESTRATOR STATUS ====================
    
    def get_orchestrator_status(self, hours: int = 1) -> OrchestratorStatus:
        """Get orchestrator status from logs."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # Get recent orchestrator logs
        logs = self.log_client.get_logs(
            start_time=cutoff,
            source_type='orchestrator_gpu',
            limit=1000,
            order_desc=True
        )
        
        # Find last activity and cycle
        last_activity = None
        last_cycle = None
        
        if logs:
            last_activity = datetime.fromisoformat(logs[0]['timestamp'].replace('Z', '+00:00'))
            last_cycle = logs[0].get('cycle_number')
        
        # Determine status
        if last_activity:
            age_minutes = (datetime.now(timezone.utc) - last_activity).total_seconds() / 60
            if age_minutes < 5:
                status = 'HEALTHY'
            elif age_minutes < 15:
                status = 'WARNING'
            else:
                status = 'STALE'
        else:
            status = 'NO_LOGS'
        
        # Extract cycle information
        cycle_starts = [log for log in logs if 'Starting orchestrator cycle' in log.get('message', '')]
        recent_cycles = []
        
        for log in cycle_starts[:10]:
            recent_cycles.append({
                'cycle_number': log.get('cycle_number'),
                'timestamp': log['timestamp'],
                'message': log['message']
            })
        
        return OrchestratorStatus(
            last_activity=last_activity,
            last_cycle=last_cycle,
            status=status,
            recent_cycles=recent_cycles,
            recent_logs=logs[:50]
        )
