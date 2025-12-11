"""Output formatting for debug data."""

import json
from datetime import datetime, timezone
from typing import Dict, Any, List
from debug.models import (
    TaskInfo, WorkerInfo, TasksSummary, WorkersSummary,
    SystemHealth, OrchestratorStatus
)


class Formatter:
    """Output formatting for debug data."""
    
    # ==================== HELPER METHODS ====================
    
    @staticmethod
    def _extract_lora_urls(params: Dict[str, Any]) -> List[str]:
        """Extract all LoRA URLs from task parameters."""
        urls = []
        
        if not params:
            return urls
        
        # Check phase_config
        phase_config = params.get('phase_config', {})
        if phase_config and isinstance(phase_config, dict):
            phases = phase_config.get('phases', [])
            for phase in phases:
                if isinstance(phase, dict):
                    loras = phase.get('loras', [])
                    for lora in loras:
                        if isinstance(lora, dict) and 'url' in lora:
                            url = lora['url']
                            if url not in urls:  # Deduplicate
                                urls.append(url)
        
        # Check additional_loras
        additional_loras = params.get('additional_loras', {})
        if additional_loras and isinstance(additional_loras, dict):
            for url in additional_loras.keys():
                if url not in urls:
                    urls.append(url)
        
        return urls
    
    # ==================== TASK FORMATTING ====================
    
    @staticmethod
    def format_task(info: TaskInfo, format_type: str = 'text', logs_only: bool = False) -> str:
        """Format task information."""
        if format_type == 'json':
            return json.dumps(info.to_dict(), indent=2, default=str)
        
        if logs_only:
            return Formatter._format_task_logs_only(info)
        
        return Formatter._format_task_text(info)
    
    @staticmethod
    def _format_task_text(info: TaskInfo) -> str:
        """Format task info as human-readable text."""
        lines = []
        
        lines.append("=" * 80)
        lines.append(f"📋 TASK: {info.task_id}")
        lines.append("=" * 80)
        
        if not info.state:
            lines.append("\n❌ Task not found in database")
            return "\n".join(lines)
        
        task = info.state
        
        # Overview section
        lines.append("\n🏷️  Overview")
        lines.append(f"   Status: {task.get('status', 'Unknown')}")
        lines.append(f"   Type: {task.get('task_type', 'Unknown')}")
        lines.append(f"   Worker: {task.get('worker_id', 'None')}")
        lines.append(f"   Attempts: {task.get('attempts', 0)}")
        
        # Timing section
        lines.append("\n⏱️  Timing")
        created_at = task.get('created_at')
        started_at = task.get('generation_started_at')
        processed_at = task.get('generation_processed_at')
        
        if created_at:
            lines.append(f"   Created: {created_at}")
            
            if started_at:
                try:
                    created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    started = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                    queue_seconds = (started - created).total_seconds()
                    lines.append(f"   Started: {started_at} (queue: {queue_seconds:.1f}s)")
                    
                    if processed_at:
                        processed = datetime.fromisoformat(processed_at.replace('Z', '+00:00'))
                        processing_seconds = (processed - started).total_seconds()
                        total_seconds = (processed - created).total_seconds()
                        lines.append(f"   Processed: {processed_at} (processing: {processing_seconds:.1f}s)")
                        lines.append(f"   Total: {total_seconds:.1f}s")
                    else:
                        now = datetime.now(timezone.utc)
                        running_seconds = (now - started).total_seconds()
                        lines.append(f"   ⚠️  Never processed (running: {running_seconds:.1f}s)")
                except Exception as e:
                    lines.append(f"   Error parsing timestamps: {e}")
            else:
                lines.append("   ⚠️  Never started")
        
        # Event Timeline from logs
        if info.logs:
            lines.append("\n📜 Event Timeline (from system_logs)")
            lines.append(f"   Found {len(info.logs)} log entries")
            lines.append("")
            
            for log in info.logs[:50]:  # Show first 50
                timestamp = log['timestamp'][11:19] if len(log['timestamp']) >= 19 else log['timestamp']
                level = log['log_level']
                source = log.get('source_id', 'unknown')[:20]
                message = log['message'][:100]
                
                # Color code by level
                level_symbol = {
                    'ERROR': '❌',
                    'WARNING': '⚠️',
                    'INFO': 'ℹ️',
                    'DEBUG': '🔍',
                    'CRITICAL': '🔥'
                }.get(level, '  ')
                
                lines.append(f"   [{timestamp}] {level_symbol} [{level:8}] [{source:20}] {message}")
            
            if len(info.logs) > 50:
                lines.append(f"\n   ... and {len(info.logs) - 50} more log entries")
        else:
            lines.append("\n📜 Event Timeline")
            lines.append("   No logs found for this task")
        
        # Parameters
        params = task.get('params')
        if params:
            lines.append("\n📝 Parameters")
            if isinstance(params, dict):
                for key, value in list(params.items())[:10]:  # Show first 10
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:100] + "..."
                    lines.append(f"   {key}: {value_str}")
                if len(params) > 10:
                    lines.append(f"   ... and {len(params) - 10} more parameters")
        
        # Error Analysis
        error_msg = task.get('error_message')
        output_location = task.get('output_location')
        
        if error_msg:
            lines.append("\n❌ Error Analysis")
            lines.append(f"   Error: {error_msg}")
            
            # Find last error log
            error_logs = [log for log in info.logs if log['log_level'] == 'ERROR']
            if error_logs:
                last_error = error_logs[-1]
                lines.append(f"   Last error log: {last_error['message'][:100]}")
        elif output_location:
            # Check if output_location contains an error message
            if output_location and 'failed' in output_location.lower():
                lines.append("\n❌ Error Analysis")
                lines.append(f"   Output (contains error): {output_location}")
                
                # Check if error message appears incomplete
                if output_location.endswith(': ') or output_location.endswith(':'):
                    lines.append("   ⚠️  ERROR MESSAGE APPEARS INCOMPLETE - missing filename or details")
                
                # Extract LoRA URLs from params if this is a LoRA loading error
                if 'safetensors' in output_location.lower() or 'lora' in output_location.lower():
                    lines.append("\n📎 LoRA URLs in Configuration:")
                    lora_urls = Formatter._extract_lora_urls(params)
                    if lora_urls:
                        for i, url in enumerate(lora_urls, 1):
                            lines.append(f"   {i}. {url}")
                    else:
                        lines.append("   No LoRAs found in phase_config")
                
                # Show surrounding error context
                error_logs = [log for log in info.logs if log['log_level'] in ['ERROR', 'WARNING']]
                if error_logs:
                    lines.append("\n🔍 Error Context (all ERROR/WARNING logs):")
                    for err in error_logs[-5:]:  # Last 5 error/warning logs
                        timestamp = err['timestamp'][11:19]
                        level = err['log_level']
                        msg = err['message'][:120]
                        lines.append(f"   [{timestamp}] {level}: {msg}")
            else:
                lines.append("\n✅ Success")
                lines.append(f"   Output: {output_location}")
        
        # Related Resources
        lines.append("\n💡 Related Resources")
        if task.get('worker_id'):
            lines.append(f"   Worker: python scripts/debug.py worker {task['worker_id']}")
        lines.append(f"   All logs: python scripts/query_logs.py --task {info.task_id}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_task_logs_only(info: TaskInfo) -> str:
        """Format only the task logs timeline."""
        lines = []
        
        lines.append(f"📜 Event Timeline for Task: {info.task_id}")
        lines.append("=" * 80)
        
        if not info.logs:
            lines.append("No logs found")
            return "\n".join(lines)
        
        for log in info.logs:
            timestamp = log['timestamp'][11:19] if len(log['timestamp']) >= 19 else log['timestamp']
            level = log['log_level']
            message = log['message']
            
            lines.append(f"[{timestamp}] [{level:8}] {message}")
        
        return "\n".join(lines)
    
    # ==================== WORKER FORMATTING ====================
    
    @staticmethod
    def format_worker(info: WorkerInfo, format_type: str = 'text', logs_only: bool = False) -> str:
        """Format worker information."""
        if format_type == 'json':
            return json.dumps(info.to_dict(), indent=2, default=str)
        
        if logs_only:
            return Formatter._format_worker_logs_only(info)
        
        return Formatter._format_worker_text(info)
    
    @staticmethod
    def _format_worker_text(info: WorkerInfo) -> str:
        """Format worker info as human-readable text."""
        lines = []
        
        lines.append("=" * 80)
        lines.append(f"👷 WORKER: {info.worker_id}")
        lines.append("=" * 80)
        
        if not info.state:
            lines.append("\n❌ Worker not found in database")
            return "\n".join(lines)
        
        worker = info.state
        now = datetime.now(timezone.utc)
        
        # Overview section
        lines.append("\n🏷️  Overview")
        lines.append(f"   Status: {worker.get('status', 'Unknown')}")
        
        created_at = worker.get('created_at')
        if created_at:
            try:
                created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                age_minutes = (now - created).total_seconds() / 60
                lines.append(f"   Created: {created_at} (age: {age_minutes:.1f}m)")
            except:
                lines.append(f"   Created: {created_at}")
        
        last_hb = worker.get('last_heartbeat')
        if last_hb:
            try:
                hb_time = datetime.fromisoformat(last_hb.replace('Z', '+00:00'))
                hb_age = (now - hb_time).total_seconds()
                
                if hb_age < 60:
                    status = '✅ HEALTHY'
                elif hb_age < 300:
                    status = '⚠️  WARNING'
                else:
                    status = '❌ STALE'
                
                lines.append(f"   Last Heartbeat: {last_hb} ({hb_age:.0f}s ago) {status}")
            except:
                lines.append(f"   Last Heartbeat: {last_hb}")
        else:
            lines.append("   Last Heartbeat: None")
        
        metadata = worker.get('metadata', {})
        lines.append(f"   RunPod ID: {metadata.get('runpod_id', 'N/A')}")
        lines.append(f"   RAM Tier: {metadata.get('ram_tier', 'Unknown')}GB")
        
        # Diagnostics section (if available)
        diagnostics = metadata.get('diagnostics', {})
        if diagnostics:
            lines.append("\n🔬 Pre-Termination Diagnostics")
            
            # VRAM usage
            if 'vram_total_mb' in diagnostics:
                vram_used = diagnostics.get('vram_used_mb', 0)
                vram_total = diagnostics.get('vram_total_mb', 0)
                vram_percent = diagnostics.get('vram_usage_percent', 0)
                lines.append(f"   VRAM: {vram_used}/{vram_total} MB ({vram_percent:.1f}%)")
                vram_ts = diagnostics.get('vram_timestamp', '')
                if vram_ts:
                    lines.append(f"   VRAM Timestamp: {vram_ts[:19]}")
            
            # Running tasks at failure
            running_tasks = diagnostics.get('running_tasks', [])
            running_count = diagnostics.get('running_tasks_count', 0)
            if running_count > 0:
                lines.append(f"   Running Tasks: {running_count}")
                for i, task in enumerate(running_tasks[:3], 1):
                    task_id = str(task.get('id', 'unknown'))[:8]
                    task_type = task.get('task_type', 'unknown')
                    age = task.get('age_seconds', 0)
                    lines.append(f"     {i}. {task_id}... ({task_type}) - {age:.0f}s old")
            else:
                lines.append(f"   Running Tasks: 0")
            
            # Pod status
            pod_status = diagnostics.get('pod_status', {})
            if pod_status:
                desired = pod_status.get('desired_status', 'N/A')
                actual = pod_status.get('actual_status', 'N/A')
                uptime = pod_status.get('uptime_seconds', 0)
                cost = pod_status.get('cost_per_hr', 0)
                lines.append(f"   Pod Status: {desired} / {actual}")
                lines.append(f"   Pod Uptime: {uptime}s (${cost:.3f}/hr)")
            
            # Collection status
            if diagnostics.get('collection_success'):
                lines.append(f"   ✅ Diagnostics collected successfully")
            elif 'collection_error' in diagnostics:
                lines.append(f"   ❌ Collection error: {diagnostics['collection_error']}")
        elif worker.get('status') in ['error', 'terminated']:
            lines.append("\n🔬 Pre-Termination Diagnostics")
            lines.append("   ⚠️  No diagnostics collected")
            lines.append("   (Worker may have failed before collection was implemented)")
        
        # Event Timeline from logs
        if info.logs:
            lines.append("\n📜 Event Timeline (from system_logs)")
            lines.append(f"   Found {len(info.logs)} log entries")
            lines.append("")
            
            for log in info.logs[:50]:  # Show first 50
                timestamp = log['timestamp'][11:19] if len(log['timestamp']) >= 19 else log['timestamp']
                level = log['log_level']
                message = log['message'][:80]
                
                level_symbol = {
                    'ERROR': '❌',
                    'WARNING': '⚠️',
                    'INFO': 'ℹ️',
                    'DEBUG': '🔍'
                }.get(level, '  ')
                
                lines.append(f"   [{timestamp}] {level_symbol} [{level:8}] {message}")
            
            if len(info.logs) > 50:
                lines.append(f"\n   ... and {len(info.logs) - 50} more log entries")
        else:
            lines.append("\n📜 Event Timeline")
            lines.append("   No logs found for this worker")
        
        # Tasks Processed
        if info.tasks:
            lines.append("\n📋 Tasks Processed (Recent 20)")
            for task in info.tasks[:20]:
                task_id = task['id'][:8]
                task_type = task.get('task_type', 'unknown')[:25]
                status = task.get('status', 'unknown')
                lines.append(f"   {task_id}... | {task_type:25} | {status}")
        else:
            lines.append("\n📋 Tasks Processed")
            lines.append("   No tasks assigned")
        
        # Issues Detected
        error_logs = [log for log in info.logs if log['log_level'] == 'ERROR']
        if error_logs:
            lines.append("\n❌ Issues Detected")
            lines.append(f"   Error logs: {len(error_logs)}")
            for err in error_logs[:3]:
                lines.append(f"   - {err['message'][:80]}")
        
        # Related Resources
        lines.append("\n💡 Related Resources")
        lines.append(f"   All logs: python scripts/query_logs.py --worker {info.worker_id}")
        if info.tasks:
            lines.append(f"   Tasks: python scripts/debug.py tasks --worker {info.worker_id}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_worker_logs_only(info: WorkerInfo) -> str:
        """Format only worker logs timeline."""
        lines = []
        
        lines.append(f"📜 Event Timeline for Worker: {info.worker_id}")
        lines.append("=" * 80)
        
        if not info.logs:
            lines.append("No logs found")
            return "\n".join(lines)
        
        for log in info.logs:
            timestamp = log['timestamp'][11:19] if len(log['timestamp']) >= 19 else log['timestamp']
            level = log['log_level']
            message = log['message']
            
            lines.append(f"[{timestamp}] [{level:8}] {message}")
        
        return "\n".join(lines)
    
    # ==================== TASKS SUMMARY FORMATTING ====================
    
    @staticmethod
    def format_tasks_summary(summary: TasksSummary, format_type: str = 'text') -> str:
        """Format tasks summary."""
        if format_type == 'json':
            return json.dumps(summary.to_dict(), indent=2, default=str)
        
        lines = []
        
        lines.append("=" * 80)
        lines.append("📊 RECENT TASKS ANALYSIS")
        lines.append("=" * 80)
        
        lines.append(f"\n📈 Overview")
        lines.append(f"   Total tasks: {summary.total_count}")
        
        if summary.tasks:
            oldest = summary.tasks[-1].get('created_at', '')
            newest = summary.tasks[0].get('created_at', '')
            lines.append(f"   Time range: {oldest[:19]} to {newest[:19]}")
        
        # Status Distribution
        lines.append(f"\n📊 Status Distribution")
        for status, count in sorted(summary.status_distribution.items()):
            percentage = (count / summary.total_count * 100) if summary.total_count > 0 else 0
            lines.append(f"   {status}: {count} ({percentage:.1f}%)")
        
        # Task Types
        if summary.task_type_distribution:
            lines.append(f"\n🔧 Task Types")
            sorted_types = sorted(summary.task_type_distribution.items(), key=lambda x: x[1], reverse=True)
            for task_type, count in sorted_types[:10]:
                percentage = (count / summary.total_count * 100) if summary.total_count > 0 else 0
                lines.append(f"   {task_type}: {count} ({percentage:.1f}%)")
        
        # Timing Analysis
        timing = summary.timing_stats
        if timing.get('avg_processing_seconds') or timing.get('avg_queue_seconds'):
            lines.append(f"\n⏱️  Timing Analysis")
            if timing.get('avg_queue_seconds'):
                lines.append(f"   Avg Queue Time: {timing['avg_queue_seconds']:.1f}s")
            if timing.get('avg_processing_seconds'):
                lines.append(f"   Avg Processing Time: {timing['avg_processing_seconds']:.1f}s")
            lines.append(f"   Tasks with timing: {timing['total_with_timing']}")
        
        # Recent Failures
        if summary.recent_failures:
            lines.append(f"\n❌ Recent Failures")
            for failure in summary.recent_failures[:5]:
                task_id = str(failure.get('task_id', 'unknown'))[:8]
                message = failure.get('message', 'Unknown error')[:60]
                lines.append(f"   {task_id}... | {message}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    # ==================== WORKERS SUMMARY FORMATTING ====================
    
    @staticmethod
    def format_workers_summary(summary: WorkersSummary, format_type: str = 'text') -> str:
        """Format workers summary."""
        if format_type == 'json':
            return json.dumps(summary.to_dict(), indent=2, default=str)
        
        lines = []
        
        lines.append("=" * 80)
        lines.append("👷 WORKERS STATUS")
        lines.append("=" * 80)
        
        lines.append(f"\n📊 Summary")
        for status, count in sorted(summary.status_counts.items()):
            lines.append(f"   {status}: {count}")
        
        lines.append(f"\n✅ Active Workers Health")
        lines.append(f"   Healthy (HB < 60s): {summary.active_healthy}")
        lines.append(f"   Stale (HB > 60s): {summary.active_stale}")
        
        # Show active workers
        active_workers = [w for w in summary.workers if w['status'] == 'active']
        if active_workers:
            lines.append(f"\n📋 Active Workers")
            for worker in active_workers[:10]:
                worker_id = worker['id']
                created = worker.get('created_at', '')[:19]
                last_hb = worker.get('last_heartbeat', 'never')
                
                if last_hb != 'never':
                    try:
                        hb_time = datetime.fromisoformat(last_hb.replace('Z', '+00:00'))
                        now = datetime.now(timezone.utc)
                        hb_age = (now - hb_time).total_seconds()
                        
                        if hb_age < 60:
                            status_symbol = '✅'
                        elif hb_age < 300:
                            status_symbol = '⚠️'
                        else:
                            status_symbol = '❌'
                        
                        hb_str = f"{hb_age:.0f}s ago"
                    except:
                        status_symbol = '❓'
                        hb_str = last_hb[:19]
                else:
                    status_symbol = '❓'
                    hb_str = 'no HB'
                
                lines.append(f"   {status_symbol} {worker_id} | Created: {created} | HB: {hb_str}")
        
        # Recent Failures
        if summary.recent_failures:
            lines.append(f"\n❌ Recently Failed ({len(summary.recent_failures)})")
            for failure in summary.recent_failures[:10]:
                worker_id = failure.get('worker_id', 'unknown')
                reason = failure.get('error_reason', 'Unknown')
                lines.append(f"   {worker_id} | {reason}")
        
        # Failure Rate Analysis
        if summary.failure_rate is not None:
            lines.append(f"\n📈 Failure Rate Analysis")
            lines.append(f"   Failure rate: {summary.failure_rate:.1%}")
            lines.append(f"   Threshold: 80.0%")
            
            if summary.failure_rate > 0.8:
                lines.append(f"   Status: ❌ BLOCKED - spawning disabled")
            else:
                lines.append(f"   Status: ✅ OK - spawning allowed")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    # ==================== SYSTEM HEALTH FORMATTING ====================
    
    @staticmethod
    def format_health(health: SystemHealth, format_type: str = 'text') -> str:
        """Format system health."""
        if format_type == 'json':
            return json.dumps(health.to_dict(), indent=2, default=str)
        
        lines = []
        
        lines.append("=" * 80)
        lines.append("🔍 SYSTEM HEALTH CHECK")
        lines.append("=" * 80)
        lines.append(f"Time: {health.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("")
        
        lines.append("📊 WORKER STATUS")
        lines.append("-" * 80)
        lines.append(f"Active: {health.workers_active}")
        lines.append(f"Spawning: {health.workers_spawning}")
        lines.append(f"Healthy: {health.workers_healthy}")
        
        if health.workers_active == 0 and health.workers_spawning == 0:
            lines.append("⚠️  WARNING: No active or spawning workers!")
        elif health.workers_healthy > 0:
            lines.append(f"✅ {health.workers_healthy} healthy workers")
        
        lines.append("")
        lines.append("📝 TASK STATUS")
        lines.append("-" * 80)
        lines.append(f"Queued: {health.tasks_queued}")
        lines.append(f"In Progress: {health.tasks_in_progress}")
        
        if health.tasks_queued > 0 and health.workers_active == 0:
            lines.append("⚠️  Tasks queued but no active workers!")
        
        if health.failure_rate is not None:
            lines.append("")
            lines.append("📈 FAILURE RATE ANALYSIS")
            lines.append("-" * 80)
            lines.append(f"Failure rate: {health.failure_rate:.1%}")
            lines.append(f"Status: {health.failure_rate_status}")
            
            if health.failure_rate_status == 'BLOCKED':
                lines.append("❌ Worker spawning is BLOCKED due to high failure rate")
        
        if health.recent_errors:
            lines.append("")
            lines.append("❌ RECENT ERRORS (Last Hour)")
            lines.append("-" * 80)
            for error in health.recent_errors[:5]:
                timestamp = error['timestamp'][11:19]
                source = error.get('source_id', 'unknown')[:20]
                message = error['message'][:60]
                lines.append(f"[{timestamp}] {source:20} | {message}")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    # ==================== ORCHESTRATOR STATUS FORMATTING ====================
    
    @staticmethod
    def format_orchestrator(status: OrchestratorStatus, format_type: str = 'text') -> str:
        """Format orchestrator status."""
        if format_type == 'json':
            return json.dumps(status.to_dict(), indent=2, default=str)
        
        lines = []
        
        lines.append("=" * 80)
        lines.append("🔄 ORCHESTRATOR STATUS")
        lines.append("=" * 80)
        
        lines.append(f"\n📊 Status")
        
        if status.last_activity:
            age_minutes = (datetime.now(timezone.utc) - status.last_activity).total_seconds() / 60
            lines.append(f"   Last Activity: {status.last_activity.strftime('%Y-%m-%d %H:%M:%S')} ({age_minutes:.1f}m ago)")
        else:
            lines.append("   Last Activity: None")
        
        if status.last_cycle:
            lines.append(f"   Last Cycle: #{status.last_cycle}")
        
        lines.append(f"   Status: {status.status}")
        
        if status.status == 'HEALTHY':
            lines.append("   ✅ Orchestrator is running normally")
        elif status.status == 'WARNING':
            lines.append("   ⚠️  Orchestrator activity is stale")
        elif status.status == 'STALE':
            lines.append("   ❌ Orchestrator may have stopped")
        else:
            lines.append("   ❌ No orchestrator logs found")
        
        # Recent Cycles
        if status.recent_cycles:
            lines.append(f"\n🔄 Recent Cycles")
            for cycle in status.recent_cycles[:10]:
                timestamp = cycle['timestamp'][11:19]
                cycle_num = cycle['cycle_number']
                lines.append(f"   Cycle #{cycle_num} at {timestamp}")
        
        # Recent Activity
        if status.recent_logs:
            lines.append(f"\n📜 Recent Activity (Last 10)")
            for log in status.recent_logs[:10]:
                timestamp = log['timestamp'][11:19]
                level = log['log_level']
                message = log['message'][:60]
                lines.append(f"   [{timestamp}] [{level:8}] {message}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)

