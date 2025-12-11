"""Output formatting for debug data."""

import json
from datetime import datetime, timezone
from typing import Dict, Any, List
from debug.models import TaskInfo, TasksSummary


class Formatter:
    """Output formatting for debug data."""
    
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
                timestamp = log['timestamp'][11:19] if len(log.get('timestamp', '')) >= 19 else log.get('timestamp', '')
                level = log.get('log_level', 'INFO')
                source = log.get('source_id', 'unknown')[:20]
                message = log.get('message', '')[:100]
                
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
                for key, value in list(params.items())[:10]:
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:100] + "..."
                    lines.append(f"   {key}: {value_str}")
                if len(params) > 10:
                    lines.append(f"   ... and {len(params) - 10} more parameters")
        
        # Output / Error
        output_location = task.get('output_location')
        if output_location:
            if 'error' in output_location.lower() or 'failed' in output_location.lower():
                lines.append("\n❌ Error")
                lines.append(f"   {output_location}")
            else:
                lines.append("\n✅ Output")
                lines.append(f"   {output_location}")
        
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
            timestamp = log['timestamp'][11:19] if len(log.get('timestamp', '')) >= 19 else log.get('timestamp', '')
            level = log.get('log_level', 'INFO')
            message = log.get('message', '')
            
            lines.append(f"[{timestamp}] [{level:8}] {message}")
        
        return "\n".join(lines)
    
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
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
