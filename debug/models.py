"""Data models for debug tool."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class TaskInfo:
    """Complete task information."""
    task_id: str
    state: Optional[Dict[str, Any]]
    logs: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'state': self.state,
            'logs': self.logs
        }


@dataclass
class TasksSummary:
    """Summary of multiple tasks."""
    tasks: List[Dict[str, Any]]
    total_count: int
    status_distribution: Dict[str, int]
    task_type_distribution: Dict[str, int]
    timing_stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tasks': self.tasks,
            'total_count': self.total_count,
            'status_distribution': self.status_distribution,
            'task_type_distribution': self.task_type_distribution,
            'timing_stats': self.timing_stats
        }
