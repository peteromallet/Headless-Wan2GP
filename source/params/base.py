"""
Base classes and utilities for typed parameter handling.

PRECEDENCE RULES (documented here as single source of truth):
- DB Tasks: top_level > orchestrator_details > orchestrator_payload
- Segments: individual_params > segment_params > orchestrator_payload
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class ParamGroup(ABC):
    """
    Base class for parameter groups.
    
    Subclasses must implement:
    - from_params(params, **context) -> cls
    - to_wgp_format() -> Dict[str, Any]
    """
    
    @classmethod
    @abstractmethod
    def from_params(cls, params: Dict[str, Any], **context) -> 'ParamGroup':
        """Parse parameters from a dict. Context can include task_id, model, etc."""
        pass
    
    @abstractmethod
    def to_wgp_format(self) -> Dict[str, Any]:
        """Convert to WGP-compatible format."""
        pass
    
    def validate(self) -> List[str]:
        """Return list of validation errors (empty if valid)."""
        return []
    
    @staticmethod
    def _get_first_of(params: Dict[str, Any], *keys, default=None):
        """Get first non-None value from a list of possible keys."""
        for key in keys:
            if key in params and params[key] is not None:
                return params[key]
        return default
    
    @staticmethod
    def _parse_list(value, separator=',') -> List[str]:
        """Parse a value that could be a list or comma-separated string."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [x.strip() for x in value.split(separator) if x.strip()]
        return [str(value)]
    
    @staticmethod
    def flatten_params(db_params: Dict[str, Any], task_id: str = "") -> Dict[str, Any]:
        """
        Flatten nested DB params with documented precedence.
        
        Precedence: top_level > orchestrator_details > orchestrator_payload
        """
        result = {}
        
        # Start with orchestrator_payload (lowest precedence)
        if "orchestrator_payload" in db_params and isinstance(db_params["orchestrator_payload"], dict):
            result.update(db_params["orchestrator_payload"])
        
        # Then orchestrator_details (medium precedence)
        if "orchestrator_details" in db_params and isinstance(db_params["orchestrator_details"], dict):
            result.update(db_params["orchestrator_details"])
        
        # Finally top-level params (highest precedence)
        for key, value in db_params.items():
            if key not in ("orchestrator_payload", "orchestrator_details"):
                result[key] = value
        
        return result


def warn_similar_key(params: Dict[str, Any], expected: str, alternatives: List[str], task_id: str = ""):
    """Log a warning if we find a similar but not exact key."""
    for alt in alternatives:
        if alt in params and expected not in params:
            logger.warning(f"Task {task_id}: Found '{alt}' but expected '{expected}' - using '{alt}'")
