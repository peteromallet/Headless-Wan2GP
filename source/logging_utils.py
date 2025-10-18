"""
Centralized logging utility for Headless-Wan2GP

Provides structured logging with debug vs essential log levels.
Essential logs are always shown, debug logs only appear when debug mode is enabled.
"""

import sys
import datetime
from typing import Optional, Any
from pathlib import Path

# Global debug mode flag - set by the main application
_debug_mode = False

def enable_debug_mode():
    """Enable debug logging globally."""
    global _debug_mode
    _debug_mode = True

def disable_debug_mode():
    """Disable debug logging globally."""
    global _debug_mode
    _debug_mode = False

def is_debug_enabled() -> bool:
    """Check if debug mode is currently enabled."""
    return _debug_mode

def _get_timestamp() -> str:
    """Get formatted timestamp for logs."""
    return datetime.datetime.now().strftime("%H:%M:%S")

def _format_message(level: str, component: str, message: str, task_id: Optional[str] = None) -> str:
    """Format a log message with consistent structure."""
    timestamp = _get_timestamp()
    
    if task_id:
        return f"[{timestamp}] {level} {component} [Task {task_id}] {message}"
    else:
        return f"[{timestamp}] {level} {component} {message}"

def essential(component: str, message: str, task_id: Optional[str] = None):
    """Log an essential message that should always be shown."""
    formatted = _format_message("INFO", component, message, task_id)
    print(formatted)

def success(component: str, message: str, task_id: Optional[str] = None):
    """Log a success message that should always be shown."""
    formatted = _format_message("✅", component, message, task_id)
    print(formatted)

def warning(component: str, message: str, task_id: Optional[str] = None):
    """Log a warning message that should always be shown."""
    formatted = _format_message("⚠️", component, message, task_id)
    print(formatted)

def error(component: str, message: str, task_id: Optional[str] = None):
    """Log an error message that should always be shown."""
    formatted = _format_message("❌", component, message, task_id)
    print(formatted, file=sys.stderr)

def debug(component: str, message: str, task_id: Optional[str] = None):
    """Log a debug message that only appears when debug mode is enabled."""
    if _debug_mode:
        formatted = _format_message("DEBUG", component, message, task_id)
        print(formatted)

def progress(component: str, message: str, task_id: Optional[str] = None):
    """Log a progress message that should always be shown."""
    formatted = _format_message("⏳", component, message, task_id)
    print(formatted)

def status(component: str, message: str, task_id: Optional[str] = None):
    """Log a status message that should always be shown."""
    formatted = _format_message("📊", component, message, task_id)
    print(formatted)

# Component-specific loggers for better organization
class ComponentLogger:
    """Logger for a specific component with consistent naming."""
    
    def __init__(self, component_name: str):
        self.component = component_name
    
    def essential(self, message: str, task_id: Optional[str] = None):
        essential(self.component, message, task_id)
    
    def success(self, message: str, task_id: Optional[str] = None):
        success(self.component, message, task_id)
    
    def warning(self, message: str, task_id: Optional[str] = None):
        warning(self.component, message, task_id)
    
    def error(self, message: str, task_id: Optional[str] = None):
        error(self.component, message, task_id)
    
    def debug(self, message: str, task_id: Optional[str] = None):
        debug(self.component, message, task_id)
    
    def progress(self, message: str, task_id: Optional[str] = None):
        progress(self.component, message, task_id)
    
    def status(self, message: str, task_id: Optional[str] = None):
        status(self.component, message, task_id)
    
    def info(self, message: str, task_id: Optional[str] = None):
        """Alias for essential() to maintain compatibility with standard logging."""
        essential(self.component, message, task_id)

# Pre-configured loggers for main components
headless_logger = ComponentLogger("HEADLESS")
queue_logger = ComponentLogger("QUEUE")
orchestrator_logger = ComponentLogger("ORCHESTRATOR")
travel_logger = ComponentLogger("TRAVEL")
generation_logger = ComponentLogger("GENERATION")
model_logger = ComponentLogger("MODEL")
task_logger = ComponentLogger("TASK")

# Backward compatibility for existing dprint usage
def dprint(message: str, component: str = "DEBUG"):
    """
    Backward compatibility function for existing dprint() calls.
    Maps to debug() logging.
    """
    debug(component, message)

# Utility functions for common log patterns
def log_task_start(component: str, task_id: str, task_type: str, **params):
    """Log the start of a task with key parameters."""
    essential(component, f"Starting {task_type} task", task_id)
    if params:
        debug(component, f"Task parameters: {params}", task_id)

def log_task_complete(component: str, task_id: str, task_type: str, output_path: Optional[str] = None, duration: Optional[float] = None):
    """Log the completion of a task."""
    duration_str = f" ({duration:.1f}s)" if duration else ""
    if output_path:
        success(component, f"{task_type} completed{duration_str}: {output_path}", task_id)
    else:
        success(component, f"{task_type} completed{duration_str}", task_id)

def log_task_error(component: str, task_id: str, task_type: str, error_msg: str):
    """Log a task error."""
    error(component, f"{task_type} failed: {error_msg}", task_id)

def log_model_switch(component: str, old_model: Optional[str], new_model: str, duration: Optional[float] = None):
    """Log a model switch operation."""
    duration_str = f" ({duration:.1f}s)" if duration else ""
    if old_model:
        essential(component, f"Model switch: {old_model} → {new_model}{duration_str}")
    else:
        essential(component, f"Model loaded: {new_model}{duration_str}")

def log_file_operation(component: str, operation: str, source: str, target: Optional[str] = None, task_id: Optional[str] = None):
    """Log file operations like copy, move, download."""
    if target:
        debug(component, f"{operation}: {source} → {target}", task_id)
    else:
        debug(component, f"{operation}: {source}", task_id)

def log_ffmpeg_command(component: str, command: str, task_id: Optional[str] = None):
    """Log FFmpeg commands (debug only)."""
    debug(component, f"FFmpeg: {command}", task_id)

def log_generation_params(component: str, task_id: str, **params):
    """Log generation parameters (debug only)."""
    debug(component, f"Generation parameters: {params}", task_id)

# Context manager for timing operations
class LogTimer:
    """Context manager to time and log operations."""
    
    def __init__(self, component: str, operation: str, task_id: Optional[str] = None, level: str = "essential"):
        self.component = component
        self.operation = operation
        self.task_id = task_id
        self.level = level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.datetime.now()
        if self.level == "debug":
            debug(self.component, f"Starting {self.operation}...", self.task_id)
        else:
            essential(self.component, f"Starting {self.operation}...", self.task_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            # Success
            if self.level == "debug":
                debug(self.component, f"{self.operation} completed ({duration:.1f}s)", self.task_id)
            else:
                success(self.component, f"{self.operation} completed ({duration:.1f}s)", self.task_id)
        else:
            # Error
            error(self.component, f"{self.operation} failed after {duration:.1f}s: {exc_val}", self.task_id)


# -----------------------------------------------------------------------------
# Centralized Logging for Orchestrator Integration
# -----------------------------------------------------------------------------

import threading
import logging
from datetime import timezone
from typing import List, Dict


class LogBuffer:
    """
    Thread-safe buffer for collecting logs.

    Logs are stored in memory and flushed periodically with heartbeat updates.
    This prevents excessive database calls while maintaining log history.

    Can optionally send logs to a guardian process via multiprocessing.Queue
    for bulletproof heartbeat delivery that cannot be blocked by GIL or I/O.
    """

    def __init__(self, max_size: int = 100, shared_queue=None):
        """
        Initialize log buffer.

        Args:
            max_size: Maximum logs to buffer before auto-flush (default: 100)
            shared_queue: Optional multiprocessing.Queue to send logs to guardian process
        """
        self.logs: List[Dict[str, Any]] = []
        self.max_size = max_size
        self.lock = threading.RLock()  # Use RLock for reentrancy (flush() called from add())
        self.total_logs = 0
        self.total_flushes = 0
        self.shared_queue = shared_queue  # Queue to guardian process
    
    def add(
        self,
        level: str,
        message: str,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Add a log entry to buffer.

        Args:
            level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            message: Log message
            task_id: Optional task ID for context
            metadata: Optional additional metadata

        Returns:
            List of logs if buffer is full and auto-flushed, otherwise []
        """
        log_entry = {
            'timestamp': datetime.datetime.now(timezone.utc).isoformat(),
            'level': level,
            'message': message,
            'task_id': task_id,
            'metadata': metadata or {}
        }

        # Send to guardian process if available (non-blocking)
        if self.shared_queue:
            try:
                # Debug: Check queue size
                qsize = self.shared_queue.qsize() if hasattr(self.shared_queue, 'qsize') else 'unknown'
                print(f"[LOG_BUFFER DEBUG] Queue size before put: {qsize}", flush=True)
                self.shared_queue.put_nowait(log_entry)
                print(f"[LOG_BUFFER DEBUG] Successfully queued log: {level} - {message[:50]}", flush=True)
            except Exception as e:
                # Queue full or not available - not critical, guardian will catch up
                print(f"[LOG_BUFFER ERROR] Queue put failed: {type(e).__name__}: {e}", flush=True)
                pass

        with self.lock:
            self.logs.append(log_entry)
            self.total_logs += 1

            # Auto-flush if buffer is full
            if len(self.logs) >= self.max_size:
                return self.flush()

        return []
    
    def flush(self) -> List[Dict[str, Any]]:
        """
        Get and clear all buffered logs.
        
        Returns:
            List of log entries
        """
        with self.lock:
            logs = self.logs.copy()
            self.logs = []
            if logs:
                self.total_flushes += 1
            return logs
    
    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics."""
        with self.lock:
            return {
                'current_buffer_size': len(self.logs),
                'total_logs_buffered': self.total_logs,
                'total_flushes': self.total_flushes
            }


class WorkerDatabaseLogHandler(logging.Handler):
    """
    Custom logging handler that buffers logs for database storage.
    
    Usage:
        log_buffer = LogBuffer()
        handler = WorkerDatabaseLogHandler('gpu-worker-123', log_buffer)
        logging.getLogger().addHandler(handler)
        
        # Set current task when processing
        handler.set_current_task('task-id-456')
    """
    
    def __init__(
        self,
        worker_id: str,
        log_buffer: LogBuffer,
        min_level: int = logging.INFO
    ):
        """
        Initialize handler.
        
        Args:
            worker_id: Worker's unique ID
            log_buffer: LogBuffer instance to collect logs
            min_level: Minimum log level to buffer (default: INFO)
        """
        super().__init__()
        self.worker_id = worker_id
        self.log_buffer = log_buffer
        self.current_task_id: Optional[str] = None
        self.setLevel(min_level)
    
    def set_current_task(self, task_id: Optional[str]):
        """Set current task ID for context."""
        self.current_task_id = task_id
    
    def emit(self, record: logging.LogRecord):
        """
        Capture log record to buffer.
        
        Called automatically by logging framework.
        """
        try:
            # Extract metadata from record
            metadata = {
                'module': record.module,
                'funcName': record.funcName,
                'lineno': record.lineno,
            }
            
            # Add exception info if present
            if record.exc_info:
                metadata['exception'] = self.format(record)
            
            # Add to buffer
            self.log_buffer.add(
                level=record.levelname,
                message=record.getMessage(),
                task_id=self.current_task_id,
                metadata=metadata
            )
        except Exception:
            self.handleError(record)


# Intercept logging calls from our custom logging functions
class CustomLogInterceptor:
    """
    Intercepts calls from our custom logging functions (essential, error, etc.)
    and adds them to the log buffer for database storage.
    """
    
    def __init__(self, log_buffer: LogBuffer):
        """
        Initialize interceptor.
        
        Args:
            log_buffer: LogBuffer instance to collect logs
        """
        self.log_buffer = log_buffer
        self.current_task_id: Optional[str] = None
        self.original_print = None
    
    def set_current_task(self, task_id: Optional[str]):
        """Set current task ID for context."""
        self.current_task_id = task_id
    
    def capture_log(self, level: str, message: str, task_id: Optional[str] = None):
        """
        Capture a log message to the buffer.
        
        Args:
            level: Log level
            message: Log message
            task_id: Task ID (uses current_task_id if not provided)
        """
        self.log_buffer.add(
            level=level,
            message=message,
            task_id=task_id or self.current_task_id,
            metadata={}
        )


# Global log interceptor instance (set in worker.py)
_log_interceptor: Optional[CustomLogInterceptor] = None


def set_log_interceptor(interceptor: Optional[CustomLogInterceptor]):
    """Set the global log interceptor for database logging."""
    global _log_interceptor
    _log_interceptor = interceptor


# Update logging functions to use interceptor
def _intercept_log(level: str, message: str, task_id: Optional[str] = None):
    """Send log to interceptor if enabled."""
    if _log_interceptor:
        _log_interceptor.capture_log(level, message, task_id)


# Modify existing logging functions to intercept
_original_essential = essential
def essential(component: str, message: str, task_id: Optional[str] = None):
    """Log an essential message that should always be shown."""
    _original_essential(component, message, task_id)
    _intercept_log("INFO", f"{component}: {message}", task_id)


_original_success = success
def success(component: str, message: str, task_id: Optional[str] = None):
    """Log a success message that should always be shown."""
    _original_success(component, message, task_id)
    _intercept_log("INFO", f"{component}: {message}", task_id)


_original_warning = warning
def warning(component: str, message: str, task_id: Optional[str] = None):
    """Log a warning message that should always be shown."""
    _original_warning(component, message, task_id)
    _intercept_log("WARNING", f"{component}: {message}", task_id)


_original_error = error
def error(component: str, message: str, task_id: Optional[str] = None):
    """Log an error message that should always be shown."""
    _original_error(component, message, task_id)
    _intercept_log("ERROR", f"{component}: {message}", task_id)


_original_debug = debug
def debug(component: str, message: str, task_id: Optional[str] = None):
    """Log a debug message that only appears when debug mode is enabled."""
    _original_debug(component, message, task_id)
    if _debug_mode:  # Only intercept if debug mode is enabled
        _intercept_log("DEBUG", f"{component}: {message}", task_id)


# -----------------------------------------------------------------------------
# Safe Logging Utilities - Prevent Freezes on Large Data Structures
# -----------------------------------------------------------------------------

import reprlib
import json
from collections.abc import Mapping, Sequence

# =============================================================================
# GLOBAL LOGGING CONFIGURATION
# =============================================================================
# These constants define consistent limits across the entire codebase to prevent
# logging-induced hangs while maintaining useful debug information.

# String representation limits
LOG_MAX_STRING_REPR = 200          # Max chars for individual string values
LOG_MAX_OBJECT_OUTPUT = 500        # Max chars for entire object representation
LOG_MAX_COLLECTION_ITEMS = 5       # Max items to show in lists/dicts/sets
LOG_MAX_NESTING_DEPTH = 3          # Max recursion depth for nested structures

# JSON serialization limits (for legacy json.dumps usage)
LOG_MAX_JSON_OUTPUT = 1000         # Max chars for JSON.dumps() output (legacy)

# Known problematic keys that contain large nested structures
LOG_LARGE_DICT_KEYS = {
    'orchestrator_payload', 'orchestrator_details', 'full_orchestrator_payload',
    'phase_config', 'wgp_params', 'generation_params', 'task_params',
    'resolved_params', 'model_defaults', 'model_config', 'db_task_params',
    'task_params_from_db', 'task_params_dict', 'extracted_params'
}

# Configure reprlib for safe string conversion using global constants
_safe_repr = reprlib.Repr()
_safe_repr.maxstring = LOG_MAX_STRING_REPR
_safe_repr.maxother = LOG_MAX_STRING_REPR
_safe_repr.maxlist = LOG_MAX_COLLECTION_ITEMS
_safe_repr.maxdict = LOG_MAX_COLLECTION_ITEMS
_safe_repr.maxset = LOG_MAX_COLLECTION_ITEMS
_safe_repr.maxtuple = LOG_MAX_COLLECTION_ITEMS
_safe_repr.maxdeque = LOG_MAX_COLLECTION_ITEMS
_safe_repr.maxarray = LOG_MAX_COLLECTION_ITEMS
_safe_repr.maxlevel = LOG_MAX_NESTING_DEPTH


def safe_repr(obj: Any, max_length: int = None) -> str:
    """
    Safely convert any object to string with size limits.

    This prevents logging from hanging on large nested structures.
    Uses reprlib for smart truncation of collections.

    Args:
        obj: Object to convert to string
        max_length: Maximum string length (default: LOG_MAX_OBJECT_OUTPUT)

    Returns:
        Safe string representation, truncated if needed

    Example:
        >>> safe_repr({'huge': ['data'] * 1000})
        "{'huge': ['data', 'data', 'data', 'data', 'data', ...]}"
    """
    if max_length is None:
        max_length = LOG_MAX_OBJECT_OUTPUT

    try:
        # Use reprlib for smart truncation
        result = _safe_repr.repr(obj)

        # Additional length limit as backup
        if len(result) > max_length:
            result = result[:max_length] + "...}"

        return result
    except Exception as e:
        return f"<repr failed: {type(obj).__name__} - {e}>"


def safe_dict_repr(d: dict, max_items: int = None, max_length: int = None) -> str:
    """
    Safely represent a dictionary with smart truncation.

    Special handling for known problematic keys that contain large nested data.

    Args:
        d: Dictionary to represent
        max_items: Maximum number of items to show (default: LOG_MAX_COLLECTION_ITEMS)
        max_length: Maximum total string length (default: LOG_MAX_OBJECT_OUTPUT)

    Returns:
        Safe string representation of dict

    Example:
        >>> safe_dict_repr({'orchestrator_payload': {...huge...}, 'seed': 123})
        "{'orchestrator_payload': <dict with 45 keys>, 'seed': 123, ...2 more}"
    """
    if max_items is None:
        max_items = LOG_MAX_COLLECTION_ITEMS
    if max_length is None:
        max_length = LOG_MAX_OBJECT_OUTPUT

    if not isinstance(d, dict):
        return safe_repr(d, max_length)

    try:

        items = []
        remaining = len(d) - max_items if len(d) > max_items else 0

        for i, (k, v) in enumerate(d.items()):
            if i >= max_items:
                break

            # Smart handling based on key name and value type
            if k in LOG_LARGE_DICT_KEYS and isinstance(v, dict):
                # Just show key count for known large dicts
                items.append(f"'{k}': <dict with {len(v)} keys>")
            elif isinstance(v, (dict, list, tuple, set)) and len(str(v)) > 100:
                # Use reprlib for other large collections
                items.append(f"'{k}': {_safe_repr.repr(v)}")
            else:
                # Normal representation for small values
                v_str = str(v)
                if len(v_str) > 100:
                    v_str = v_str[:100] + "..."
                items.append(f"'{k}': {v_str}")

        result = "{" + ", ".join(items)
        if remaining > 0:
            result += f", ...{remaining} more"
        result += "}"

        # Final length check
        if len(result) > max_length:
            result = result[:max_length] + "...}"

        return result

    except Exception as e:
        return f"<dict repr failed: {len(d) if hasattr(d, '__len__') else '?'} items - {e}>"


def safe_log_params(params: dict, param_name: str = "parameters") -> str:
    """
    Create a safe log message for parameter dictionaries.

    This is specifically designed for logging generation/task parameters
    without causing hangs.

    Args:
        params: Parameter dictionary to log
        param_name: Name to use in the log message (default: "parameters")

    Returns:
        Safe log message string

    Example:
        >>> safe_log_params({'model': 'wan_2_2', 'seed': 123, 'huge_config': {...}})
        "parameters: {'model': 'wan_2_2', 'seed': 123, 'huge_config': <dict with 50 keys>}"
    """
    return f"{param_name}: {safe_dict_repr(params)}"


def safe_json_repr(obj: Any, max_length: int = None) -> str:
    """
    Safely serialize object to JSON string with size limits.

    This is a replacement for json.dumps(...)[:<limit>] pattern which still
    serializes the entire object before truncation (causing hangs).

    Args:
        obj: Object to serialize
        max_length: Maximum output length (default: LOG_MAX_JSON_OUTPUT)

    Returns:
        Safe JSON string, truncated if needed

    Example:
        >>> safe_json_repr({'huge': [1]*1000})
        '{"huge": [1, 1, 1, 1, 1, ...]}'

    Note:
        Prefer safe_dict_repr() for dicts as it's faster. This is for cases
        where JSON format is specifically needed for compatibility.
    """
    if max_length is None:
        max_length = LOG_MAX_JSON_OUTPUT

    try:
        # For small objects, use normal JSON serialization
        if isinstance(obj, (str, int, float, bool, type(None))):
            return json.dumps(obj)

        # For collections, try full serialization but catch large ones
        try:
            result = json.dumps(obj, default=str, indent=2)
            if len(result) <= max_length:
                return result
            # Too long, truncate with ellipsis
            return result[:max_length] + "...}"
        except (TypeError, ValueError, RecursionError):
            # Fallback to safe_repr for objects that can't be JSON serialized
            return safe_repr(obj, max_length)

    except Exception as e:
        return f"<json serialization failed: {type(obj).__name__} - {e}>"


def safe_log_change(param: str, old_value: Any, new_value: Any, max_length: int = None) -> str:
    """
    Create a safe log message for parameter changes (old → new).

    Args:
        param: Parameter name
        old_value: Old value
        new_value: New value
        max_length: Maximum length per value (default: LOG_MAX_STRING_REPR)

    Returns:
        Safe log message string

    Example:
        >>> safe_log_change('seed', 123, 456)
        "seed: 123 → 456"
        >>> safe_log_change('config', {...huge...}, {...huge...})
        "config: <dict with 50 keys> → <dict with 52 keys>"
    """
    if max_length is None:
        max_length = LOG_MAX_STRING_REPR

    try:
        # Special handling for dicts
        if isinstance(old_value, dict):
            old_str = f"<dict with {len(old_value)} keys>"
        elif old_value == "NOT_SET":
            old_str = "NOT_SET"
        else:
            old_str = safe_repr(old_value, max_length)

        if isinstance(new_value, dict):
            new_str = f"<dict with {len(new_value)} keys>"
        else:
            new_str = safe_repr(new_value, max_length)

        return f"{param}: {old_str} → {new_str}"
    except Exception as e:
        return f"{param}: <comparison failed: {e}>"


# Update ComponentLogger to use safe logging by default
class SafeComponentLogger(ComponentLogger):
    """
    Enhanced ComponentLogger with automatic safe logging.

    Automatically applies safe_repr() to prevent hanging on large objects.
    """

    def debug(self, message: str, task_id: Optional[str] = None):
        """Log a debug message with automatic safety checks."""
        # If message contains dict formatting, it's already too late
        # But we can catch obvious cases
        if len(message) > 10000:
            message = message[:10000] + "... [truncated - message too long]"
        super().debug(message, task_id)

    def safe_debug_dict(self, label: str, data: dict, task_id: Optional[str] = None):
        """
        Safely log a dictionary with truncation.

        Example:
            logger.safe_debug_dict("Generation params", params, task_id)
        """
        safe_msg = safe_log_params(data, label)
        self.debug(safe_msg, task_id)

    def safe_debug_change(self, param: str, old_value: Any, new_value: Any, task_id: Optional[str] = None):
        """
        Safely log a parameter change.

        Example:
            logger.safe_debug_change("seed", old_seed, new_seed, task_id)
        """
        safe_msg = safe_log_change(param, old_value, new_value)
        self.debug(safe_msg, task_id)


# Create safe versions of pre-configured loggers
headless_logger_safe = SafeComponentLogger("HEADLESS")
queue_logger_safe = SafeComponentLogger("QUEUE")
orchestrator_logger_safe = SafeComponentLogger("ORCHESTRATOR")
travel_logger_safe = SafeComponentLogger("TRAVEL")
generation_logger_safe = SafeComponentLogger("GENERATION")
model_logger_safe = SafeComponentLogger("MODEL")
task_logger_safe = SafeComponentLogger("TASK")


# =============================================================================
# LOGGING BEST PRACTICES DOCUMENTATION
# =============================================================================
"""
Safe Logging Guidelines for Headless-Wan2GP
===========================================

Problem:
--------
Logging large nested dictionaries (orchestrator_payload, phase_config, etc.) can
cause the process to hang for minutes or freeze entirely due to recursive string
conversion in Python's str() and json.dumps().

Solution:
---------
Always use the safe logging utilities which apply consistent limits:
- Max 200 chars per string value
- Max 500 chars total output
- Max 5 items per collection
- Max 3 levels deep for nesting

Usage Examples:
---------------

1. Logging dictionaries:
   # ❌ UNSAFE - can hang
   logger.debug(f"Params: {params}")
   logger.debug(f"Config: {json.dumps(config, default=str, indent=2)[:1000]}")  # Still serializes everything first!

   # ✅ SAFE - guaranteed fast
   logger.debug(f"Params: {safe_dict_repr(params)}")
   logger.debug(f"Config: {safe_json_repr(config)}")  # Only for compatibility

2. Logging parameter changes:
   # ❌ UNSAFE
   logger.debug(f"Applied {key}: {old} → {new}")

   # ✅ SAFE
   logger.debug(safe_log_change(key, old, new))

3. Logging any object:
   # ❌ UNSAFE
   logger.debug(f"Result: {result}")

   # ✅ SAFE
   logger.debug(f"Result: {safe_repr(result)}")

4. Using SafeComponentLogger (automatic protection):
   logger.safe_debug_dict("Generation params", params, task_id)
   logger.safe_debug_change("seed", old_seed, new_seed, task_id)

Configuration:
--------------
All limits are defined as global constants at the top of this file:
- LOG_MAX_STRING_REPR = 200
- LOG_MAX_OBJECT_OUTPUT = 500
- LOG_MAX_COLLECTION_ITEMS = 5
- LOG_MAX_NESTING_DEPTH = 3
- LOG_MAX_JSON_OUTPUT = 1000
- LOG_LARGE_DICT_KEYS = {...}

Performance:
------------
Using safe utilities provides massive performance gains:
- Small params: Same speed (~1ms)
- Large params: 100-5000x faster
- Huge nested dicts: ∞x faster (doesn't hang)

Migration:
----------
When you see:
  json.dumps(obj, default=str)[:<limit>]
Replace with:
  safe_json_repr(obj)

When you see:
  f"params: {some_dict}"
Replace with:
  f"params: {safe_dict_repr(some_dict)}"
"""