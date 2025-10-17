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
        self.lock = threading.Lock()
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
                self.shared_queue.put_nowait(log_entry)
            except:
                # Queue full or not available - not critical, guardian will catch up
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