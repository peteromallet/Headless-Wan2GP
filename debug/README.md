# Debug Tool

One unified tool to investigate tasks, workers, and system health.

## Quick Reference

```bash
# Task investigation
debug.py task <task_id>              # Complete task analysis with logs
debug.py task <task_id> --logs-only  # Just the log timeline

# Worker investigation
debug.py worker <worker_id>                  # Full worker analysis
debug.py worker <worker_id> --check-logging  # Is worker.py running?
debug.py worker <worker_id> --startup        # View initialization logs
debug.py worker <worker_id> --logs-only      # Just the log timeline

# System overview
debug.py tasks                 # Recent task statistics
debug.py workers               # Recent worker status
debug.py health                # Overall system health
debug.py orchestrator          # Is orchestrator running?

# Configuration & Infrastructure
debug.py config                # View timing/scaling settings
debug.py config --explain      # With detailed explanations
debug.py runpod                # Find orphaned pods ($$$ leak!)
debug.py runpod --terminate    # Terminate orphaned pods

# Output formats
--json                         # JSON output for any command
```

## Common Scenarios

### "Why did this task fail?"
```bash
debug.py task <task_id>
# Shows: task state, error message, timing, worker assignment, full log timeline
```

### "Worker isn't doing anything"
```bash
debug.py worker <worker_id> --check-logging
# Verifies if worker.py actually started

debug.py worker <worker_id> --startup
# Shows installation/initialization process
```

### "System seems unhealthy"
```bash
debug.py health
# Quick overview: worker status, task queue, recent errors

debug.py orchestrator
# Is orchestrator running? Recent cycles?

debug.py workers
# List all recent workers with failure analysis
```

### "We're wasting money"
```bash
debug.py runpod
# Finds pods running in RunPod but not tracked in database
# Shows daily/monthly cost waste
```

### "What are my timeout settings?"
```bash
debug.py config --explain
# Shows all timing configuration with explanations
# Grace periods, idle timeouts, scale-up thresholds
```

## Architecture

```
debug/
├── README.md          # This file
├── __init__.py        # Package init
├── models.py          # Data models (Task, Worker, LogEntry)
├── client.py          # DebugClient - unified data access
├── formatters.py      # Output formatting (text/JSON)
└── commands/          # Command handlers
    ├── task.py        # Task investigation
    ├── worker.py      # Worker investigation
    ├── tasks.py       # Multi-task analysis
    ├── workers.py     # Multi-worker analysis
    ├── health.py      # System health
    ├── orchestrator.py # Orchestrator status
    ├── config.py      # Configuration display
    └── runpod.py      # RunPod sync
```

## Data Sources

All commands use **system_logs** as the primary data source, augmented with current state from `tasks` and `workers` tables. This "logs-first" approach gives you a complete timeline of what actually happened.

## Tips

- Use `--json` for scripting/automation
- Use `--hours N` to adjust time window (default: 24h)
- Worker diagnostics show pre-termination VRAM, running tasks, pod status
- RunPod sync can save serious money by finding orphaned pods
- Config command is invaluable for debugging timing issues









