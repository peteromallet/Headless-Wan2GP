"""Orchestrator status command."""

from debug.client import DebugClient
from debug.formatters import Formatter


def run(client: DebugClient, options: dict):
    """Handle 'debug.py orchestrator' command."""
    try:
        hours = options.get('hours', 1)
        
        status = client.get_orchestrator_status(hours=hours)
        
        format_type = options.get('format', 'text')
        output = Formatter.format_orchestrator(status, format_type)
        print(output)
        
    except Exception as e:
        print(f"❌ Error checking orchestrator status: {e}")
        import traceback
        if options.get('debug'):
            traceback.print_exc()









