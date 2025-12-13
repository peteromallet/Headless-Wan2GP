"""System health command."""

from debug.client import DebugClient
from debug.formatters import Formatter


def run(client: DebugClient, options: dict):
    """Handle 'debug.py health' command."""
    try:
        health = client.get_system_health()
        
        format_type = options.get('format', 'text')
        output = Formatter.format_health(health, format_type)
        print(output)
        
    except Exception as e:
        print(f"❌ Error checking system health: {e}")
        import traceback
        if options.get('debug'):
            traceback.print_exc()









