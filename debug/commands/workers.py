"""Workers listing command."""

from debug.client import DebugClient
from debug.formatters import Formatter


def run(client: DebugClient, options: dict):
    """Handle 'debug.py workers' command."""
    try:
        hours = options.get('hours', 2)
        detailed = options.get('detailed', False)
        
        summary = client.get_workers_summary(hours=hours, detailed=detailed)
        
        format_type = options.get('format', 'text')
        output = Formatter.format_workers_summary(summary, format_type)
        print(output)
        
    except Exception as e:
        print(f"❌ Error analyzing workers: {e}")
        import traceback
        if options.get('debug'):
            traceback.print_exc()









