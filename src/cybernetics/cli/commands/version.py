"""Command for showing version information.

This module implements the 'cybernetics version' command.
"""

import click


@click.command()
def cli():
    """Show version information."""
    try:
        # Lazy import version only when needed
        from cybernetics._version import __version__

        click.echo(f"cybernetics {__version__}")
    except ImportError:
        click.echo("cybernetics (version unavailable)")
