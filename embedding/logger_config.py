"""Logging configuration for the evaluation framework."""
import sys
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

console = Console()

def setup_logger():
    """Configure loguru logger with rich formatting."""
    # Remove default handler
    logger.remove()
    
    # Add custom rich handler
    format_str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    logger.add(
        sys.stdout,
        format=format_str,
        level="INFO",
        colorize=True
    )
    
    return logger

def get_progress():
    """Create a rich progress bar."""
    return Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console
    )