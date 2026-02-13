"""
Logging configuration with audit trail support.
"""
import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_dir: Path | None = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days"
) -> None:
    """
    Configure loguru logger with console and file output.

    Args:
        log_dir: Directory for log files (default: project_root/logs)
        level: Minimum log level
        rotation: When to rotate log files
        retention: How long to keep old logs
    """
    # Remove default handler
    logger.remove()

    # Console handler with colors
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        colorize=True
    )

    # File handler for all logs
    if log_dir is None:
        log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    logger.add(
        log_dir / "stock_predict.log",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=rotation,
        retention=retention,
        compression="zip"
    )

    # Separate file for audit trail (data operations)
    logger.add(
        log_dir / "audit.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        filter=lambda record: record["extra"].get("audit", False),
        rotation=rotation,
        retention="30 days"
    )


def get_logger(name: str = "stock_predict"):
    """Get a logger instance with the given name."""
    return logger.bind(name=name)


def audit_log(message: str, **kwargs) -> None:
    """
    Log an audit message for data operations.

    Args:
        message: Audit message
        **kwargs: Additional context to include
    """
    context = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    full_message = f"{message} | {context}" if context else message
    logger.bind(audit=True).info(full_message)
