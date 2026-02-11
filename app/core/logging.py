"""Structured logging configuration using ``structlog``.

In production, logs are emitted as JSON lines (machine-parsable for ELK /
Datadog / CloudWatch). In development, a human-friendly console renderer is
used instead. The format is controlled by the ``LOG_FORMAT`` env var.
"""

import logging
import sys
from typing import Literal

import structlog


def setup_logging(log_level: str = "INFO", log_format: Literal["json", "console"] = "json") -> None:
    """Configure ``structlog`` and the stdlib root logger.

    Args:
        log_level: Minimum severity (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: ``"json"`` for machine-readable output, ``"console"`` for
            coloured, human-friendly output.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level.upper())

    # Quieten noisy third-party loggers
    for noisy in ("uvicorn.access", "ultralytics"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named, bound logger instance.

    Args:
        name: Logger name — typically ``__name__`` of the calling module.
    """
    return structlog.get_logger(name)
