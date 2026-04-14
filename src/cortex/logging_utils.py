import logging

from logging.handlers import RotatingFileHandler

from .config import config

_configured = False


def get_logger(name: str = "cortex") -> logging.Logger:
    """
    Return an app logger configured to write to a local file.

    Console output is intentionally minimal (Rich handles user-facing messages).
    """

    global _configured

    logger = logging.getLogger(name)

    if _configured:
        return logger

    level = getattr(logging, str(config.log_level).upper(), logging.INFO)
    logger.setLevel(level)

    # Don't duplicate messages via root handlers.
    logger.propagate = False

    # Ensure parent directory exists (e.g. logs/).
    config.log_file.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        filename=str(config.log_file),
        maxBytes=int(config.log_max_bytes),
        backupCount=int(config.log_backups),
        encoding="utf-8",
    )

    handler.setLevel(level)

    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.addHandler(handler)
    _configured = True

    return logger
