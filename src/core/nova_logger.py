import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

logger = logging.getLogger("lumos")


def setup_logger():
    """Configures the root logger for the entire Lumos ecosystem."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / "lumos_system.log"

    handler = TimedRotatingFileHandler(
        filename=log_file,
        when="D",
        interval=1,
        backupCount=14,
        encoding="utf-8",
    )

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    root_logger.addHandler(logging.StreamHandler())

    logging.info("--- Lumos System Log Initialized ---")
