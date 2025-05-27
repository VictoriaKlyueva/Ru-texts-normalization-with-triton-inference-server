import logging
import sys
from pathlib import Path


def setup_logging():
    """
        Configure logger to write into file and console.
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("./data")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Formatter for logs
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with UTF-8 encoding
    log_file = log_dir / "log_file.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()
