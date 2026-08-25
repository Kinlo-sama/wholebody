import logging
import sys
from typing import Optional


class ANSIColorFormatter(logging.Formatter):
    """Colorized console log formatter."""

    COLORS = {
        logging.DEBUG: "\033[36m",    # Cyan
        logging.INFO: "\033[32m",     # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",    # Red
        logging.CRITICAL: "\033[35m", # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        time_str = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        prefix = f"{self.BOLD}[{time_str} WholeBody {record.levelname}]{self.RESET}"
        message = record.getMessage()
        return f"{color}{prefix} {message}{self.RESET}"


def get_logger(name: str = "wholebody", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Get or configure a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ANSIColorFormatter())
        logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                "[%(asctime)s WholeBody %(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    logger.propagate = False
    return logger
