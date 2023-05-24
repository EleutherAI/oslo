import inspect
import logging
from pathlib import Path
import torch.distributed as dist
from typing import List, Union, Optional


class DistributedLogger:
    """A distributed event logger class.

    Args:
        name (str): The name of the logger.
        level (str): The logging level. Default: 'INFO'.
    """

    __instances = dict()
    LOG_LEVELS = ["INFO", "DEBUG", "WARNING", "ERROR"]

    @staticmethod
    def get_instance(name: str):
        """Get the unique logger instance based on name."""
        if name not in DistributedLogger.__instances:
            DistributedLogger.__instances[name] = DistributedLogger(name=name)
        return DistributedLogger.__instances[name]

    def __init__(self, name: str, level: str = "INFO"):
        formatter = logging.Formatter("oslo - %(name)s - %(levelname)s: %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        self._name = name
        self._logger = logging.getLogger(name)
        self.set_level(level)
        self._logger.addHandler(handler)
        self._logger.propagate = False

    @staticmethod
    def __get_call_info():
        # Retrieve the record of the third stack frame
        caller_record = inspect.stack()[2]
        # Extract necessary details
        return caller_record.filename, caller_record.lineno, caller_record.function

    @staticmethod
    def _get_rank():
        return dist.get_rank() if dist.is_initialized() else 0

    def set_level(self, level: str) -> None:
        """Set the logging level."""
        assert level in self.LOG_LEVELS, f"Invalid logging level: {level}"
        self._logger.setLevel(getattr(logging, level))

    def log_to_file(
        self,
        path: Union[str, Path],
        mode: str = "a",
        level: str = "INFO",
        suffix: str = None,
    ) -> None:
        """Save the logs to file."""
        assert level in self.LOG_LEVELS, f"Invalid logging level: {level}"
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        log_file_name = (
            f"rank_{self._get_rank()}_{suffix}.log"
            if suffix
            else f"rank_{self._get_rank()}.log"
        )
        path = path / log_file_name

        file_handler = logging.FileHandler(path, mode)
        file_handler.setLevel(getattr(logging, level))
        formatter = logging.Formatter("oslo - %(name)s - %(levelname)s: %(message)s")
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

    def _log(self, level: str, message: str, ranks: List[int] = None) -> None:
        if ranks is None or self._get_rank() in ranks:
            message_prefix = f"{self.__get_call_info()}:"
            log_function = getattr(self._logger, level)
            log_function(message_prefix)
            log_function(message)

    def info(self, message: str, ranks: List[int] = None) -> None:
        """Log an info message."""
        self._log("info", message, ranks)

    def warning(self, message: str, ranks: List[int] = None) -> None:
        """Log a warning message."""
        self._log("warning", message, ranks)

    def debug(self, message: str, ranks: List[int] = None) -> None:
        """Log a debug message."""
        self._log("debug", message, ranks)

    def error(self, message: str, ranks: List[int] = None) -> None:
        """Log an error message."""
        self._log("error", message, ranks)
