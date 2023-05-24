import inspect
import logging
from pathlib import Path
from typing import List, Union, Optional

import torch.distributed as dist

class DistributedLogger:
    """This is a distributed event logger class essentially based on :class:`logging`.

    Args:
        name (str): The name of the logger.
    """

    __instances = dict()

    @staticmethod
    def get_instance(name: str):
        """Get the unique single logger instance based on name.

        Args:
            name (str): The name of the logger.

        Returns:
            DistributedLogger: A DistributedLogger object
        """
        if name in DistributedLogger.__instances:
            return DistributedLogger.__instances[name]
        else:
            logger = DistributedLogger(name=name)
            return logger

    def __init__(self, name, log_level: Union[int, str] = 'INFO'):
        if name in DistributedLogger.__instances:
            raise ValueError(
                'Logger with the same name has been created, you should use colossalai.logging.get_dist_logger')
        else:
            if isinstance(log_level, str):
                log_level = getattr(logging, log_level)
            formatter = logging.Formatter('oslo - %(name)s - %(levelname)s: %(message)s')
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)

            self._name = name
            self._logger = logging.getLogger(name)
            self._logger.setLevel(log_level)
            self._logger.addHandler(handler)
            self._logger.propagate = False

            DistributedLogger.__instances[name] = self

    @staticmethod
    def __get_call_info():
        # Retrieve the record of the third stack frame 
        caller_record = inspect.stack()[2]

        # Extract necessary details
        file_name = caller_record.filename
        line_number = caller_record.lineno
        function_name = caller_record.function

        return file_name, line_number, function_name

    @staticmethod
    def _check_valid_logging_level(level: str):
        assert level in ['INFO', 'DEBUG', 'WARNING', 'ERROR'], 'found invalid logging level'

    def set_level(self, level: str) -> None:
        """Set the logging level

        Args:
            level (str): Can only be INFO, DEBUG, WARNING and ERROR.
        """
        self._check_valid_logging_level(level)
        self._logger.setLevel(getattr(logging, level))

    def log_to_file(self, path: Union[str, Path], mode: str = 'a', level: str = 'INFO', suffix: str = None) -> None:
        """Save the logs to file

        Args:
            path (A string or pathlib.Path object): The file to save the log.
            mode (str): The mode to write log into the file.
            level (str): Can only be INFO, DEBUG, WARNING and ERROR.
            suffix (str): The suffix string of log's name.
        """
        assert isinstance(path, (str, Path)), \
            f'expected argument path to be type str or Path, but got {type(path)}'
        self._check_valid_logging_level(level)

        if isinstance(path, str):
            path = Path(path)

        # create log directory
        path.mkdir(parents=True, exist_ok=True)

        # set the default file name if path is a directory
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
            
        if suffix is not None:
            log_file_name = f'rank_{rank}_{suffix}.log'
        else:
            log_file_name = f'rank_{rank}.log'
        path = path.joinpath(log_file_name)

        # add file handler
        file_handler = logging.FileHandler(path, mode)
        file_handler.setLevel(getattr(logging, level))
        formatter = logging.Formatter('oslo - %(name)s - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

    def _log(self,
             level: str,
             message: str,
             ranks: List[int] = None) -> None:
        if ranks is None:
            getattr(self._logger, level)(message)
        else:
            local_rank = dist.get_rank()
            if local_rank in ranks:
                getattr(self._logger, level)(message)

    def info(self, message: str, ranks: List[int] = None) -> None:
        """Log an info message.

        Args:
            message (str): The message to be logged.
            ranks (List[int]): List of parallel ranks.
        """
        message_prefix = "{}:{} {}".format(*self.__get_call_info())
        self._log('info', message_prefix, ranks)
        self._log('info', message, ranks)

    def warning(self, message: str, ranks: List[int] = None) -> None:
        """Log a warning message.

        Args:
            message (str): The message to be logged.
            ranks (List[int]): List of parallel ranks.
        """
        message_prefix = "{}:{} {}".format(*self.__get_call_info())
        self._log('warning', message_prefix, ranks)
        self._log('warning', message, ranks)

    def debug(self, message: str, ranks: List[int] = None) -> None:
        """Log a debug message.

        Args:
            message (str): The message to be logged.
            ranks (List[int]): List of parallel ranks.
        """
        message_prefix = "{}:{} {}".format(*self.__get_call_info())
        self._log('debug', message_prefix, ranks)
        self._log('debug', message, ranks)

    def error(self, message: str, ranks: List[int] = None) -> None:
        """Log an error message.

        Args:
            message (str): The message to be logged.
            ranks (List[int]): List of parallel ranks.
        """
        message_prefix = "{}:{} {}".format(*self.__get_call_info())
        self._log('error', message_prefix, ranks)
        self._log('error', message, ranks)