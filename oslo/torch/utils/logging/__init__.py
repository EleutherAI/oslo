from .logger import DistributedLogger

__all__ = ["get_dist_logger", "DistributedLogger"]


def get_dist_logger(name: str = "oslo") -> DistributedLogger:
    """Get logger instance based on name. The DistributedLogger will create singleton instances,
    which means that only one logger instance is created per name.

    Args:
        name (str): name of the logger, name must be unique

    Returns:
        DistributedLogger: A DistributedLogger object
    """
    return DistributedLogger.get_instance(name=name)
