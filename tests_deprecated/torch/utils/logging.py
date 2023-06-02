import unittest
import torch.multiprocessing as mp
import os
from pathlib import Path
import logging

from oslo.torch.utils.logging import DistributedLogger


def worker_fn(rank, log_file_path):
    logger = DistributedLogger.get_instance(f"TestLogger{rank}")
    logger.set_level("INFO")
    logger.attach_file_handler(log_file_path / f"rank_{rank}.log")
    logger.info(f"Test log message from rank {rank}")
    assert (log_file_path / f"rank_{rank}.log").exists()


class TestDistributedLogger(unittest.TestCase):
    def setUp(self):
        self.log_file_path = Path("./logs")
        self.log_file_path.mkdir(parents=True, exist_ok=True)

    def test_instance_creation(self):
        logger = DistributedLogger.get_instance("TestLogger")
        self.assertIsNotNone(logger)

    def test_set_level(self):
        logger = DistributedLogger.get_instance("TestLogger")
        logger.set_level("ERROR")
        self.assertEqual(logger._logger.level, logging.ERROR)

    def test_attach_file_handler(self):
        logger = DistributedLogger.get_instance("TestLogger")
        logger.set_level("INFO")
        logger.attach_file_handler(self.log_file_path / "test.log")
        logger.info("Test log message")
        self.assertTrue((self.log_file_path / "test.log").exists())

    def test_multiprocess_logging(self):
        num_processes = 4
        mp.spawn(worker_fn, args=(self.log_file_path,), nprocs=num_processes)


if __name__ == "__main__":
    unittest.main()
