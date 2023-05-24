import unittest
import torch.multiprocessing as mp
import os
from pathlib import Path
import logging

from oslo.torch.utils.logging import DistributedLogger

class TestDistributedLogger(unittest.TestCase):
    def setUp(self):
        self.log_file_path = Path("./logs")
        self.log_file_path.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.log_file_path.exists():
            for file in self.log_file_path.glob('*.log'):
                os.remove(file)

    def test_instance_creation(self):
        logger = DistributedLogger.get_instance("TestLogger")
        self.assertIsNotNone(logger)

    def test_set_level(self):
        logger = DistributedLogger.get_instance("TestLogger")
        logger.set_level('ERROR')
        self.assertEqual(logger._logger.level, logging.ERROR)

    def test_log_to_file(self):
        logger = DistributedLogger.get_instance("TestLogger")
        logger.set_level('INFO')
        logger.log_to_file(self.log_file_path)
        logger.info("Test log message")
        self.assertTrue(self.log_file_path.exists())

    def worker_fn(self, rank):
        logger = DistributedLogger.get_instance(f"TestLogger{rank}")
        logger.set_level('INFO')
        logger.log_to_file(self.log_file_path / f'rank_{rank}.log')
        logger.info(f"Test log message from rank {rank}")
        self.assertTrue((self.log_file_path / f'rank_{rank}.log').exists())

    def test_multiprocess_logging(self):
        num_processes = 4
        mp.spawn(self.worker_fn, nprocs=num_processes)

if __name__ == '__main__':
    unittest.main()
