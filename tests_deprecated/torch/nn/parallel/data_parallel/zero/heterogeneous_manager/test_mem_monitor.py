import unittest
from unittest.mock import patch

from oslo.torch.nn.parallel.data_parallel.zero.memory_tracer.memory_monitor import (
    SyncCudaMemoryMonitor,
)


class TestSyncCudaMemoryMonitor(unittest.TestCase):
    @patch("torch.cuda.synchronize")
    @patch("torch.cuda.reset_peak_memory_stats")
    @patch("torch.cuda.max_memory_allocated", return_value=1024)
    def test_sync_cuda_memory_monitor_methods(
        self, mock_max_memory_allocated, mock_reset_peak_memory_stats, mock_synchronize
    ):
        # Create a SyncCudaMemoryMonitor instance
        sync_cuda_mem_monitor = SyncCudaMemoryMonitor()

        # Test the start method
        sync_cuda_mem_monitor.start()
        mock_synchronize.assert_called_once()
        mock_reset_peak_memory_stats.assert_called_once()

        # Test the finish method
        max_usage = sync_cuda_mem_monitor.finish()
        self.assertIsInstance(max_usage, int)
        self.assertEqual(max_usage, 1024)  # The mock max_memory_allocated returns 1024


if __name__ == "__main__":
    unittest.main()
