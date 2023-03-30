import unittest
from unittest.mock import MagicMock, patch

from oslo.torch.nn.parallel.data_parallel.zero.heterogeneous_manager.memory_tracer.chunk_memstats_collector import (
    ChunkMemStatsCollector,
)

class TestChunkMemStatsCollector(unittest.TestCase):
    def setUp(self):
        self.mock_chunk_manager = MagicMock()
        self.mock_chunk_manager.total_mem = {"cuda": 1000}
        self.mock_sync_cuda_memory_monitor = MagicMock()
        self.mock_memstats = MagicMock()

    def create_collector(self):
        collector= ChunkMemStatsCollector(self.mock_chunk_manager)
        collector._mem_monitor = self.mock_sync_cuda_memory_monitor
        collector._memstats = self.mock_memstats
        return collector

    def test_start_and_finish_collection(self):
        collector = self.create_collector()

        collector.start_collection()
        self.assertTrue(collector._start_flag)
        self.mock_sync_cuda_memory_monitor.start.assert_called_once()

        collector.finish_collection()
        self.assertFalse(collector._start_flag)
        self.mock_sync_cuda_memory_monitor.finish.assert_called_once()

    def test_record_and_sample_overall_data(self):
        collector = self.create_collector()

        collector.start_collection()
        collector.record_model_data_volume()
        self.mock_memstats.record_max_cuda_model_data.assert_called_with(1000)

        collector.sample_overall_data()
        self.mock_sync_cuda_memory_monitor.finish.assert_called_once()
        self.mock_memstats.record_max_cuda_overall_data.assert_called_once()

    def test_clear(self):
        collector = self.create_collector()

        collector.clear()
        self.assertFalse(collector._start_flag)
        self.assertEqual(collector._step_idx, 0)
        self.assertEqual(collector._step_total, 0)


if __name__ == "__main__":
    unittest.main()