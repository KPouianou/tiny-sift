# tiny_sift/tests/unit/test_reservoir_benchmarking.py

"""
Unit tests for Reservoir Sampling benchmarking hooks.
"""

import math
import unittest
import sys
import time  # For performance tracking tests
from collections import Counter

# Adjust import path based on your project structure
# Assuming tests are run from the project root or similar
from tiny_sift.algorithms.reservoir import ReservoirSampling, WeightedReservoirSampling


class TestReservoirBenchmarking(unittest.TestCase):
    """Test cases for benchmarking hooks in Reservoir Sampling."""

    def test_estimate_size_reservoir(self):
        """Test memory estimation for ReservoirSampling."""
        rs = ReservoirSampling(size=10)
        size_empty = rs.estimate_size()
        self.assertGreater(size_empty, 50, "Empty reservoir should have base size")

        rs.update("a")
        rs.update(123)
        size_partial = rs.estimate_size()
        self.assertGreater(size_partial, size_empty, "Size should increase with items")

        # Fill the reservoir
        for i in range(10):
            rs.update(f"item-{i}")
        size_full = rs.estimate_size()
        self.assertGreater(size_full, size_partial, "Size should increase when full")

        # Check with large items
        rs_large = ReservoirSampling(size=5)
        large_item = "x" * 1024
        rs_large.update(large_item)
        size_large_item = rs_large.estimate_size()
        self.assertGreater(
            size_large_item,
            size_empty + sys.getsizeof(large_item),
            "Size should reflect large item size",
        )

    def test_estimate_size_weighted(self):
        """Test memory estimation for WeightedReservoirSampling."""
        wrs = WeightedReservoirSampling(size=10)
        size_empty = wrs.estimate_size()
        self.assertGreater(
            size_empty, 50, "Empty weighted reservoir should have base size"
        )

        wrs.update("a", weight=1.0)
        wrs.update(123, weight=2.5)
        size_partial = wrs.estimate_size()
        self.assertGreater(size_partial, size_empty, "Size should increase with items")

        # Fill the reservoir
        for i in range(10):
            wrs.update(f"item-{i}", weight=float(i + 1))
        size_full = wrs.estimate_size()
        self.assertGreater(size_full, size_partial, "Size should increase when full")

        # Compare with unweighted version (should be larger due to weight/key storage)
        rs = ReservoirSampling(size=10)
        for i in range(12):  # Add same number of items
            rs.update(f"item-{i if i < 10 else i-10}" if i < 12 else "other")
        self.assertGreater(
            size_full,
            rs.estimate_size(),
            "Weighted reservoir should use more memory than unweighted",
        )

    def test_get_stats_reservoir(self):
        """Test get_stats for ReservoirSampling."""
        rs = ReservoirSampling(size=10)
        stats_empty = rs.get_stats()
        self.assertEqual(stats_empty["type"], "ReservoirSampling")
        self.assertEqual(stats_empty["sample_size"], 0)
        self.assertEqual(stats_empty["max_sample_size"], 10)
        self.assertEqual(stats_empty["sample_fullness_pct"], 0.0)
        self.assertNotIn("avg_update_time_ns", stats_empty)  # Perf tracking not enabled

        # Enable performance tracking
        rs.enable_performance_tracking()
        rs.update("a")
        rs.update(1)
        stats_partial = rs.get_stats()
        self.assertEqual(stats_partial["sample_size"], 2)
        self.assertEqual(stats_partial["sample_fullness_pct"], 20.0)
        self.assertIn("avg_update_time_ns", stats_partial)
        self.assertIn(
            "recent_update_times_ns", stats_partial
        )  # Specific key if enabled

        # Fill reservoir
        for i in range(10):
            rs.update(i)
        stats_full = rs.get_stats()
        self.assertEqual(stats_full["sample_size"], 10)
        self.assertEqual(stats_full["sample_fullness_pct"], 100.0)
        self.assertEqual(stats_full["items_processed"], 12)

        # Disable tracking
        rs.disable_performance_tracking()
        rs.update(100)
        stats_no_perf = rs.get_stats()

        self.assertIn("avg_update_time_ns", stats_no_perf)
        self.assertNotIn("recent_update_times_ns", stats_no_perf)
        self.assertNotIn("min_update_time_ns", stats_no_perf)
        self.assertNotIn("max_update_time_ns", stats_no_perf)

    def test_get_stats_weighted(self):
        """Test get_stats for WeightedReservoirSampling."""
        wrs = WeightedReservoirSampling(size=5)
        stats_empty = wrs.get_stats()
        self.assertEqual(stats_empty["type"], "WeightedReservoirSampling")
        self.assertEqual(stats_empty["sample_size"], 0)
        self.assertEqual(stats_empty["max_sample_size"], 5)
        self.assertEqual(stats_empty["sample_fullness_pct"], 0.0)
        self.assertEqual(stats_empty["total_stream_weight"], 0.0)
        self.assertNotIn("total_weight_in_sample", stats_empty)

        wrs.update("a", 1.0)
        wrs.update("b", 4.0)
        stats_partial = wrs.get_stats()
        self.assertEqual(stats_partial["sample_size"], 2)
        self.assertEqual(stats_partial["max_sample_size"], 5)
        self.assertEqual(stats_partial["sample_fullness_pct"], 40.0)
        self.assertEqual(stats_partial["items_processed"], 2)
        self.assertEqual(stats_partial["total_stream_weight"], 5.0)
        self.assertEqual(stats_partial["total_weight_in_sample"], 5.0)
        self.assertEqual(stats_partial["min_weight_in_sample"], 1.0)
        self.assertEqual(stats_partial["max_weight_in_sample"], 4.0)
        self.assertEqual(stats_partial["avg_weight_in_sample"], 2.5)
        self.assertIn("min_key_in_sample", stats_partial)
        self.assertIn("max_key_in_sample", stats_partial)

    def test_error_bounds_reservoir(self):
        """Test error_bounds for ReservoirSampling."""
        rs = ReservoirSampling(size=10)
        bounds_init = rs.error_bounds()
        self.assertEqual(
            bounds_init["sampling_type"], "Uniform Reservoir (Algorithm R)"
        )
        self.assertIn("equal probability", bounds_init["property"].lower())
        self.assertEqual(bounds_init["inclusion_probability"], "N/A (empty stream)")

        for i in range(20):
            rs.update(i)
        bounds_full = rs.error_bounds()
        self.assertIn("10/20", bounds_full["inclusion_probability"])
        self.assertIn("0.5000", bounds_full["inclusion_probability"])

    def test_error_bounds_weighted(self):
        """Test error_bounds for WeightedReservoirSampling."""
        wrs = WeightedReservoirSampling(size=10)
        bounds = wrs.error_bounds()
        self.assertEqual(bounds["sampling_type"], "Weighted Reservoir (A-Exp-Res)")
        self.assertIn("proportional to item weight", bounds["property"])

    def test_assess_representativeness_reservoir(self):
        """Test assess_representativeness for ReservoirSampling."""
        rs = ReservoirSampling(size=10)
        rep_empty = rs.assess_representativeness()
        self.assertEqual(rep_empty["assessment_type"], "Sample Internal Statistics")
        self.assertEqual(rep_empty["sample_size"], 0)
        self.assertEqual(rep_empty["message"], "Sample is empty.")

        # Add numeric data
        for i in range(15):
            rs.update(float(i * 10))
        rep_numeric = rs.assess_representativeness()
        self.assertEqual(rep_numeric["sample_size"], 10)
        self.assertEqual(rep_numeric["data_type_assessed"], "numeric")
        self.assertIn("sample_min", rep_numeric)
        self.assertIn("sample_max", rep_numeric)
        self.assertIn("sample_mean", rep_numeric)
        self.assertIn("sample_stdev", rep_numeric)
        # Basic check on values (sample will contain numbers >= 0)
        self.assertGreaterEqual(rep_numeric["sample_min"], 0)
        self.assertLessEqual(rep_numeric["sample_max"], 140)

        # Add non-numeric hashable data
        rs_str = ReservoirSampling(size=5)
        rs_str.update("apple")
        rs_str.update("banana")
        rs_str.update("apple")  # Duplicate
        rep_str = rs_str.assess_representativeness()
        self.assertEqual(rep_str["sample_size"], 3)
        self.assertEqual(rep_str["data_type_assessed"], "hashable")
        self.assertEqual(rep_str["unique_items_in_sample"], 2)
        self.assertAlmostEqual(rep_str["sample_diversity_ratio"], 2 / 3)

    def test_assess_representativeness_weighted(self):
        """Test assess_representativeness for WeightedReservoirSampling."""
        wrs = WeightedReservoirSampling(size=5)
        rep_empty = wrs.assess_representativeness()
        self.assertEqual(
            rep_empty["assessment_type"], "Weighted Sample Internal Statistics"
        )
        self.assertEqual(rep_empty["sample_size"], 0)
        self.assertEqual(rep_empty["message"], "Sample is empty.")

        # Add numeric data with weights
        wrs.update(10.0, weight=1.0)
        wrs.update(50.0, weight=10.0)
        wrs.update(30.0, weight=5.0)
        rep_numeric = wrs.assess_representativeness()

        self.assertEqual(rep_numeric["sample_size"], 3)
        self.assertEqual(rep_numeric["data_type_assessed"], "numeric")
        # Check weight stats
        self.assertEqual(rep_numeric["total_weight_in_sample"], 16.0)
        self.assertEqual(rep_numeric["min_weight_in_sample"], 1.0)
        self.assertEqual(rep_numeric["max_weight_in_sample"], 10.0)
        self.assertAlmostEqual(rep_numeric["avg_weight_in_sample"], 16.0 / 3.0)
        self.assertIn("stdev_weight_in_sample", rep_numeric)
        # Check fraction comparisons
        self.assertAlmostEqual(
            rep_numeric["sample_weight_fraction"], 16.0 / 16.0
        )  # Total stream weight is 16
        self.assertAlmostEqual(
            rep_numeric["sample_count_fraction"], 3.0 / 3.0
        )  # 3 items processed
        # Check value stats
        self.assertEqual(rep_numeric["sample_min_value"], 10.0)
        self.assertEqual(rep_numeric["sample_max_value"], 50.0)
        self.assertAlmostEqual(rep_numeric["sample_mean_value"], (10 + 50 + 30) / 3.0)
        self.assertAlmostEqual(
            rep_numeric["sample_weighted_mean_value"],
            (10 * 1 + 50 * 10 + 30 * 5) / 16.0,
        )

        # Add non-numeric data
        wrs_str = WeightedReservoirSampling(size=3)
        wrs_str.update("low", weight=1)
        wrs_str.update("high", weight=100)
        rep_str = wrs_str.assess_representativeness()
        self.assertEqual(rep_str["sample_size"], 2)
        self.assertEqual(rep_str["data_type_assessed"], "hashable")
        self.assertEqual(rep_str["unique_items_in_sample"], 2)


if __name__ == "__main__":
    unittest.main()
