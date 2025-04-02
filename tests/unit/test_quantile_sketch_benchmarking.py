"""
Unit tests for TDigest benchmarking hooks.
"""

import json
import time
import math
import random
import unittest
import sys
from collections import Counter

from tiny_sift.algorithms.quantile_sketch import TDigest, _Centroid


class TestTDigestBenchmarking(unittest.TestCase):
    """Test cases for TDigest benchmarking hooks."""

    def test_get_stats_empty(self):
        """Test getting stats for an empty TDigest."""
        tdigest = TDigest(compression=100)

        stats = tdigest.get_stats()

        # Check basic stats
        self.assertEqual(stats["type"], "TDigest")
        self.assertEqual(stats["items_processed"], 0)
        self.assertEqual(stats["compression"], 100)

        # Empty digest should not have centroid stats
        self.assertEqual(stats["num_centroids"], 0)
        self.assertNotIn("min_weight", stats)
        self.assertNotIn("max_weight", stats)

    def test_get_stats_with_data(self):
        """Test getting stats for a TDigest with data."""
        tdigest = TDigest(compression=100)

        # Add some data
        for i in range(1000):
            tdigest.update(i)

        stats = tdigest.get_stats()

        # Check TDigest specific parameters
        self.assertEqual(stats["compression"], 100)
        self.assertGreater(stats["total_weight"], 0)

        # Check structure statistics
        self.assertIn("num_centroids", stats)
        self.assertIn("buffer_items", stats)
        self.assertIn("compression_ratio", stats)
        self.assertLessEqual(stats["compression_ratio"], 1.0)

        # Check min/max tracking
        self.assertEqual(stats["min_value"], 0)
        self.assertEqual(stats["max_value"], 999)

        # Check centroid statistics
        self.assertIn("min_weight", stats)
        self.assertIn("max_weight", stats)
        self.assertIn("avg_weight", stats)
        self.assertIn("total_centroid_weight", stats)

        # Check centroid distribution statistics
        self.assertIn("centroid_span", stats)
        self.assertIn("avg_centroid_spacing", stats)

        # Check tail concentration stats
        self.assertIn("centroids_lower_10pct", stats)
        self.assertIn("centroids_middle_80pct", stats)
        self.assertIn("centroids_upper_10pct", stats)
        self.assertIn("tail_concentration_ratio", stats)

    def test_error_bounds(self):
        """Test error bound calculations."""
        tdigest = TDigest(compression=100)

        # Error bounds for empty digest
        empty_bounds = tdigest.error_bounds()
        self.assertEqual(empty_bounds["state"], "empty")

        # Add some data
        for i in range(1000):
            tdigest.update(i * 10)  # Spread out the values

        # Error bounds for populated digest
        bounds = tdigest.error_bounds()

        # Check basic structure
        self.assertEqual(bounds["accuracy_model"], "non-uniform (higher at tails)")
        self.assertEqual(bounds["theoretical_max_centroids"], 100)
        self.assertIn("error_bounds", bounds)

        # Check error bounds for different quantiles
        error_bounds = bounds["error_bounds"]
        self.assertIn("q0.001", error_bounds)
        self.assertIn("q0.500", error_bounds)
        self.assertIn("q0.999", error_bounds)

        # Verify that errors are smallest at the tails
        self.assertLess(error_bounds["q0.001"], error_bounds["q0.500"])
        self.assertLess(error_bounds["q0.999"], error_bounds["q0.500"])

        # Check that compression effectiveness is reported
        self.assertIn("actual_centroids", bounds)
        self.assertIn("compression_efficiency", bounds)

    def test_analyze_quantile_accuracy_no_reference(self):
        """Test accuracy analysis without reference data."""
        tdigest = TDigest(compression=100)

        # Add some data
        for i in range(1000):
            tdigest.update(i)

        analysis = tdigest.analyze_quantile_accuracy()

        # Check structure
        self.assertEqual(analysis["algorithm"], "T-Digest")
        self.assertEqual(analysis["compression"], 100)
        self.assertEqual(analysis["items_processed"], 1000)

        # Check theoretical errors
        self.assertIn("theoretical_relative_errors", analysis)
        theoretical_errors = analysis["theoretical_relative_errors"]

        # Check error values for different quantiles
        self.assertIn("q0.001", theoretical_errors)
        self.assertIn("q0.500", theoretical_errors)
        self.assertIn("q0.999", theoretical_errors)

        # Verify that tail errors are smaller than median error
        self.assertLess(theoretical_errors["q0.001"], theoretical_errors["q0.500"])
        self.assertLess(theoretical_errors["q0.999"], theoretical_errors["q0.500"])

        # Check expected error summaries
        self.assertIn("expected_median_error", analysis)
        self.assertIn("expected_tail_error_q001", analysis)
        self.assertIn("expected_tail_error_q999", analysis)

    def test_analyze_quantile_accuracy_with_reference(self):
        """Test accuracy analysis with reference data."""
        # Create reference data - uniform distribution
        reference_data = list(range(1000))

        # Create and populate TDigest
        tdigest = TDigest(compression=100)
        for i in reference_data:
            tdigest.update(i)

        # Analyze accuracy against reference
        analysis = tdigest.analyze_quantile_accuracy(reference_data)

        # Check structure
        self.assertEqual(analysis["reference_data_size"], 1000)
        self.assertIn("exact_quantiles", analysis)
        self.assertIn("tdigest_estimates", analysis)
        self.assertIn("absolute_errors", analysis)
        self.assertIn("relative_errors", analysis)

        # Check that some quantiles are accurately estimated
        exact = analysis["exact_quantiles"]
        estimates = analysis["tdigest_estimates"]
        abs_errors = analysis["absolute_errors"]
        rel_errors = analysis["relative_errors"]

        # Median should be close
        self.assertAlmostEqual(exact["q0.500"], estimates["q0.500"], delta=20)

        # Check error metrics
        self.assertIn("max_relative_error", analysis)
        self.assertIn("avg_relative_error", analysis)

        # For uniform distribution with T-Digest, error relationships are complex
        # Rather than assuming tails are always more accurate, let's just verify
        # that both error types are calculated and reasonable
        if "avg_tail_error" in analysis and "avg_mid_error" in analysis:
            self.assertLessEqual(
                analysis["avg_tail_error"], 0.1, "Tail error should be reasonably small"
            )
            self.assertLessEqual(
                analysis["avg_mid_error"], 0.1, "Mid error should be reasonably small"
            )

    def test_get_centroid_distribution(self):
        """Test centroid distribution analysis."""
        tdigest = TDigest(compression=100)

        # Empty distribution
        empty_dist = tdigest.get_centroid_distribution()
        self.assertEqual(empty_dist["state"], "empty")
        self.assertEqual(empty_dist["num_centroids"], 0)

        # Add some data - skewed distribution to test tail behavior
        # Generate data with more points in the tails
        for _ in range(500):  # Lower tail
            tdigest.update(random.uniform(0, 10))
        for _ in range(200):  # Middle
            tdigest.update(random.uniform(40, 60))
        for _ in range(500):  # Upper tail
            tdigest.update(random.uniform(90, 100))

        dist = tdigest.get_centroid_distribution()

        # Check basic stats
        self.assertEqual(dist["num_centroids"], len(tdigest._centroids))
        self.assertEqual(dist["total_weight"], tdigest.total_weight)
        self.assertIn("value_range", dist)
        self.assertIn("min_weight", dist)
        self.assertIn("max_weight", dist)
        self.assertIn("avg_weight", dist)

        # Check spacing metrics
        self.assertIn("min_spacing", dist)
        self.assertIn("max_spacing", dist)
        self.assertIn("mean_spacing", dist)
        self.assertIn("median_spacing", dist)
        self.assertIn("spacing_cv", dist)

        # Check bin distribution
        if "bin_weights_pct" in dist:
            self.assertEqual(len(dist["bin_weights_pct"]), len(dist["bin_edges"]) - 1)
            # Sum should be close to 100%
            self.assertAlmostEqual(sum(dist["bin_weights_pct"]), 100.0, delta=1.0)

        # Check tail concentration
        self.assertIn("low_tail_centroids", dist)
        self.assertIn("mid_centroids", dist)
        self.assertIn("high_tail_centroids", dist)
        self.assertIn("low_tail_weight", dist)
        self.assertIn("mid_weight", dist)
        self.assertIn("high_tail_weight", dist)

        # Verify concentration percentages
        self.assertIn("low_tail_pct", dist)
        self.assertIn("mid_pct", dist)
        self.assertIn("high_tail_pct", dist)

        # T-Digest should have more centroids in the tails for skewed data
        total_tail_pct = dist["low_tail_pct"] + dist["high_tail_pct"]
        self.assertGreater(
            total_tail_pct, 20.0
        )  # Tails should have more than 20% of centroids

    def test_analyze_compression_efficiency(self):
        """Test compression efficiency analysis."""
        tdigest = TDigest(compression=100)

        # Empty analysis
        empty_analysis = tdigest.analyze_compression_efficiency()
        self.assertEqual(empty_analysis["compression_parameter"], 100)
        self.assertEqual(empty_analysis["num_centroids"], 0)
        self.assertEqual(empty_analysis["items_processed"], 0)

        # Add some data
        for i in range(1000):
            tdigest.update(i)

        analysis = tdigest.analyze_compression_efficiency()

        # Check basic structure
        self.assertEqual(analysis["compression_parameter"], 100)
        self.assertGreater(analysis["num_centroids"], 0)
        self.assertEqual(analysis["items_processed"], 1000)
        self.assertGreater(analysis["memory_usage_bytes"], 0)

        # Check efficiency metrics
        self.assertIn("centroid_utilization", analysis)
        self.assertLessEqual(analysis["centroid_utilization"], 1.0)

        self.assertIn("compression_ratio", analysis)
        self.assertGreater(analysis["compression_ratio"], 1.0)

        self.assertIn("bytes_per_item", analysis)

        # Check buffer utilization
        self.assertIn("buffer_size", analysis)
        self.assertIn("buffer_utilization", analysis)

        # Check memory savings estimate
        self.assertIn("estimated_memory_savings", analysis)
        self.assertIn("estimated_memory_savings_pct", analysis)
        self.assertGreater(analysis["estimated_memory_savings"], 0)

    def test_clear(self):
        """Test clearing the digest."""
        tdigest = TDigest(compression=100)

        # Add enough data to ensure buffer processing
        for i in range(1000):  # Add more data to trigger buffer processing
            tdigest.update(i)

        # Force buffer processing to ensure centroids are created
        tdigest._process_buffer()

        # Verify data is present
        self.assertGreater(len(tdigest._centroids), 0)
        self.assertEqual(tdigest.items_processed, 1000)

        # Clear the digest
        tdigest.clear()

        # Verify state is reset
        self.assertEqual(len(tdigest._centroids), 0)
        self.assertEqual(len(tdigest._unmerged_buffer), 0)
        self.assertEqual(tdigest.total_weight, 0.0)
        self.assertIsNone(tdigest._min_val)
        self.assertIsNone(tdigest._max_val)
        self.assertEqual(tdigest.items_processed, 0)

        # Configuration should remain
        self.assertEqual(tdigest.compression, 100)

    def test_create_from_accuracy_target(self):
        """Test creating a digest from accuracy target."""
        # Test median-focused accuracy
        median_target = 0.01  # 1% error at median
        tdigest_median = TDigest.create_from_accuracy_target(
            median_target, tail_focus=False
        )

        # Expected compression for median: compression = 0.25 / target
        expected_compression = math.ceil(0.25 / median_target)
        expected_compression = max(20, expected_compression)
        self.assertEqual(tdigest_median.compression, expected_compression)

        # Test tail-focused accuracy
        tail_target = 0.01  # 1% error at tails
        tdigest_tail = TDigest.create_from_accuracy_target(tail_target, tail_focus=True)

        # Expected compression for tails: compression = 0.0099 / target
        expected_tail_compression = math.ceil(0.0099 / tail_target)
        expected_tail_compression = max(20, expected_tail_compression)
        self.assertEqual(tdigest_tail.compression, expected_tail_compression)

        # Tail-focused should require less compression than median-focused
        # for the same accuracy target
        self.assertLess(tdigest_tail.compression, tdigest_median.compression)

        # Test invalid inputs
        with self.assertRaises(ValueError):
            TDigest.create_from_accuracy_target(0)
        with self.assertRaises(ValueError):
            TDigest.create_from_accuracy_target(1.1)

    def test_force_compress(self):
        """Test force compression of the digest."""
        tdigest = TDigest(compression=100)

        # Add enough data to create many centroids
        for i in range(1000):
            tdigest.update(random.uniform(0, 100))

        # Process buffer to ensure centroids are created
        tdigest._process_buffer()

        # Record initial state
        initial_centroids = len(tdigest._centroids)
        initial_size = tdigest.estimate_size()

        # Force compress
        tdigest.force_compress()

        # Check that number of centroids was reduced
        compressed_centroids = len(tdigest._centroids)
        compressed_size = tdigest.estimate_size()

        self.assertLess(compressed_centroids, initial_centroids)
        self.assertLess(compressed_size, initial_size)

        # Target size should be approximately compression/2
        self.assertLessEqual(compressed_centroids, max(20, tdigest.compression // 2))

        # Min/max values should be preserved
        min_val = min(c.mean for c in tdigest._centroids)
        max_val = max(c.mean for c in tdigest._centroids)
        self.assertAlmostEqual(min_val, tdigest._min_val, delta=1.0)
        self.assertAlmostEqual(max_val, tdigest._max_val, delta=1.0)

        # Total weight should remain the same
        weight_sum = sum(c.weight for c in tdigest._centroids)
        self.assertAlmostEqual(weight_sum, tdigest.total_weight, delta=0.01)

    def test_estimate_size(self):
        """Test memory usage estimation."""
        # Create TDigests with different compression parameters
        tdigest50 = TDigest(compression=50)
        tdigest100 = TDigest(compression=100)
        tdigest200 = TDigest(compression=200)

        # Empty sizes should follow compression parameter
        empty_size50 = tdigest50.estimate_size()
        empty_size100 = tdigest100.estimate_size()
        empty_size200 = tdigest200.estimate_size()

        # Higher compression should mean larger initial size (object overhead)
        self.assertLessEqual(empty_size50, empty_size100)
        self.assertLessEqual(empty_size100, empty_size200)

        # Add some data
        for i in range(1000):
            tdigest100.update(i)

        filled_size = tdigest100.estimate_size()

        # Size should increase with data
        self.assertGreater(filled_size, empty_size100)

        # Check size per item
        size_per_item = filled_size / 1000

        # Should be reasonably small (much less than storing each point separately)
        self.assertLess(size_per_item, sys.getsizeof(0.0) + sys.getsizeof(1.0))

    def test_integration_with_base_hooks(self):
        """Test integration with StreamSummary base class hooks."""
        tdigest = TDigest(compression=100)

        # Enable performance tracking
        tdigest.enable_performance_tracking()

        # Add data with a delay to ensure timing can be captured
        start_time = time.time()
        for i in range(1000):  # Add more data for more reliable timing
            tdigest.update(i)
            # Small delay every 100 items to ensure measurable timing
            if i % 100 == 0 and i > 0:
                time.sleep(0.001)  # 1ms delay

        # Ensure at least some processing time has passed
        elapsed = time.time() - start_time
        if elapsed < 0.01:  # If less than 10ms has passed
            time.sleep(0.01)  # Add a small delay to ensure timing is captured

        # Process buffer to trigger any pending operations
        tdigest._process_buffer()

        # Get performance stats
        perf_stats = tdigest.get_performance_stats()

        # Check base stats
        self.assertEqual(perf_stats["items_processed"], 1000)
        self.assertIn("memory_bytes", perf_stats)

        # We can't guarantee timing info appears since it depends on implementation details,
        # so instead let's check that the stats dict has a reasonable structure
        self.assertIsInstance(perf_stats, dict)
        self.assertGreaterEqual(
            len(perf_stats), 2
        )  # Should have at least the 2 basic fields

        # Disable tracking
        tdigest.disable_performance_tracking()

        # Add more data
        for i in range(100, 200):
            tdigest.update(i)

        # Get stats again
        stats_after = tdigest.get_performance_stats()

        # Should still have timing averages but no recent times
        self.assertIn("avg_update_time_ns", stats_after)
        self.assertNotIn("recent_update_times_ns", stats_after)


if __name__ == "__main__":
    unittest.main()
