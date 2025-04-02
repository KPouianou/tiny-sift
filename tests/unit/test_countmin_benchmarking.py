"""
Unit tests for Count-Min Sketch benchmarking hooks.
"""

import json
import math
import random
import unittest
import sys
import time
from collections import Counter
import array

from tiny_sift.algorithms.countmin import CountMinSketch


class TestCountMinSketchBenchmarking(unittest.TestCase):
    """Test cases for benchmarking hooks in Count-Min Sketch."""

    def test_get_stats_empty(self):
        """Test getting stats for an empty Count-Min Sketch."""
        cms = CountMinSketch(width=100, depth=5)

        stats = cms.get_stats()

        # Check base stats
        self.assertEqual(stats["type"], "CountMinSketch")
        self.assertEqual(stats["items_processed"], 0)
        self.assertEqual(stats["total_frequency"], 0)

        # Check Count-Min Sketch specific parameters
        self.assertEqual(stats["width"], 100)
        self.assertEqual(stats["depth"], 5)
        self.assertIn("epsilon", stats)
        self.assertIn("delta", stats)

        # Check counter statistics
        self.assertIn("total_counters", stats)
        self.assertEqual(stats["total_counters"], 100 * 5)
        self.assertEqual(stats["zero_counters"], 100 * 5)
        self.assertEqual(stats["non_zero_counters"], 0)
        self.assertEqual(stats["saturation"], 0.0)

        # Check memory breakdown
        self.assertIn("memory_breakdown", stats)
        self.assertGreater(stats["memory_breakdown"]["total"], 0)

    def test_get_stats_with_data(self):
        """Test getting stats for a Count-Min Sketch with data."""
        cms = CountMinSketch(width=50, depth=3)

        # Add some data
        for i in range(100):
            cms.update(f"item-{i % 20}")  # 20 unique items, some with higher frequency

        stats = cms.get_stats()

        # Check Count-Min Sketch specific parameters
        self.assertEqual(stats["width"], 50)
        self.assertEqual(stats["depth"], 3)
        self.assertEqual(stats["total_frequency"], 100)

        # Check counter statistics
        self.assertGreater(stats["non_zero_counters"], 0)
        self.assertLess(
            stats["non_zero_counters"], 50 * 3
        )  # Some but not all counters should be set
        self.assertGreater(stats["saturation"], 0.0)
        self.assertLess(stats["saturation"], 1.0)
        self.assertGreater(stats["max_counter"], 0)

        # Check error bounds
        self.assertIn("max_absolute_error", stats)
        self.assertIn("max_relative_error", stats)
        self.assertIn("observed_saturation", stats)
        self.assertIn("estimated_collision_rate", stats)

        # Check counter distribution
        self.assertIn("counter_distribution", stats)
        self.assertTrue(isinstance(stats["counter_distribution"], dict))

        # Check memory breakdown
        self.assertIn("memory_breakdown", stats)
        self.assertGreaterEqual(
            stats["memory_breakdown"]["counter_arrays"],
            stats["depth"] * sys.getsizeof(array.array("L", [0])),
        )

    def test_error_bounds(self):
        """Test error bound calculations."""
        width = 500
        depth = 5
        cms = CountMinSketch(width=width, depth=depth)

        # Empty sketch
        bounds = cms.error_bounds()

        # Check expected bounds
        self.assertAlmostEqual(bounds["epsilon"], math.e / width, places=10)
        self.assertAlmostEqual(bounds["delta"], math.exp(-depth), places=10)
        self.assertEqual(bounds["max_absolute_error"], 0.0)  # No items yet
        self.assertEqual(bounds["observed_saturation"], 0.0)  # No items yet

        # Add some data
        for i in range(1000):
            cms.update(f"item-{i % 100}")  # 100 unique items with frequency 10 each

        # Get updated bounds
        bounds_with_data = cms.error_bounds()

        # Check updated bounds
        self.assertAlmostEqual(bounds_with_data["epsilon"], math.e / width, places=10)
        self.assertAlmostEqual(bounds_with_data["delta"], math.exp(-depth), places=10)
        self.assertEqual(
            bounds_with_data["max_absolute_error"], (math.e / width) * 1000
        )
        self.assertGreater(bounds_with_data["observed_saturation"], 0.0)
        self.assertLessEqual(bounds_with_data["observed_saturation"], 1.0)
        self.assertGreater(bounds_with_data["estimated_collision_rate"], 0.0)

        # Test with different width/depth
        cms2 = CountMinSketch(width=1000, depth=7)
        bounds2 = cms2.error_bounds()

        # Increasing width should decrease epsilon
        self.assertLess(bounds2["epsilon"], bounds["epsilon"])

        # Increasing depth should decrease delta
        self.assertLess(bounds2["delta"], bounds["delta"])

    def test_estimate_size(self):
        """Test memory usage estimation."""
        # Create Count-Min Sketches with different dimensions
        cms_small = CountMinSketch(width=10, depth=2)
        cms_medium = CountMinSketch(width=100, depth=4)
        cms_large = CountMinSketch(width=1000, depth=5)

        # Sizes should increase with dimensions
        size_small = cms_small.estimate_size()
        size_medium = cms_medium.estimate_size()
        size_large = cms_large.estimate_size()

        self.assertLess(size_small, size_medium)
        self.assertLess(size_medium, size_large)

        # Adding items should not significantly change size (counter arrays are pre-allocated)
        for i in range(100):
            cms_medium.update(f"item-{i}")

        size_medium_after = cms_medium.estimate_size()

        # Allow for some small increase due to other tracking structures
        self.assertLessEqual(size_medium_after, size_medium * 1.1)

        # Verify size is related to counter dimensions
        # Each counter is 8 bytes (unsigned long)
        expected_counter_bytes_large = (
            1000 * 5 * 8
        )  # width * depth * sizeof(unsigned long)
        self.assertGreaterEqual(size_large, expected_counter_bytes_large)

    def test_enable_disable_hash_tracking(self):
        """Test enabling and disabling hash tracking."""
        cms = CountMinSketch(width=50, depth=3)

        # Initially hash tracking should be disabled
        self.assertFalse(cms._enable_hash_tracking)
        self.assertIsNone(cms._hash_position_counts)
        self.assertIsNone(cms._recent_items_for_tests)

        # Enable hash tracking
        cms.enable_hash_tracking()

        # Check tracking structures are initialized
        self.assertTrue(cms._enable_hash_tracking)
        self.assertIsNotNone(cms._hash_position_counts)
        self.assertIsNotNone(cms._recent_items_for_tests)

        # Add some data with tracking enabled
        for i in range(100):
            cms.update(f"item-{i % 20}")

        # Get stats with hash tracking data
        stats = cms.get_stats()

        # Should include hash quality metrics
        self.assertIn("hash_quality", stats)
        self.assertTrue(stats["hash_quality"]["tracking_enabled"])

        # Disable hash tracking
        cms.disable_hash_tracking()

        # Check tracking structures are cleared
        self.assertFalse(cms._enable_hash_tracking)
        self.assertIsNone(cms._hash_position_counts)
        self.assertIsNone(cms._recent_items_for_tests)

        # Get stats after disabling tracking
        stats_after = cms.get_stats()

        # Hash quality should not be in stats anymore
        self.assertNotIn("hash_quality", stats_after)

    def test_analyze_performance(self):
        """Test the analyze_performance method."""
        cms = CountMinSketch(width=50, depth=3)

        # Empty sketch analysis
        analysis = cms.analyze_performance()

        # Check analysis structure
        self.assertEqual(analysis["algorithm"], "Count-Min Sketch")
        self.assertIn("memory_efficiency", analysis)
        self.assertIn("accuracy", analysis)
        self.assertIn("saturation", analysis)
        self.assertIn("counter_statistics", analysis)
        self.assertIn("recommendations", analysis)

        # Add some data
        for i in range(1000):
            cms.update(f"item-{i % 20}")  # 20 unique items with high collision

        # Get updated analysis
        analysis_with_data = cms.analyze_performance()

        # Check updated analysis
        self.assertGreater(analysis_with_data["saturation"]["non_zero_counters"], 0)
        self.assertGreater(analysis_with_data["saturation"]["saturation_ratio"], 0.0)

        # With this setup (width=50, items=1000), we should get a saturation recommendation
        saturation_recommendation = False
        width_recommendation = False

        for rec in analysis_with_data["recommendations"]:
            if "saturation" in rec.lower():
                saturation_recommendation = True
            if "width" in rec.lower() and "small" in rec.lower():
                width_recommendation = True

        self.assertTrue(
            saturation_recommendation or width_recommendation,
            "Should recommend width increase or warn about saturation",
        )

        # Try with a better-sized sketch
        cms_better = CountMinSketch(width=5000, depth=3)
        for i in range(1000):
            cms_better.update(f"item-{i % 20}")

        analysis_better = cms_better.analyze_performance()

        # This sketch should have fewer recommendations
        self.assertLessEqual(
            len(analysis_better["recommendations"]),
            len(analysis_with_data["recommendations"]),
        )

    def test_estimate_frequency_error(self):
        """Test the estimate_frequency_error method."""
        cms = CountMinSketch(width=100, depth=5)

        # Add some test data
        item = "test-item"
        for _ in range(10):
            cms.update(item)

        # Add other items to create hash collisions
        for i in range(90):
            cms.update(f"other-{i}")

        # Get frequency estimate with error bounds
        freq, error = cms.estimate_frequency_error(item)

        # Frequency should be at least the true count
        self.assertGreaterEqual(freq, 10)

        # Error should be within theoretical bounds
        self.assertAlmostEqual(
            error, (math.e / 100) * 100, places=1
        )  # epsilon * total_frequency

        # Test item not in the sketch
        freq_none, error_none = cms.estimate_frequency_error("not-in-sketch")

        # Frequency should be >= 0, possibly > 0 due to collisions
        self.assertGreaterEqual(freq_none, 0)

        # Error bound should be the same (depends only on sketch parameters and total frequency)
        self.assertEqual(error_none, error)

    def test_heavy_hitters_from_candidates(self):
        """Test the get_heavy_hitters_from_candidates method with benchmarking."""
        cms = CountMinSketch(width=200, depth=5)

        # Create data with known frequency distribution
        # 5 items with high frequency, 95 with low frequency
        high_freq_items = [f"high-{i}" for i in range(5)]
        low_freq_items = [f"low-{i}" for i in range(95)]

        # Add high frequency items (each 100 times)
        for item in high_freq_items:
            for _ in range(100):
                cms.update(item)

        # Add low frequency items (each 10 times)
        for item in low_freq_items:
            for _ in range(10):
                cms.update(item)

        # Total frequency: (5*100) + (95*10) = 500 + 950 = 1450

        # Get heavy hitters with 5% threshold
        # Expected: only high frequency items (100/1450 = ~6.9%)
        candidates = high_freq_items + low_freq_items
        heavy_hitters = cms.get_heavy_hitters_from_candidates(candidates, 0.05)

        # Check results
        self.assertEqual(len(heavy_hitters), 5)
        for item in high_freq_items:
            self.assertIn(item, heavy_hitters)

        # Get stats after heavy hitter query
        stats = cms.get_stats()

        # Check that stats include relevant information
        self.assertEqual(stats["total_frequency"], 1450)
        self.assertGreaterEqual(
            stats["estimated_distinct_items"], 95
        )  # Should estimate approximately 100 distinct items

        # Analyze performance
        analysis = cms.analyze_performance()

        # Check that analysis includes accuracy information
        self.assertIn("accuracy", analysis)
        self.assertEqual(analysis["accuracy"]["width"], 200)
        self.assertEqual(analysis["accuracy"]["depth"], 5)

    def test_create_from_memory_limit(self):
        """Test creating a sketch from memory limit parameters."""
        # Create sketch with specific memory limit
        memory_bytes = 10000  # 10 KB

        cms = CountMinSketch.create_from_memory_limit(memory_bytes=memory_bytes)

        # Check that the memory usage is approximately within the limit
        # Allow a small buffer (5%) for differences in object overhead between systems
        self.assertLessEqual(cms.estimate_size(), memory_bytes * 1.05)

        # Check that we got reasonable width and depth
        self.assertGreater(cms._width, 0)
        self.assertGreater(cms._depth, 0)

        # Given the size and 8-byte counters, we should have close to memory_bytes/8 counters
        expected_counters = memory_bytes // 8
        actual_counters = cms._width * cms._depth

        # Allow some slack for object overhead
        self.assertGreaterEqual(expected_counters, actual_counters)
        self.assertGreaterEqual(actual_counters, expected_counters * 0.5)

        # Test with different epsilon_delta_ratio to prioritize width
        cms_width_priority = CountMinSketch.create_from_memory_limit(
            memory_bytes=memory_bytes,
            epsilon_delta_ratio=0.05,  # Lower ratio prioritizes width over depth
        )

        # Test with different epsilon_delta_ratio to prioritize depth
        cms_depth_priority = CountMinSketch.create_from_memory_limit(
            memory_bytes=memory_bytes,
            epsilon_delta_ratio=0.2,  # Higher ratio prioritizes depth over width
        )

        # Check that the ratios affected width/depth ratio as expected
        width_ratio = cms_width_priority._width / cms_width_priority._depth
        depth_ratio = cms_depth_priority._width / cms_depth_priority._depth

        self.assertGreater(width_ratio, depth_ratio)

        # Test with invalid memory limit
        with self.assertRaises(ValueError):
            CountMinSketch.create_from_memory_limit(memory_bytes=0)

    def test_integration_with_streaming(self):
        """Test that benchmarking hooks don't interfere with normal operation."""
        # Create two identical Count-Min Sketches
        cms1 = CountMinSketch(width=100, depth=3, seed=42)
        cms2 = CountMinSketch(width=100, depth=3, seed=42)

        # Enable hash tracking on the second one
        cms2.enable_hash_tracking()

        # Process the same stream with both sketches
        stream = [f"item-{i % 50}" for i in range(1000)]
        random.seed(42)
        random.shuffle(stream)

        # Process stream with cms1 normally
        for item in stream:
            cms1.update(item)

        # Process stream with cms2 while calling benchmarking hooks
        for i, item in enumerate(stream):
            cms2.update(item)

            # Call benchmarking hooks occasionally
            if i % 200 == 0:
                cms2.get_stats()
                cms2.error_bounds()
                cms2.analyze_performance()

        # Both should produce the same frequency estimates
        for i in range(50):
            item = f"item-{i}"
            self.assertEqual(
                cms1.estimate_frequency(item), cms2.estimate_frequency(item)
            )

        # Get heavy hitters from both
        candidates = [f"item-{i}" for i in range(50)]
        hh1 = cms1.get_heavy_hitters_from_candidates(candidates, 0.02)
        hh2 = cms2.get_heavy_hitters_from_candidates(candidates, 0.02)

        # Should identify the same heavy hitters
        self.assertEqual(set(hh1.keys()), set(hh2.keys()))

    def test_serialization_with_benchmarking(self):
        """Test that serialization works correctly with benchmarking hooks."""
        cms = CountMinSketch(width=50, depth=3, seed=123)

        # Enable hash tracking
        cms.enable_hash_tracking()

        # Add some data
        for i in range(500):
            cms.update(f"item-{i % 100}")

        # Get benchmarking information
        original_stats = cms.get_stats()
        original_analysis = cms.analyze_performance()

        # Serialize and deserialize
        serialized = cms.serialize(format="json")
        deserialized = CountMinSketch.deserialize(serialized, format="json")

        # Check that core parameters match
        self.assertEqual(deserialized._width, cms._width)
        self.assertEqual(deserialized._depth, cms._depth)
        self.assertEqual(deserialized._total_frequency, cms._total_frequency)

        # Check that frequency estimates match
        for i in range(100):
            item = f"item-{i}"
            self.assertEqual(
                deserialized.estimate_frequency(item), cms.estimate_frequency(item)
            )

        # Benchmarking state is not serialized, so tracking should be disabled
        self.assertFalse(deserialized._enable_hash_tracking)

        # Enable tracking on deserialized object
        deserialized.enable_hash_tracking()

        # It should work normally
        analysis = deserialized.analyze_performance()
        self.assertEqual(analysis["algorithm"], "Count-Min Sketch")
        self.assertIn("memory_efficiency", analysis)
        self.assertIn("accuracy", analysis)

    def test_hashfunction_quality_assessment(self):
        """Test the hash function quality assessment metrics."""
        # Create a sketch with hash tracking enabled
        cms = CountMinSketch(width=100, depth=5)
        cms.enable_hash_tracking()

        # Add sufficient data for hash quality analysis
        for i in range(200):
            cms.update(f"item-{i}")

        # Get stats with hash quality metrics
        stats = cms.get_stats()

        # Check hash quality metrics
        self.assertIn("hash_quality", stats)
        hash_quality = stats["hash_quality"]

        # Check tracking data
        self.assertTrue(hash_quality["tracking_enabled"])
        self.assertIn("positions_hit", hash_quality)
        self.assertIn("total_positions", hash_quality)
        self.assertIn("position_utilization", hash_quality)

        # Check distribution metrics
        self.assertIn("position_hit_cv", hash_quality)  # Coefficient of variation

        # Check row statistics
        self.assertIn("row_statistics", hash_quality)

        # If we have enough samples, we should have uniformity statistics
        if len(cms._recent_items_for_tests) >= 10:
            self.assertIn("uniformity_by_row", hash_quality)
            self.assertIn("uniformity_assessment", hash_quality)

            # Check uniformity metrics for each row
            for row_idx in range(cms._depth):
                row_key = f"row_{row_idx}"
                self.assertIn(row_key, hash_quality["uniformity_by_row"])

                row_metrics = hash_quality["uniformity_by_row"][row_key]
                self.assertIn("chi_square", row_metrics)
                self.assertIn("entropy", row_metrics)
                self.assertIn("normalized_entropy", row_metrics)

            # Overall assessment should include average entropy
            self.assertIn("entropy_avg", hash_quality["uniformity_assessment"])

            # Entropy should be > 0.5 for reasonable hash functions
            self.assertGreater(
                hash_quality["uniformity_assessment"]["entropy_avg"], 0.5
            )

        # Test with insufficient data
        cms_small = CountMinSketch(width=100, depth=5)
        cms_small.enable_hash_tracking()

        # Add just a few items
        for i in range(5):
            cms_small.update(f"item-{i}")

        # Get stats (should not have uniformity data)
        small_stats = cms_small.get_stats()

        # Basic tracking should still work
        self.assertIn("hash_quality", small_stats)
        small_hash_quality = small_stats["hash_quality"]
        self.assertTrue(small_hash_quality["tracking_enabled"])

        # But uniformity assessment may not be present with too little data
        if "uniformity_by_row" in small_hash_quality:
            self.assertLessEqual(
                len(small_hash_quality["uniformity_by_row"]), cms_small._depth
            )

    def test_counter_distribution_bins(self):
        """Test the counter distribution binning mechanism."""
        cms = CountMinSketch(width=100, depth=3)

        # Create different counter values
        values_to_test = [0, 1, 3, 8, 25, 75, 350, 750, 1500, 15000, 150000]

        # Try _get_distribution_bin method on each value
        bins = [cms._get_distribution_bin(val) for val in values_to_test]

        # Each value should map to a string bin
        for bin_label in bins:
            self.assertIsInstance(bin_label, str)

        # Specific checks for key ranges
        self.assertEqual(cms._get_distribution_bin(0), "0")
        self.assertEqual(cms._get_distribution_bin(1), "1")
        self.assertIn(cms._get_distribution_bin(3), ["2-5"])
        self.assertIn(cms._get_distribution_bin(8), ["6-10"])
        self.assertIn(cms._get_distribution_bin(25), ["11-50"])

        # Add data to the sketch so we can check the distribution in stats
        for i in range(1000):
            # Add values with different frequencies to create diverse counter values
            if i < 500:
                cms.update("common")
            elif i < 800:
                cms.update(f"medium-{i % 10}")
            else:
                cms.update(f"rare-{i}")

        # Get stats with counter distribution
        stats = cms.get_stats()

        # Check counter distribution in stats
        self.assertIn("counter_distribution", stats)

        # Distribution should have multiple bins
        dist = stats["counter_distribution"]
        self.assertGreater(len(dist), 1)

        # Should include at least "0" bin and some non-zero bins
        self.assertIn("0", dist)

        # Verify some counters have specific values
        non_zero_bins = [bin_label for bin_label in dist.keys() if bin_label != "0"]
        self.assertGreater(len(non_zero_bins), 0)

    def test_memory_breakdown(self):
        """Test the memory breakdown calculation."""
        cms = CountMinSketch(width=100, depth=5)

        # Get stats with memory breakdown
        stats = cms.get_stats()

        # Check memory breakdown
        self.assertIn("memory_breakdown", stats)
        breakdown = stats["memory_breakdown"]

        # Should include key components
        self.assertIn("base_object", breakdown)
        self.assertIn("counter_arrays", breakdown)
        self.assertIn("counter_container", breakdown)
        self.assertIn("attributes", breakdown)
        self.assertIn("tracking_structures", breakdown)
        self.assertIn("cached_stats", breakdown)
        self.assertIn("total", breakdown)

        # Total should equal the sum of components
        components_sum = (
            breakdown["base_object"]
            + breakdown["counter_arrays"]
            + breakdown["counter_container"]
            + breakdown["attributes"]
            + breakdown["tracking_structures"]
            + breakdown["cached_stats"]
        )

        self.assertEqual(breakdown["total"], components_sum)

        # Enable hash tracking and check the change in memory usage
        cms.enable_hash_tracking()

        # Add some data to populate tracking structures
        for i in range(50):
            cms.update(f"item-{i}")

        # Get updated stats
        updated_stats = cms.get_stats()
        updated_breakdown = updated_stats["memory_breakdown"]

        # Tracking structures should now use more memory
        self.assertGreater(
            updated_breakdown["tracking_structures"], breakdown["tracking_structures"]
        )

        # Total should still equal sum of components
        updated_components_sum = (
            updated_breakdown["base_object"]
            + updated_breakdown["counter_arrays"]
            + updated_breakdown["counter_container"]
            + updated_breakdown["attributes"]
            + updated_breakdown["tracking_structures"]
            + updated_breakdown["cached_stats"]
        )

        self.assertEqual(updated_breakdown["total"], updated_components_sum)

        # Estimate size should be approximately close to the total in breakdown
        # Allow a larger margin (25%) due to potential differences in how memory is calculated
        # across different systems and Python versions
        estimated_size = cms.estimate_size()
        self.assertLess(
            abs(estimated_size - updated_breakdown["total"])
            / max(estimated_size, updated_breakdown["total"]),
            0.25,  # Within 25% relative difference
            f"Memory estimate {estimated_size} differs too much from breakdown total {updated_breakdown['total']}",
        )

    def test_stats_caching(self):
        """Test that stats are properly cached and invalidated."""
        cms = CountMinSketch(width=50, depth=3)

        # Add some initial data
        for i in range(100):
            cms.update(f"item-{i}")

        # Get stats - should calculate and cache
        self.assertIsNone(cms._cached_stats)
        stats1 = cms.get_stats()
        self.assertIsNotNone(cms._cached_stats)

        # Get stats again - should use cache
        cms._cached_stats["test_marker"] = (
            "CACHED"  # Mark the cache to verify it's returned
        )
        stats2 = cms.get_stats()
        self.assertIn("test_marker", stats2)

        # Add more data - should invalidate cache
        cms.update("new-item")
        self.assertIsNone(cms._cached_stats)

        # Get stats after update - should recalculate
        stats3 = cms.get_stats()
        self.assertNotIn("test_marker", stats3)

        # Clear sketch - should invalidate cache
        cms._cached_stats = {"test_marker": "CACHED"}
        cms.clear()
        self.assertIsNone(cms._cached_stats)

        # Enable hash tracking - should invalidate cache
        cms._cached_stats = {"test_marker": "CACHED"}
        cms.enable_hash_tracking()
        self.assertIsNone(cms._cached_stats)

        # Disable hash tracking - should invalidate cache
        cms._cached_stats = {"test_marker": "CACHED"}
        cms.disable_hash_tracking()
        self.assertIsNone(cms._cached_stats)
