"""
Unit tests for Exponential Histogram algorithm.
"""

import json
import math
import random
import time
import unittest
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from tiny_sift.algorithms.histogram import ExponentialHistogram, Bucket


class TestExponentialHistogram(unittest.TestCase):
    """Test cases for Exponential Histogram."""

    def test_init(self):
        """Test initialization with valid and invalid parameters."""
        # Valid initialization
        eh = ExponentialHistogram(window_size=1000, error_bounds=0.01)
        self.assertEqual(eh._window_size, 1000)
        self.assertEqual(eh._error_bound, 0.01)
        self.assertEqual(eh._k, 100)  # 1/0.01 = 100
        self.assertFalse(eh._is_time_based)

        # Time-based window
        eh_time = ExponentialHistogram(
            window_size=3600, error_bounds=0.05, is_time_based=True
        )
        self.assertEqual(eh_time._window_size, 3600)
        self.assertEqual(eh_time._error_bound, 0.05)
        self.assertEqual(eh_time._k, 20)  # 1/0.05 = 20
        self.assertTrue(eh_time._is_time_based)

        # Invalid error bound
        with self.assertRaises(ValueError):
            ExponentialHistogram(window_size=1000, error_bounds=0)

        with self.assertRaises(ValueError):
            ExponentialHistogram(window_size=1000, error_bounds=1.5)

        # Invalid window size
        with self.assertRaises(ValueError):
            ExponentialHistogram(window_size=0, error_bounds=0.01)

    def test_update_count_based(self):
        """Test updating with count-based window."""
        eh = ExponentialHistogram(window_size=10, error_bounds=0.1)

        # Add 5 items with different values
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            eh.update(value=value)

        # Check totals
        self.assertEqual(eh._total_count, 5)
        self.assertEqual(eh._total_sum, sum(values))

        # Check estimate functions
        self.assertEqual(eh.estimate_count(), 5)
        self.assertEqual(eh.estimate_sum(), sum(values))

        # Check window stats
        stats = eh.get_window_stats()
        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["sum"], sum(values))
        self.assertEqual(stats["average"], sum(values) / 5)

    def test_window_expiration_count_based(self):
        """Test that items are correctly expired from count-based windows."""
        eh = ExponentialHistogram(window_size=5, error_bounds=0.1)

        # Add 10 items (exceeding window size)
        for i in range(10):
            eh.update(value=float(i))

        # The window should contain only the last 5 items (5,6,7,8,9)
        self.assertEqual(eh.estimate_count(), 5)
        self.assertEqual(eh.estimate_sum(), 5 + 6 + 7 + 8 + 9)

        # Check average
        self.assertEqual(eh.get_window_stats()["average"], (5 + 6 + 7 + 8 + 9) / 5)

        # Add one more item, the oldest (5) should be removed
        eh.update(value=10.0)

        self.assertEqual(eh.estimate_count(), 5)
        self.assertEqual(eh.estimate_sum(), 6 + 7 + 8 + 9 + 10)

    def test_update_time_based(self):
        """Test updating with time-based window."""
        # Use a small window of 2 seconds
        eh = ExponentialHistogram(window_size=2, error_bounds=0.1, is_time_based=True)

        # Add items with timestamps
        # t = 0s
        eh.update(value=1.0, timestamp=1000)  # 1000s
        # t = 0.5s
        eh.update(value=2.0, timestamp=1000.5)  # 1000.5s
        # t = 1s
        eh.update(value=3.0, timestamp=1001)  # 1001s
        # t = 2s
        eh.update(value=4.0, timestamp=1002)  # 1002s
        # t = 2.5s
        eh.update(value=5.0, timestamp=1002.5)  # 1002.5s

        # At this point, we should have aged out the first value (1.0)
        # since the window is 2 seconds and 1002.5 - 1000 > 2
        self.assertEqual(eh.estimate_count(), 4)
        self.assertEqual(eh.estimate_sum(), 2.0 + 3.0 + 4.0 + 5.0)

        # Let's add one more at t = 3s
        eh.update(value=6.0, timestamp=1003)  # 1003s

        # Now we should have aged out the second value (2.0) as well
        self.assertEqual(eh.estimate_count(), 4)
        self.assertEqual(eh.estimate_sum(), 3.0 + 4.0 + 5.0 + 6.0)

    def test_bucket_merging(self):
        """Test that buckets are correctly merged when we exceed k."""
        # Use a small k value (k = 2, error_bounds = 0.5)
        eh = ExponentialHistogram(window_size=100, error_bounds=0.5)
        self.assertEqual(eh._k, 2)  # 1/0.5 = 2

        # Add 5 items to force bucket merges
        for i in range(5):
            eh.update(value=1.0)

        # Check the bucket structure
        # We should have merged 1-size buckets multiple times
        self.assertLessEqual(len(eh._buckets[1]), 2)  # At most 2 buckets of size 1

        # We should have some larger buckets from merges
        total_buckets = sum(len(buckets) for buckets in eh._buckets.values())
        self.assertGreaterEqual(total_buckets, 2)  # At least 2 buckets total

        # But the count and sum should still be correct
        self.assertEqual(eh.estimate_count(), 5)
        self.assertEqual(eh.estimate_sum(), 5.0)

        # Add more items to force more merges
        for i in range(5):
            eh.update(value=2.0)

        # Check totals again
        self.assertEqual(eh.estimate_count(), 10)
        self.assertEqual(eh.estimate_sum(), 5.0 + 10.0)

    def test_min_max_tracking(self):
        """Test tracking of min and max values."""
        eh = ExponentialHistogram(window_size=100, error_bounds=0.1, track_min_max=True)

        # Add some values
        values = [5.0, 2.0, 8.0, 1.0, 10.0]
        for value in values:
            eh.update(value=value)

        # Check min and max
        min_val, max_val = eh.estimate_min_max()
        self.assertEqual(min_val, 1.0)
        self.assertEqual(max_val, 10.0)

        # Check via stats
        stats = eh.get_window_stats()
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 10.0)

        # Add more values to force bucket merges
        for _ in range(20):
            eh.update(value=3.0)

        # Min and max should still be preserved
        min_val, max_val = eh.estimate_min_max()
        self.assertEqual(min_val, 1.0)
        self.assertEqual(max_val, 10.0)

        # Now let's add a new min and max
        eh.update(value=0.5)  # New min
        eh.update(value=15.0)  # New max

        min_val, max_val = eh.estimate_min_max()
        self.assertEqual(min_val, 0.5)
        self.assertEqual(max_val, 15.0)

    def test_serialization(self):
        """Test serialization and deserialization."""
        eh = ExponentialHistogram(window_size=100, error_bounds=0.1, track_min_max=True)

        # Add some items
        for i in range(10):
            eh.update(value=float(i))

        # Serialize to dict
        data = eh.to_dict()

        # Check dict contents
        self.assertEqual(data["type"], "ExponentialHistogram")
        self.assertEqual(data["window_size"], 100)
        self.assertEqual(data["error_bounds"], 0.1)
        self.assertEqual(data["total_count"], 10)
        self.assertEqual(data["total_sum"], sum(range(10)))

        # Deserialize from dict
        eh2 = ExponentialHistogram.from_dict(data)

        # Check that the deserialized object matches the original
        self.assertEqual(eh2._window_size, eh._window_size)
        self.assertEqual(eh2._error_bound, eh._error_bound)
        self.assertEqual(eh2._track_min_max, eh._track_min_max)
        self.assertEqual(eh2._total_count, eh._total_count)
        self.assertEqual(eh2._total_sum, eh._total_sum)

        # Check that estimates match
        self.assertEqual(eh2.estimate_count(), eh.estimate_count())
        self.assertEqual(eh2.estimate_sum(), eh.estimate_sum())

        # Check min/max
        min1, max1 = eh.estimate_min_max()
        min2, max2 = eh2.estimate_min_max()
        self.assertEqual(min1, min2)
        self.assertEqual(max1, max2)

        # Test JSON serialization
        json_str = eh.serialize(format="json")
        eh3 = ExponentialHistogram.deserialize(json_str, format="json")

        # Check that the deserialized object matches
        self.assertEqual(eh3.estimate_count(), eh.estimate_count())
        self.assertEqual(eh3.estimate_sum(), eh.estimate_sum())

    def test_merge(self):
        """Test merging two histograms."""
        # Create two histograms
        eh1 = ExponentialHistogram(window_size=100, error_bounds=0.1)
        eh2 = ExponentialHistogram(window_size=100, error_bounds=0.1)

        # Add different items to each
        for i in range(5):
            eh1.update(value=1.0)

        for i in range(5):
            eh2.update(value=2.0)

        # Merge the histograms
        merged = eh1.merge(eh2)

        # Check the merged histogram
        self.assertEqual(merged.estimate_count(), 10)
        self.assertEqual(merged.estimate_sum(), 5.0 + 10.0)

        # Try merging with incompatible parameters
        eh3 = ExponentialHistogram(window_size=200, error_bounds=0.1)
        with self.assertRaises(ValueError):
            eh1.merge(eh3)

        eh4 = ExponentialHistogram(window_size=100, error_bounds=0.2)
        with self.assertRaises(ValueError):
            eh1.merge(eh4)

        eh5 = ExponentialHistogram(
            window_size=100, error_bounds=0.1, is_time_based=True
        )
        with self.assertRaises(ValueError):
            eh1.merge(eh5)

    def test_accuracy(self):
        """Test the accuracy of the histogram estimates."""
        # Use a moderate error bound
        error_bounds = 0.1
        eh = ExponentialHistogram(window_size=1000, error_bounds=error_bounds)

        # Add a sequence of values
        true_sum = 0
        for i in range(500):
            value = float(i)
            eh.update(value=value)
            true_sum += value

        # The count should be exact for smaller windows
        self.assertEqual(eh.estimate_count(), 500)

        # The sum should be within error_bounds of the true sum
        estimated_sum = eh.estimate_sum()
        relative_error = abs(estimated_sum - true_sum) / true_sum
        self.assertLessEqual(relative_error, error_bounds)

        # Add more items to fill the window
        for i in range(500, 1000):
            value = float(i)
            eh.update(value=value)

        # Window should be approximately full, within error bounds
        count = eh.estimate_count()
        self.assertGreaterEqual(count, int(1000 * (1 - error_bounds)))
        self.assertLessEqual(count, int(1000 * (1 + error_bounds)))

        # Add more items to force expiration
        for i in range(1000, 1500):
            value = float(i)
            eh.update(value=value)

        # Count should still be approximately the window size, within error bounds
        count = eh.estimate_count()
        self.assertGreaterEqual(count, int(1000 * (1 - error_bounds)))
        self.assertLessEqual(count, int(1000 * (1 + error_bounds)))

        # The window should contain approximately the last 1000 items
        expected_sum = sum(float(i) for i in range(500, 1500))
        estimated_sum = eh.estimate_sum()
        relative_error = abs(estimated_sum - expected_sum) / expected_sum
        self.assertLessEqual(relative_error, 0.1)

    def test_compress(self):
        """Test the compress method."""
        eh = ExponentialHistogram(window_size=1000, error_bounds=0.1)

        # Add items
        for i in range(100):
            eh.update(value=1.0)

        # Check initial bucket structure
        initial_buckets = sum(len(buckets) for buckets in eh._buckets.values())

        # Compress the histogram
        eh.compress()

        # Check that we have fewer buckets after compression
        compressed_buckets = sum(len(buckets) for buckets in eh._buckets.values())
        self.assertLessEqual(compressed_buckets, initial_buckets)

        # But the count and sum should still be correct
        self.assertEqual(eh.estimate_count(), 100)
        self.assertEqual(eh.estimate_sum(), 100.0)

    def test_clear(self):
        """Test clearing the histogram."""
        eh = ExponentialHistogram(window_size=100, error_bounds=0.1)

        # Add some items
        for i in range(10):
            eh.update(value=float(i))

        # Verify non-zero state
        self.assertEqual(eh.estimate_count(), 10)
        self.assertGreater(eh.estimate_sum(), 0)

        # Clear the histogram
        eh.clear()

        # Verify state is reset
        self.assertEqual(eh.estimate_count(), 0)
        self.assertEqual(eh.estimate_sum(), 0.0)
        self.assertEqual(len(eh._buckets), 0)
        self.assertEqual(eh._items_processed, 0)

    def test_create_from_error_rate(self):
        """Test creating a histogram from error rate parameters."""
        # Create a histogram with specific error bound
        eh = ExponentialHistogram.create_from_error_rate(
            relative_error=0.05, window_size=200, is_time_based=True
        )

        # Check parameters
        self.assertEqual(eh._error_bound, 0.05)
        self.assertEqual(eh._window_size, 200)
        self.assertTrue(eh._is_time_based)
        self.assertEqual(eh._k, 20)  # 1/0.05 = 20

        # Test with invalid error rate
        with self.assertRaises(ValueError):
            ExponentialHistogram.create_from_error_rate(relative_error=-0.1)

        with self.assertRaises(ValueError):
            ExponentialHistogram.create_from_error_rate(relative_error=1.5)

    def test_estimate_size(self):
        """Test memory usage estimation."""
        eh = ExponentialHistogram(window_size=1000, error_bounds=0.1)

        # Empty histogram should have a minimum size
        initial_size = eh.estimate_size()
        self.assertGreater(initial_size, 0)

        # Add items to increase memory usage
        for i in range(100):
            eh.update(value=float(i))

        # Size should increase
        filled_size = eh.estimate_size()
        self.assertGreater(filled_size, initial_size)

    def test_time_based_with_real_time(self):
        """Test time-based window with actual time passing."""
        # This is a timing-dependent test, so it may be flaky
        # We'll use a very small window (0.1 seconds)
        eh = ExponentialHistogram(window_size=0.1, error_bounds=0.1, is_time_based=True)

        # Add an item now
        eh.update(value=1.0)

        # Verify it's in the window
        self.assertEqual(eh.estimate_count(), 1)

        # Wait for 0.2 seconds (longer than the window)
        time.sleep(0.2)

        # Add another item to trigger window expiration check
        eh.update(value=2.0)

        # The first item should have expired
        self.assertEqual(eh.estimate_count(), 1)
        self.assertEqual(eh.estimate_sum(), 2.0)

    def test_realistic_use_case(self):
        """Test a more realistic use case with mixed values and timestamp patterns."""
        # Create a histogram with 1 hour window and 1% error
        eh = ExponentialHistogram(
            window_size=3600, error_bounds=0.01, is_time_based=True, track_min_max=True
        )

        # Simulate a stream of measurements over 2 hours
        # Each measurement has a timestamp and a value
        # We'll generate 1 measurement per minute
        base_time = 1000000  # Some arbitrary start time

        # Keep track of what should be in the window
        expected_values = []

        # Simulate 2 hours of data (120 minutes)
        for minute in range(120):
            # Calculate timestamp for this measurement
            timestamp = base_time + (minute * 60)

            # Generate a random value (e.g., temperature reading)
            value = 20.0 + random.uniform(-5.0, 5.0)

            # Add to histogram
            eh.update(value=value, timestamp=timestamp)

            # Keep track of values in the last hour
            expected_values.append((timestamp, value))

            # Remove values older than 1 hour
            cutoff_time = timestamp - 3600
            expected_values = [(t, v) for t, v in expected_values if t >= cutoff_time]

            # Every 10 minutes, check the histogram against expected values
            if minute % 10 == 9:
                # Calculate expected statistics
                expected_count = len(expected_values)
                expected_sum = sum(v for _, v in expected_values)
                expected_min = min(v for _, v in expected_values)
                expected_max = max(v for _, v in expected_values)

                # Get histogram statistics
                stats = eh.get_window_stats()

                # Check with appropriate tolerance for the error bound
                self.assertEqual(stats["count"], expected_count)

                # Sum should be within 1% of expected (our error bound)
                sum_error = (
                    abs(stats["sum"] - expected_sum) / expected_sum
                    if expected_sum
                    else 0
                )
                self.assertLessEqual(sum_error, 0.01)

                # Min/max should be exact
                self.assertEqual(stats["min"], expected_min)
                self.assertEqual(stats["max"], expected_max)

        # Final check of entire window
        stats = eh.get_window_stats()
        # The count should be approximately 60 minutes, within error bounds
        expected_count = 60
        max_count = int(math.ceil(expected_count * (1 + eh._error_bound)))
        min_count = int(math.floor(expected_count * (1 - eh._error_bound)))
        self.assertGreaterEqual(stats["count"], min_count)
        self.assertLessEqual(stats["count"], max_count)


if __name__ == "__main__":
    unittest.main()
