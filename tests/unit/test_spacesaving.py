"""
Unit tests for Space-Saving algorithm.
"""

import json
import math
import random
import unittest
from collections import Counter

from tiny_sift.algorithms.spacesaving import SpaceSaving, CounterEntry


class TestCounterEntry(unittest.TestCase):
    """Test cases for the CounterEntry helper class."""

    def test_init(self):
        """Test initialization with different parameters."""
        # Default initialization
        entry = CounterEntry("test")
        self.assertEqual(entry.item, "test")
        self.assertEqual(entry.count, 1)
        self.assertEqual(entry.error, 0)

        # Custom values
        entry = CounterEntry("test", count=10, error=2)
        self.assertEqual(entry.item, "test")
        self.assertEqual(entry.count, 10)
        self.assertEqual(entry.error, 2)

    def test_comparison(self):
        """Test comparison operators."""
        e1 = CounterEntry("A", count=5)
        e2 = CounterEntry("B", count=10)
        e3 = CounterEntry("C", count=5)  # Same count as e1

        # Less than (used by heap)
        self.assertTrue(e1 < e2)
        self.assertFalse(e2 < e1)
        self.assertFalse(e1 < e3)  # Equal counts
        self.assertFalse(e3 < e1)

    def test_equality(self):
        """Test equality based on item."""
        e1 = CounterEntry("X", count=5)
        e2 = CounterEntry("X", count=10)  # Same item, different count
        e3 = CounterEntry("Y", count=5)  # Different item, same count

        # Equality based on item, not count
        self.assertEqual(e1, e2)
        self.assertNotEqual(e1, e3)
        self.assertNotEqual(e2, e3)

        # Not equal to other types
        self.assertNotEqual(e1, "X")
        self.assertNotEqual(e1, 5)

    def test_serialization(self):
        """Test to_dict and from_dict methods."""
        entry = CounterEntry("test_item", count=15, error=3)

        # Convert to dict
        data = entry.to_dict()
        self.assertEqual(data["item"], "test_item")
        self.assertEqual(data["count"], 15)
        self.assertEqual(data["error"], 3)

        # Convert back from dict
        entry2 = CounterEntry.from_dict(data)
        self.assertEqual(entry2.item, "test_item")
        self.assertEqual(entry2.count, 15)
        self.assertEqual(entry2.error, 3)


class TestSpaceSaving(unittest.TestCase):
    """Test cases for the SpaceSaving algorithm."""

    def test_init(self):
        """Test initialization with different parameters."""
        # Default initialization
        ss = SpaceSaving()
        self.assertEqual(ss._capacity, 100)  # Default
        self.assertEqual(len(ss._counters), 0)
        self.assertEqual(ss._total_count, 0)

        # Custom capacity
        ss = SpaceSaving(capacity=50)
        self.assertEqual(ss._capacity, 50)

        # Invalid capacity
        with self.assertRaises(ValueError):
            SpaceSaving(capacity=0)
        with self.assertRaises(ValueError):
            SpaceSaving(capacity=-10)

    def test_update_single_item(self):
        """Test updating with a single item."""
        ss = SpaceSaving(capacity=5)

        # Add one item
        ss.update("A")

        # Check state
        self.assertEqual(ss._total_count, 1)
        self.assertEqual(ss.items_processed, 1)
        self.assertEqual(len(ss._counters), 1)
        self.assertEqual(ss.estimate_frequency("A"), 1)
        self.assertEqual(ss.estimate_frequency("B"), 0)  # Item not seen yet

        # Add the same item again
        ss.update("A")

        # Check updated state
        self.assertEqual(ss._total_count, 2)
        self.assertEqual(ss.items_processed, 2)
        self.assertEqual(len(ss._counters), 1)
        self.assertEqual(ss.estimate_frequency("A"), 2)

    def test_update_with_replacement(self):
        """Test updating when the capacity is reached and replacement occurs."""
        ss = SpaceSaving(capacity=3)

        # Fill the capacity
        ss.update("A")
        ss.update("B")
        ss.update("C")

        # Verify initial state
        self.assertEqual(len(ss._counters), 3)
        self.assertEqual(ss.estimate_frequency("A"), 1)
        self.assertEqual(ss.estimate_frequency("B"), 1)
        self.assertEqual(ss.estimate_frequency("C"), 1)

        # Add a new item, forcing replacement
        ss.update("D")

        # One of A, B, or C should have been replaced with D
        # Since they all have count 1, any could be replaced
        # In Space-Saving, the new item gets the minimum count (1) + the new count (1) = 2
        # This preserves the guarantee that no frequency is underestimated
        self.assertEqual(len(ss._counters), 3)  # Still at capacity
        self.assertEqual(ss.estimate_frequency("D"), 2)

        # The total count of items tracked should match the items processed
        tracked_total = sum(entry.count for entry in ss._counters)
        self.assertEqual(tracked_total, 4)
        self.assertEqual(ss._total_count, 4)
        self.assertEqual(ss.items_processed, 4)

    def test_update_with_count(self):
        """Test updating with a count greater than 1."""
        ss = SpaceSaving(capacity=3)

        # Add items with different counts
        ss.update("A", count=5)
        ss.update("B", count=3)

        # Check state
        self.assertEqual(ss._total_count, 8)
        self.assertEqual(ss.items_processed, 2)
        self.assertEqual(ss.estimate_frequency("A"), 5)
        self.assertEqual(ss.estimate_frequency("B"), 3)

        # Invalid count
        with self.assertRaises(ValueError):
            ss.update("C", count=-1)

        # Zero count (should be no-op)
        ss.update("D", count=0)
        self.assertEqual(ss._total_count, 8)  # Unchanged
        self.assertEqual(ss.items_processed, 2)  # Unchanged
        self.assertEqual(ss.estimate_frequency("D"), 0)

    def test_get_heavy_hitters(self):
        """Test identifying heavy hitters based on threshold."""
        ss = SpaceSaving(capacity=5)

        # Add items with different frequencies
        ss.update("A", count=50)  # 50% of total
        ss.update("B", count=30)  # 30% of total
        ss.update("C", count=15)  # 15% of total
        ss.update("D", count=5)  # 5% of total

        # Test different thresholds
        heavy_hitters_10 = ss.get_heavy_hitters(0.1)  # 10% threshold
        self.assertEqual(len(heavy_hitters_10), 3)
        self.assertTrue("A" in heavy_hitters_10)
        self.assertTrue("B" in heavy_hitters_10)
        self.assertTrue("C" in heavy_hitters_10)
        self.assertFalse("D" in heavy_hitters_10)

        heavy_hitters_25 = ss.get_heavy_hitters(0.25)  # 25% threshold
        self.assertEqual(len(heavy_hitters_25), 2)
        self.assertTrue("A" in heavy_hitters_25)
        self.assertTrue("B" in heavy_hitters_25)

        # Invalid threshold
        with self.assertRaises(ValueError):
            ss.get_heavy_hitters(-0.1)
        with self.assertRaises(ValueError):
            ss.get_heavy_hitters(1.1)

    def test_get_top_k(self):
        """Test getting the top-k most frequent items."""
        ss = SpaceSaving(capacity=10)

        # Add items with different frequencies
        items_to_add = []
        items_to_add.extend(["A"] * 50)
        items_to_add.extend(["B"] * 40)
        items_to_add.extend(["C"] * 30)
        items_to_add.extend(["D"] * 20)
        items_to_add.extend(["E"] * 10)

        for item in items_to_add:
            ss.update(item)

        # Get top 3
        top_3 = ss.get_top_k(3)
        self.assertEqual(len(top_3), 3)

        # Check order and values
        self.assertEqual(top_3[0][0], "A")
        self.assertEqual(top_3[1][0], "B")
        self.assertEqual(top_3[2][0], "C")

        self.assertEqual(top_3[0][1], 50)
        self.assertEqual(top_3[1][1], 40)
        self.assertEqual(top_3[2][1], 30)

        # Get all items (k=None)
        all_items = ss.get_top_k()
        self.assertEqual(len(all_items), 5)

        # Get more items than available
        big_k = ss.get_top_k(10)
        self.assertEqual(len(big_k), 5)  # Only 5 distinct items exist

    def test_estimate_frequency_error(self):
        """Test getting frequency estimates with error bounds."""
        ss = SpaceSaving(capacity=3)

        # Fill capacity
        ss.update("A", count=10)
        ss.update("B", count=7)
        ss.update("C", count=3)

        # Check frequencies and errors
        freq_a, error_a = ss.estimate_frequency_error("A")
        self.assertEqual(freq_a, 10)
        self.assertEqual(error_a, 0)  # No replacement yet

        # Add a new item that causes replacement
        ss.update("D", count=2)

        # Check for the new item
        freq_d, error_d = ss.estimate_frequency_error("D")
        self.assertEqual(freq_d, 3 + 2)  # Should be prev min (3) + new count (2)
        self.assertEqual(error_d, 3)  # Error should be the replaced count

        # Check for a non-tracked item
        freq_e, error_e = ss.estimate_frequency_error("E")
        self.assertEqual(freq_e, 0)
        # Error should be min count in current counters
        min_count = min(entry.count for entry in ss._counters)
        self.assertEqual(error_e, min_count)

    def test_query(self):
        """Test the query convenience method."""
        ss = SpaceSaving(capacity=3)
        ss.update("A", count=5)

        # Query with positional argument
        self.assertEqual(ss.query("A"), 5)
        self.assertEqual(ss.query("B"), 0)

        # Query with keyword argument
        self.assertEqual(ss.query(item="A"), 5)

        # Missing item argument
        with self.assertRaises(ValueError):
            ss.query()

    def test_merge(self):
        """Test merging two Space-Saving sketches."""
        ss1 = SpaceSaving(capacity=5)
        ss2 = SpaceSaving(capacity=5)

        # Add different items to each sketch
        ss1.update("A", count=10)
        ss1.update("B", count=8)
        ss1.update("C", count=6)

        ss2.update("C", count=4)  # Overlap with ss1
        ss2.update("D", count=7)
        ss2.update("E", count=3)

        # Merge the sketches
        merged = ss1.merge(ss2)

        # Check merged state
        self.assertEqual(merged._capacity, 5)
        self.assertEqual(merged._total_count, 38)  # 24 from ss1 + 14 from ss2
        self.assertEqual(merged.items_processed, 6)  # 3 from ss1 + 3 from ss2

        # Check frequencies in merged sketch
        self.assertEqual(merged.estimate_frequency("A"), 10)
        self.assertEqual(merged.estimate_frequency("B"), 8)
        self.assertEqual(merged.estimate_frequency("C"), 10)  # 6 + 4
        self.assertEqual(merged.estimate_frequency("D"), 7)

        # Check counts are preserved
        # Get top items to verify
        top_items = merged.get_top_k()
        top_dict = {item: freq for item, freq in top_items}

        # Total tracked should equal total items
        self.assertEqual(sum(top_dict.values()), 38)

        # Merge incompatible sketches
        ss3 = SpaceSaving(capacity=10)  # Different capacity
        with self.assertRaises(ValueError):
            ss1.merge(ss3)

        # Merge with non-SpaceSaving object
        with self.assertRaises(TypeError):
            ss1.merge("not a sketch")

    def test_serialization(self):
        """Test serializing and deserializing the sketch."""
        ss = SpaceSaving(capacity=5)

        # Add some items
        ss.update("A", count=15)
        ss.update("B", count=10)
        ss.update("C", count=5)

        # Serialize to dict
        data = ss.to_dict()

        # Check dictionary
        self.assertEqual(data["type"], "SpaceSaving")
        self.assertEqual(data["capacity"], 5)
        self.assertEqual(data["total_count"], 30)
        self.assertEqual(data["items_processed"], 3)
        self.assertEqual(len(data["counters"]), 3)

        # Deserialize
        ss2 = SpaceSaving.from_dict(data)

        # Check deserialized state
        self.assertEqual(ss2._capacity, 5)
        self.assertEqual(ss2._total_count, 30)
        self.assertEqual(ss2.items_processed, 3)
        self.assertEqual(ss2.estimate_frequency("A"), 15)
        self.assertEqual(ss2.estimate_frequency("B"), 10)
        self.assertEqual(ss2.estimate_frequency("C"), 5)

        # Check heap property
        min_entry = ss2._counters[0]
        self.assertEqual(min_entry.item, "C")
        self.assertEqual(min_entry.count, 5)

    def test_clear(self):
        """Test clearing the sketch."""
        ss = SpaceSaving(capacity=5)

        # Add some items
        ss.update("A", count=10)
        ss.update("B", count=5)

        # Clear the sketch
        ss.clear()

        # Check state after clearing
        self.assertEqual(len(ss._counters), 0)
        self.assertEqual(ss._total_count, 0)
        self.assertEqual(ss.items_processed, 0)
        self.assertEqual(ss.estimate_frequency("A"), 0)
        self.assertEqual(ss.estimate_frequency("B"), 0)

    def test_len(self):
        """Test __len__ method."""
        ss = SpaceSaving(capacity=5)

        # Empty sketch
        self.assertEqual(len(ss), 0)

        # Add some items
        ss.update("A")
        ss.update("B")

        # Length should be number of items being tracked
        self.assertEqual(len(ss), 2)

        # Add duplicate items
        ss.update("A")
        ss.update("A")

        # Length should still be the number of unique items
        self.assertEqual(len(ss), 2)

    def test_estimate_size(self):
        """Test memory usage estimation."""
        ss = SpaceSaving(capacity=10)

        # Empty sketch
        empty_size = ss.estimate_size()
        self.assertGreater(empty_size, 0)

        # Add some items
        for i in range(5):
            ss.update(f"item-{i}")

        # Size should increase
        filled_size = ss.estimate_size()
        self.assertGreater(filled_size, empty_size)

    def test_create_from_error_rate(self):
        """Test creating a sketch from error rate parameters."""
        # Test with typical values
        threshold = 0.05  # 5% frequency
        error_rate = 0.1  # 10% error

        ss = SpaceSaving.create_from_error_rate(threshold, error_rate)

        # Expected capacity is 1/(threshold * error_rate)
        expected_capacity = math.ceil(1.0 / (threshold * error_rate))
        self.assertEqual(ss._capacity, expected_capacity)

        # Invalid parameters
        with self.assertRaises(ValueError):
            SpaceSaving.create_from_error_rate(0, 0.1)
        with self.assertRaises(ValueError):
            SpaceSaving.create_from_error_rate(-0.1, 0.1)
        with self.assertRaises(ValueError):
            SpaceSaving.create_from_error_rate(1.1, 0.1)
        with self.assertRaises(ValueError):
            SpaceSaving.create_from_error_rate(0.1, 0)
        with self.assertRaises(ValueError):
            SpaceSaving.create_from_error_rate(0.1, 1.0)

    def test_error_bounds(self):
        """Test the error bounds of the algorithm with real data."""
        # Create a sketch with limited capacity
        capacity = 10
        ss = SpaceSaving(capacity=capacity)

        # Generate a stream with known distribution
        n_items = 1000

        # True counts
        true_counts = {"A": 300, "B": 200, "C": 100}

        # Create stream
        stream = []
        stream.extend(["A"] * true_counts["A"])
        stream.extend(["B"] * true_counts["B"])
        stream.extend(["C"] * true_counts["C"])

        # Add remaining items with a long tail distribution (all < 1%)
        remaining = n_items - sum(true_counts.values())
        for i in range(remaining):
            stream.append(f"rare-{i % 400}")  # Each rare item appears 1-2 times

        # Shuffle the stream
        random.shuffle(stream)

        # Process the stream
        for item in stream:
            ss.update(item)

        # Check error bounds for main items
        for item, true_count in true_counts.items():
            estimated, max_error = ss.estimate_frequency_error(item)

            # Error should not exceed theoretical bound: n_items / capacity
            theoretical_bound = n_items / capacity
            actual_error = abs(estimated - true_count)

            # Check that the error is within predicted bound
            self.assertLessEqual(
                actual_error,
                theoretical_bound,
                f"Error for {item} exceeds theoretical bound: {actual_error} > {theoretical_bound}",
            )

            # Reported max_error should be realistic
            self.assertLessEqual(
                actual_error,
                max_error,
                f"Actual error {actual_error} exceeds reported max error {max_error}",
            )


if __name__ == "__main__":
    unittest.main()
