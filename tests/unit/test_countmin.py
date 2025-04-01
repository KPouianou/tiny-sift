"""
Unit tests for Count-Min Sketch algorithm.
"""

import json
import random
import unittest
from collections import Counter

from tiny_sift.algorithms.countmin import CountMinSketch


class TestCountMinSketch(unittest.TestCase):
    """Test cases for Count-Min Sketch."""

    def test_init(self):
        """Test initialization with valid and invalid parameters."""
        # Valid initialization
        cms = CountMinSketch(width=10, depth=5)
        self.assertEqual(cms._width, 10)
        self.assertEqual(cms._depth, 5)

        # Invalid width
        with self.assertRaises(ValueError):
            CountMinSketch(width=0, depth=5)

        # Invalid depth
        with self.assertRaises(ValueError):
            CountMinSketch(width=10, depth=0)

    def test_update_and_query(self):
        """Test updating and querying frequencies."""
        cms = CountMinSketch(width=100, depth=5, seed=42)

        # Add some items
        cms.update("A", count=5)
        cms.update("B", count=3)
        cms.update("A", count=2)

        # Check estimated frequencies
        self.assertEqual(cms.estimate_frequency("A"), 7)
        self.assertEqual(cms.estimate_frequency("B"), 3)
        self.assertEqual(cms.estimate_frequency("C"), 0)  # Not added

        # Test invalid update
        with self.assertRaises(ValueError):
            cms.update("D", count=-1)

        # Test zero count update (should be a no-op)
        cms.update("E", count=0)
        self.assertEqual(cms.estimate_frequency("E"), 0)

    def test_error_bounds(self):
        """Test that error bounds are calculated correctly."""
        width = 1000
        depth = 5
        cms = CountMinSketch(width=width, depth=depth)

        # Calculate expected bounds
        expected_epsilon = 2.718281828459045 / width  # e / width
        expected_delta = 0.006737946999085467  # e^(-depth)

        # Get actual bounds
        bounds = cms.error_bounds()

        # Check bounds are close to expected (allow for floating point errors)
        self.assertAlmostEqual(bounds["epsilon"], expected_epsilon, places=10)
        self.assertAlmostEqual(bounds["delta"], expected_delta, places=10)

    def test_collisions(self):
        """Test behavior with hash collisions."""
        # Use a very small width to force collisions
        cms = CountMinSketch(width=5, depth=3, seed=42)

        # Add many different items to cause collisions
        true_counts = {}
        for i in range(100):
            item = f"item-{i}"
            count = random.randint(1, 5)
            true_counts[item] = count
            cms.update(item, count=count)

        # Verify all estimates are at least the true counts
        for item, true_count in true_counts.items():
            est_count = cms.estimate_frequency(item)
            self.assertGreaterEqual(
                est_count,
                true_count,
                f"Estimate for {item} is {est_count}, which is less than true count {true_count}",
            )

    def test_heavy_hitters(self):
        """Test getting heavy hitters from candidates."""
        cms = CountMinSketch(width=1000, depth=5, seed=42)

        # Create a stream with some heavy hitters
        items = ["A", "B", "C", "D", "E"]
        counts = {"A": 1000, "B": 500, "C": 100, "D": 50, "E": 10}

        # Update the sketch
        for item, count in counts.items():
            cms.update(item, count)

        # Test get_heavy_hitters_from_candidates with different thresholds
        # Threshold 0.4 (40%) should include only A
        heavy_hitters = cms.get_heavy_hitters_from_candidates(items, 0.4)
        self.assertEqual(set(heavy_hitters.keys()), {"A"})

        # Threshold 0.2 (20%) should include A and B
        heavy_hitters = cms.get_heavy_hitters_from_candidates(items, 0.2)
        self.assertEqual(set(heavy_hitters.keys()), {"A", "B"})

        # Threshold 0.01 (1%) should include A, B, C, D
        heavy_hitters = cms.get_heavy_hitters_from_candidates(items, 0.01)
        self.assertEqual(set(heavy_hitters.keys()), {"A", "B", "C", "D"})

        # Test with invalid threshold
        with self.assertRaises(ValueError):
            cms.get_heavy_hitters_from_candidates(items, -0.1)

        with self.assertRaises(ValueError):
            cms.get_heavy_hitters_from_candidates(items, 1.1)

        # Test original get_heavy_hitters method (should raise NotImplementedError)
        with self.assertRaises(NotImplementedError):
            cms.get_heavy_hitters(0.1)

    def test_serialization(self):
        """Test serialization and deserialization."""
        cms = CountMinSketch(width=10, depth=5, seed=42)

        # Add some items
        cms.update("A", count=5)
        cms.update("B", count=3)

        # Serialize to dict
        data = cms.to_dict()

        # Check dict contents
        self.assertEqual(data["type"], "CountMinSketch")
        self.assertEqual(data["width"], 10)
        self.assertEqual(data["depth"], 5)
        self.assertEqual(data["total_frequency"], 8)

        # Deserialize from dict
        cms2 = CountMinSketch.from_dict(data)

        # Check that the deserialized object matches the original
        self.assertEqual(cms2._width, cms._width)
        self.assertEqual(cms2._depth, cms._depth)
        self.assertEqual(cms2._total_frequency, cms._total_frequency)
        self.assertEqual(cms2.estimate_frequency("A"), cms.estimate_frequency("A"))
        self.assertEqual(cms2.estimate_frequency("B"), cms.estimate_frequency("B"))

        # Test JSON serialization
        json_str = cms.serialize(format="json")
        cms3 = CountMinSketch.deserialize(json_str, format="json")

        # Check that the deserialized object matches the original
        self.assertEqual(cms3._width, cms._width)
        self.assertEqual(cms3._depth, cms._depth)
        self.assertEqual(cms3.estimate_frequency("A"), cms.estimate_frequency("A"))

    def test_merge(self):
        """Test merging two sketches."""
        # Create two sketches with different items
        cms1 = CountMinSketch(width=100, depth=5, seed=42)
        cms1.update("A", count=5)
        cms1.update("B", count=3)

        cms2 = CountMinSketch(width=100, depth=5, seed=42)
        cms2.update("B", count=2)
        cms2.update("C", count=4)

        # Merge the sketches
        merged = cms1.merge(cms2)

        # Check merged frequencies
        self.assertEqual(merged.estimate_frequency("A"), 5)
        self.assertEqual(merged.estimate_frequency("B"), 5)  # 3 + 2
        self.assertEqual(merged.estimate_frequency("C"), 4)

        # Check total frequency and items processed
        self.assertEqual(merged._total_frequency, 14)  # 5 + 3 + 2 + 4
        self.assertEqual(merged._items_processed, 4)  # Each item counts as one

        # Test merging with incompatible dimensions
        cms3 = CountMinSketch(width=50, depth=5, seed=42)
        with self.assertRaises(ValueError):
            cms1.merge(cms3)

    def test_create_from_error_rate(self):
        """Test creating a sketch from error rate parameters."""
        # Create a sketch with specific error bounds
        epsilon = 0.01  # 1% error
        delta = 0.001  # 0.1% probability of exceeding error

        cms = CountMinSketch.create_from_error_rate(epsilon, delta, seed=42)

        # Check dimensions are calculated correctly
        expected_width = 272  # ceil(e / epsilon)
        expected_depth = 7  # ceil(ln(1/delta))

        self.assertEqual(cms._width, expected_width)
        self.assertEqual(cms._depth, expected_depth)

        # Test with invalid error rates
        with self.assertRaises(ValueError):
            CountMinSketch.create_from_error_rate(-0.01, 0.001)

        with self.assertRaises(ValueError):
            CountMinSketch.create_from_error_rate(0.01, -0.001)

    def test_clear(self):
        """Test clearing the sketch."""
        cms = CountMinSketch(width=10, depth=5, seed=42)

        # Add some items
        cms.update("A", count=5)
        cms.update("B", count=3)

        # Verify counts
        self.assertEqual(cms.estimate_frequency("A"), 5)
        self.assertEqual(cms._total_frequency, 8)

        # Clear the sketch
        cms.clear()

        # Verify all counts are reset
        self.assertEqual(cms.estimate_frequency("A"), 0)
        self.assertEqual(cms.estimate_frequency("B"), 0)
        self.assertEqual(cms._total_frequency, 0)
        self.assertEqual(cms._items_processed, 0)

    def test_accuracy_with_increasing_width(self):
        """Test how accuracy improves with width."""
        # Create a dataset
        true_counts = Counter()
        stream = []

        # Generate a stream with Zipf distribution
        items = list(range(1000))
        weights = [1.0 / (i + 1) for i in range(len(items))]
        total = sum(weights)
        weights = [w / total for w in weights]

        # Generate 10,000 items
        for _ in range(10000):
            item = random.choices(items, weights=weights, k=1)[0]
            stream.append(item)
            true_counts[item] += 1

        # Test with different widths
        widths = [10, 100, 1000]
        avg_errors = []

        for width in widths:
            cms = CountMinSketch(width=width, depth=5, seed=42)

            # Process the stream
            for item in stream:
                cms.update(item)

            # Measure average error
            total_error = 0
            for item, true_count in true_counts.items():
                estimated = cms.estimate_frequency(item)
                error = estimated - true_count
                self.assertGreaterEqual(error, 0)  # Error should never be negative
                total_error += error

            avg_error = total_error / len(true_counts)
            avg_errors.append(avg_error)

        # Verify that error decreases as width increases
        for i in range(1, len(avg_errors)):
            self.assertLess(avg_errors[i], avg_errors[i - 1])


if __name__ == "__main__":
    unittest.main()
