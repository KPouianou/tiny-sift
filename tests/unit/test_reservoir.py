"""
Unit tests for Reservoir Sampling algorithms.
"""

import json
import random
import unittest
from collections import Counter

from tiny_sift.algorithms.reservoir import ReservoirSampling, WeightedReservoirSampling


class TestReservoirSampling(unittest.TestCase):
    """Test cases for standard Reservoir Sampling."""
    
    def test_init(self):
        """Test initialization with valid and invalid parameters."""
        # Valid initialization
        rs = ReservoirSampling(size=10)
        self.assertEqual(rs._size, 10)
        self.assertEqual(len(rs.get_sample()), 0)
        
        # Invalid size
        with self.assertRaises(ValueError):
            ReservoirSampling(size=0)
        
        with self.assertRaises(ValueError):
            ReservoirSampling(size=-5)
    
    def test_update_small_stream(self):
        """Test updating with a stream smaller than the reservoir size."""
        rs = ReservoirSampling(size=10)
        
        # Add 5 items (less than reservoir size)
        for i in range(5):
            rs.update(i)
        
        # Check that all items are in the sample
        sample = rs.get_sample()
        self.assertEqual(len(sample), 5)
        self.assertEqual(set(sample), set(range(5)))
        self.assertEqual(rs.items_processed, 5)
    
    def test_update_large_stream(self):
        """Test updating with a stream larger than the reservoir size."""
        # Use a fixed seed for reproducibility
        rs = ReservoirSampling(size=10, seed=42)
        
        # Add 100 items (more than reservoir size)
        for i in range(100):
            rs.update(i)
        
        # Check that the sample has the correct size
        sample = rs.get_sample()
        self.assertEqual(len(sample), 10)
        self.assertEqual(rs.items_processed, 100)
        
        # Check that all items in the sample are from the stream
        for item in sample:
            self.assertIn(item, range(100))
    
    def test_statistical_properties(self):
        """Test the statistical properties of the reservoir sampling."""
        size = 100
        stream_size = 10000
        samples = 100
        
        # Run multiple samplings to check distribution
        all_samples = []
        for _ in range(samples):
            rs = ReservoirSampling(size=size)
            for i in range(stream_size):
                rs.update(i)
            all_samples.extend(rs.get_sample())
        
        # Count occurrences of each item
        counter = Counter(all_samples)
        
        # Each item should appear approximately the same number of times
        # The expected count for each item is:
        # (size / stream_size) * samples * size
        expected_count = (size / stream_size) * samples * size
        tolerance = 0.3  # Allow for some statistical variation
        
        # Check a few random items
        for i in random.sample(range(stream_size), 20):
            self.assertGreaterEqual(counter[i], expected_count * (1 - tolerance))
            self.assertLessEqual(counter[i], expected_count * (1 + tolerance))
    
    def test_serialization(self):
        """Test serialization and deserialization."""
        rs = ReservoirSampling(size=10, seed=42)
        
        # Add some items
        for i in range(20):
            rs.update(i)
        
        # Serialize to dict
        data = rs.to_dict()
        
        # Check dict contents
        self.assertEqual(data["type"], "ReservoirSampling")
        self.assertEqual(data["size"], 10)
        self.assertEqual(data["items_processed"], 20)
        self.assertEqual(len(data["reservoir"]), 10)
        
        # Deserialize from dict
        rs2 = ReservoirSampling.from_dict(data)
        
        # Check that the deserialized object matches the original
        self.assertEqual(rs2._size, rs._size)
        self.assertEqual(rs2._items_processed, rs._items_processed)
        self.assertEqual(rs2.get_sample(), rs.get_sample())
        
        # Test JSON serialization
        json_str = rs.serialize(format='json')
        rs3 = ReservoirSampling.deserialize(json_str, format='json')
        
        # Check that the deserialized object matches the original
        self.assertEqual(rs3._size, rs._size)
        self.assertEqual(rs3._items_processed, rs._items_processed)
        self.assertEqual(rs3.get_sample(), rs.get_sample())
    
    def test_merge(self):
        """Test merging two reservoirs."""
        # Create two reservoirs with different streams
        rs1 = ReservoirSampling(size=10, seed=42)
        for i in range(50):
            rs1.update(i)
        
        rs2 = ReservoirSampling(size=10, seed=43)
        for i in range(50, 100):
            rs2.update(i)
        
        # Verify item counts before merging
        self.assertEqual(rs1.items_processed, 50)
        self.assertEqual(rs2.items_processed, 50)
        
        # Merge the reservoirs
        rs_merged = rs1.merge(rs2)
        
        # Check that the merged reservoir has the correct size
        self.assertEqual(len(rs_merged.get_sample()), 10)
        # Check that items_processed is correctly combined
        self.assertEqual(rs_merged.items_processed, 100)
        
        # Check that all items in the merged sample are from the streams
        for item in rs_merged.get_sample():
            self.assertIn(item, range(100))
    
    def test_estimate_size(self):
        """Test memory usage estimation."""
        rs = ReservoirSampling(size=10)
        
        # Add some items
        for i in range(10):
            rs.update(i)
        
        # Check that the estimated size is positive
        size = rs.estimate_size()
        self.assertGreater(size, 0)
        
        # Check that the size increases with more complex items
        rs2 = ReservoirSampling(size=10)
        for i in range(10):
            rs2.update("x" * 1000)  # Large strings
        
        self.assertGreater(rs2.estimate_size(), size)


class TestWeightedReservoirSampling(unittest.TestCase):
    """Test cases for Weighted Reservoir Sampling (Algorithm A-Exp-Res)."""
    
    def test_init(self):
        """Test initialization with valid and invalid parameters."""
        # Valid initialization
        wrs = WeightedReservoirSampling(size=10)
        self.assertEqual(wrs._size, 10)
        self.assertEqual(len(wrs.get_sample()), 0)
        
        # Invalid size
        with self.assertRaises(ValueError):
            WeightedReservoirSampling(size=0)
    
    def test_update_with_weights(self):
        """Test updating with weighted items."""
        wrs = WeightedReservoirSampling(size=5, seed=42)
        
        # Add items with different weights
        wrs.update("A", weight=10.0)
        wrs.update("B", weight=1.0)
        wrs.update("C", weight=5.0)
        
        # Check that all items are in the sample
        sample = wrs.get_sample()
        self.assertEqual(len(sample), 3)
        self.assertEqual(set(sample), set(["A", "B", "C"]))
        
        # Check total weight
        self.assertEqual(wrs._total_weight, 16.0)
        
        # Invalid weight
        with self.assertRaises(ValueError):
            wrs.update("D", weight=0.0)
        
        with self.assertRaises(ValueError):
            wrs.update("E", weight=-1.0)
    
    def test_weighted_distribution(self):
        """Test that items with higher weights appear more frequently."""
        size = 1000
        samples = 100
        
        # Items with their weights
        items = {
            "A": 10.0,  # High weight
            "B": 1.0,   # Low weight
            "C": 5.0    # Medium weight
        }
        
        # Run multiple samplings to check distribution
        counts = Counter()
        for _ in range(samples):
            wrs = WeightedReservoirSampling(size=size, seed=random.randint(0, 10000))
            
            # Add each item many times to ensure statistical significance
            for _ in range(10000):
                for item, weight in items.items():
                    wrs.update(item, weight)
            
            # Count occurrences in the sample
            counts.update(wrs.get_sample())
        
        # Check that higher weighted items appear more frequently
        # The ratio of counts should be approximately the ratio of weights
        count_ratio_ab = counts["A"] / counts["B"]
        weight_ratio_ab = items["A"] / items["B"]
        
        count_ratio_ac = counts["A"] / counts["C"]
        weight_ratio_ac = items["A"] / items["C"]
        
        count_ratio_bc = counts["B"] / counts["C"]
        weight_ratio_bc = items["B"] / items["C"]
        
        # Allow for some statistical variation
        tolerance = 0.3
        
        self.assertGreaterEqual(count_ratio_ab, weight_ratio_ab * (1 - tolerance))
        self.assertLessEqual(count_ratio_ab, weight_ratio_ab * (1 + tolerance))
        
        self.assertGreaterEqual(count_ratio_ac, weight_ratio_ac * (1 - tolerance))
        self.assertLessEqual(count_ratio_ac, weight_ratio_ac * (1 + tolerance))
        
        self.assertGreaterEqual(count_ratio_bc, weight_ratio_bc * (1 - tolerance))
        self.assertLessEqual(count_ratio_bc, weight_ratio_bc * (1 + tolerance))
    
    def test_serialization(self):
        """Test serialization and deserialization."""
        wrs = WeightedReservoirSampling(size=10, seed=42)
        
        # Add some items with weights
        wrs.update("A", weight=10.0)
        wrs.update("B", weight=1.0)
        wrs.update("C", weight=5.0)
        
        # Serialize to dict
        data = wrs.to_dict()
        
        # Check dict contents
        self.assertEqual(data["type"], "WeightedReservoirSampling")
        self.assertEqual(data["size"], 10)
        self.assertEqual(data["items_processed"], 3)
        self.assertEqual(data["total_weight"], 16.0)
        self.assertEqual(len(data["weighted_reservoir"]), 3)
        
        # Deserialize from dict
        wrs2 = WeightedReservoirSampling.from_dict(data)
        
        # Check that the deserialized object matches the original
        self.assertEqual(wrs2._size, wrs._size)
        self.assertEqual(wrs2._items_processed, wrs._items_processed)
        self.assertEqual(wrs2._total_weight, wrs._total_weight)
        self.assertEqual(wrs2.get_sample(), wrs.get_sample())
        
        # Test JSON serialization
        json_str = wrs.serialize(format='json')
        wrs3 = WeightedReservoirSampling.deserialize(json_str, format='json')
        
        # Check that the deserialized object matches the original
        self.assertEqual(wrs3._size, wrs._size)
        self.assertEqual(wrs3._items_processed, wrs._items_processed)
        self.assertEqual(wrs3._total_weight, wrs._total_weight)
        self.assertEqual(wrs3.get_sample(), wrs.get_sample())
    
    def test_get_weighted_sample(self):
        """Test getting weighted sample with weights."""
        wrs = WeightedReservoirSampling(size=5, seed=42)
        
        # Add items with different weights
        items = [("A", 10.0), ("B", 1.0), ("C", 5.0)]
        for item, weight in items:
            wrs.update(item, weight)
        
        # Get the weighted sample
        weighted_sample = wrs.get_weighted_sample()
        
        # Check that the weighted sample contains the correct items with weights
        self.assertEqual(len(weighted_sample), 3)
        
        # Convert to dict for easier comparison
        sample_dict = {item: weight for item, weight in weighted_sample}
        
        # Check each item's weight
        for item, weight in items:
            self.assertIn(item, sample_dict)
            self.assertEqual(sample_dict[item], weight)


if __name__ == "__main__":
    unittest.main()