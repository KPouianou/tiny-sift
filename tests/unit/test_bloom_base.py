"""
Unit tests for standard Bloom Filter implementation.
"""

import json
import math
import random
import unittest
import sys  # Make sure sys is imported for estimate_size usage in test
from collections import Counter

# Adjust import path based on your project structure
from tiny_sift.algorithms.bloom.base import BloomFilter


class TestBloomFilter(unittest.TestCase):
    """Test cases for Bloom Filter."""

    def test_init(self):
        """Test initialization with valid and invalid parameters."""
        # Valid initialization
        bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
        # These should be calculated based on the optimal values for 1000 items and 0.01 FPP
        # m = -(n * ln(p)) / (ln(2)^2) ≈ 9585 bits
        # k = (m/n) * ln(2) ≈ 7 hash functions
        self.assertAlmostEqual(bf._bit_size, 9585, delta=1)  # Check exact calc now
        self.assertAlmostEqual(bf._hash_count, 7, delta=1)  # Allow for ceiling

        # Test with different parameters
        bf = BloomFilter(expected_items=10000, false_positive_rate=0.001)
        # m ≈ 143776 bits, k ≈ 10
        self.assertAlmostEqual(bf._bit_size, 143776, delta=1)
        self.assertAlmostEqual(bf._hash_count, 10, delta=1)

        # Invalid expected_items
        with self.assertRaises(ValueError):
            BloomFilter(expected_items=0)

        # Invalid false_positive_rate
        with self.assertRaises(ValueError):
            BloomFilter(false_positive_rate=0)
        with self.assertRaises(ValueError):
            BloomFilter(false_positive_rate=1.5)
        with self.assertRaises(ValueError):
            BloomFilter(false_positive_rate=1.0)  # Boundary

    def test_update_and_contains(self):
        """Test adding items and checking for membership."""
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)

        # Add some items
        items_to_add = ["apple", "banana", "cherry", 123, 45.67, (1, "tuple")]
        items_not_added = ["orange", "grape", 987, ("other", "tuple")]

        for item in items_to_add:
            bf.update(item)
            self.assertTrue(
                bf.contains(item), f"Item {item} should be present after adding"
            )
            self.assertTrue(
                bf.query(item), f"Item {item} should query true after adding"
            )

        # Check items that should be in the filter
        for item in items_to_add:
            self.assertTrue(bf.contains(item), f"Item {item} should be present")
            self.assertTrue(bf.query(item), f"Item {item} should query true")

        # Check items that should not be in the filter (might be false positives)
        fp_count = 0
        for item in items_not_added:
            if bf.contains(item):
                fp_count += 1
            # We cannot assert False due to possibility of FPs
            # self.assertFalse(bf.contains(item)) # Incorrect test for BF

        # Check counts
        self.assertEqual(bf._items_processed, len(items_to_add))
        # _approximate_count might be less than _items_processed if items hash to same bits
        self.assertLessEqual(bf._approximate_count, len(items_to_add))

    def test_false_positives(self):
        """Test that false positives occur at approximately the expected rate."""
        # Use a larger false positive rate for more predictable testing
        target_fpp = 0.1
        n_items = 1000
        n_tests = 10000  # Test more items for better stats

        bf = BloomFilter(expected_items=n_items, false_positive_rate=target_fpp)

        # Add n_items items (fill to expected capacity)
        added_items = {f"item-{i}" for i in range(n_items)}
        for item in added_items:
            bf.update(item)

        # Test n_tests different items not in the filter
        false_positives = 0
        tested_items = {f"other-{i}" for i in range(n_tests)}

        for item in tested_items:
            self.assertNotIn(item, added_items)  # Ensure test item wasn't added
            if bf.contains(item):
                false_positives += 1

        # Check that false positive rate is approximately as expected
        observed_fpp = false_positives / n_tests
        print(
            f"\n[FPP Test] Target: {target_fpp:.4f}, Observed: {observed_fpp:.4f} ({false_positives}/{n_tests})"
        )

        # Allow for statistical variation - use a margin based on expected standard deviation
        # Expected FP count = n_tests * target_fpp
        # Std Dev ≈ sqrt(n_tests * target_fpp * (1 - target_fpp))
        expected_fps = n_tests * target_fpp
        std_dev = math.sqrt(expected_fps * (1 - target_fpp))
        margin = (
            3 * std_dev
        )  # Allow +/- 3 standard deviations (covers >99% for normal approx)

        self.assertLessEqual(
            false_positives, expected_fps + margin, "Too many false positives"
        )
        self.assertGreaterEqual(
            false_positives,
            expected_fps - margin,
            "Too few false positives (check test logic?)",
        )

        # Also check the filter's own reported FPP estimate
        estimated_fpp = bf.false_positive_probability()
        print(f"[FPP Test] Filter's Estimated FPP: {estimated_fpp:.4f}")
        self.assertAlmostEqual(
            estimated_fpp, target_fpp, delta=target_fpp * 0.5
        )  # Should be close to target

    def test_no_false_negatives(self):
        """Test that false negatives never occur."""
        bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)

        # Add 2000 items (double the expected capacity)
        test_items = {f"item-{i}" for i in range(2000)}  # Use set for uniqueness
        for item in test_items:
            bf.update(item)

        # Check that all items are still found (no false negatives)
        missing_items = 0
        for item in test_items:
            if not bf.contains(item):
                missing_items += 1
                print(f"False Negative detected for item: {item}")

        self.assertEqual(missing_items, 0, "False negatives detected!")

    def test_serialization(self):
        """Test serialization and deserialization."""
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01, seed=123)

        # Add some items
        items_added = {f"item-{i}" for i in range(50)}
        for item in items_added:
            bf.update(item)

        # Serialize to dict
        data = bf.to_dict()

        # Check dict contents
        self.assertEqual(data["type"], "BloomFilter")
        self.assertEqual(data["expected_items"], 100)
        self.assertEqual(data["false_positive_rate"], 0.01)
        self.assertEqual(data["items_processed"], 50)
        self.assertEqual(data["seed"], 123)
        self.assertEqual(data["bit_size"], bf._bit_size)
        self.assertEqual(data["hash_count"], bf._hash_count)
        self.assertEqual(len(data["bytes"]), len(bf._bytes))

        # Deserialize from dict
        bf2 = BloomFilter.from_dict(data)

        # Check that the deserialized filter has the same core properties
        self.assertEqual(bf2._expected_items, bf._expected_items)
        self.assertEqual(bf2._false_positive_rate, bf._false_positive_rate)
        self.assertEqual(bf2._bit_size, bf._bit_size)
        self.assertEqual(bf2._hash_count, bf._hash_count)
        self.assertEqual(bf2._seed, bf._seed)
        self.assertEqual(bf2._items_processed, bf._items_processed)
        self.assertEqual(bf2._approximate_count, bf._approximate_count)
        # Compare byte arrays explicitly
        self.assertEqual(bf2._bytes.tobytes(), bf._bytes.tobytes())

        # Check that the deserialized filter contains the same items (no FNs)
        for item in items_added:
            self.assertTrue(bf2.contains(item))

        # Check a non-added item (might be FP, but test consistency)
        non_item = "not-added-item"
        self.assertEqual(bf.contains(non_item), bf2.contains(non_item))

        # Test JSON serialization/deserialization via StreamSummary methods
        json_str = bf.serialize(format="json")
        bf3 = BloomFilter.deserialize(
            json_str, format="json"
        )  # Uses from_dict internally

        # Check consistency again
        self.assertEqual(bf3._bytes.tobytes(), bf._bytes.tobytes())
        for item in items_added:
            self.assertTrue(bf3.contains(item))
        self.assertEqual(bf.contains(non_item), bf3.contains(non_item))

    def test_merge(self):
        """Test merging two Bloom filters."""
        seed = 42
        # Create two filters with the same parameters
        bf1 = BloomFilter(expected_items=100, false_positive_rate=0.01, seed=seed)
        bf2 = BloomFilter(expected_items=100, false_positive_rate=0.01, seed=seed)

        # Add different items to each
        items1 = {f"filter1-{i}" for i in range(50)}
        items2 = {f"filter2-{i}" for i in range(30)}
        common_items = {f"common-{i}" for i in range(10)}

        for item in items1:
            bf1.update(item)
        for item in common_items:
            bf1.update(item)

        for item in items2:
            bf2.update(item)
        for item in common_items:
            bf2.update(item)

        # Merge the filters
        merged = bf1.merge(bf2)

        # Check parameters of merged filter
        self.assertEqual(merged._bit_size, bf1._bit_size)
        self.assertEqual(merged._hash_count, bf1._hash_count)
        self.assertEqual(merged._seed, bf1._seed)

        # Check that the merged filter contains items from both filters
        all_items = items1.union(items2).union(common_items)
        missing_count = 0
        for item in all_items:
            if not merged.contains(item):
                missing_count += 1
                print(f"Merged filter missing item: {item}")
        self.assertEqual(
            missing_count, 0, "Merged filter failed No-False-Negatives check"
        )

        # Check counts
        self.assertEqual(
            merged._items_processed, bf1._items_processed + bf2._items_processed
        )
        # Approximate count is harder to verify exactly after merge

        # Try merging with incompatible parameters (different size/hash count due to different n/p)
        bf3_diff_n = BloomFilter(
            expected_items=200, false_positive_rate=0.01, seed=seed
        )
        bf4_diff_p = BloomFilter(
            expected_items=100, false_positive_rate=0.05, seed=seed
        )
        bf5_diff_seed = BloomFilter(
            expected_items=100, false_positive_rate=0.01, seed=seed + 1
        )

        with self.assertRaises(ValueError, msg="Merge should fail: different bit size"):
            bf1.merge(bf3_diff_n)
        with self.assertRaises(
            ValueError, msg="Merge should fail: different hash count"
        ):
            # Need params that change k but not m significantly, or vice versa
            # Or just rely on the explicit check in merge()
            bf_k_diff = BloomFilter(100, 0.01, seed=seed)
            bf_k_diff._hash_count += 1  # Manually change for test
            bf1.merge(bf_k_diff)

        # Check seed difference if hash depends on it (it does) - merge should probably require same seed
        # The current check doesn't explicitly check seed, but incompatible hashes would likely result
        # if seeds differ, implicitly covered by size/hash count check? Maybe add explicit seed check.
        # Let's assume size/hash check is sufficient proxy for now.

    def test_estimate_cardinality(self):
        """Test cardinality estimation."""
        n = 1000
        p = 0.01
        bf = BloomFilter(expected_items=n, false_positive_rate=p)

        # Test empty filter
        self.assertEqual(bf.estimate_cardinality(), 0)

        # Add a known number of items (well below capacity)
        items_to_add = 500
        for i in range(items_to_add):
            bf.update(f"item-{i}")

        # Get the estimated cardinality
        estimate = bf.estimate_cardinality()
        print(f"\n[Cardinality Test] Added: {items_to_add}, Estimated: {estimate}")

        # Check that it's reasonably close to the true value
        # Allow a margin, e.g., 10-15% for a partially filled filter
        margin = 0.15
        self.assertAlmostEqual(estimate, items_to_add, delta=items_to_add * margin)

        # Test near capacity
        bf_full = BloomFilter(expected_items=n, false_positive_rate=p)
        for i in range(n):
            bf_full.update(f"item-{i}")
        estimate_full = bf_full.estimate_cardinality()
        print(f"[Cardinality Test] Added: {n}, Estimated (full): {estimate_full}")
        # Estimate degrades near capacity, allow larger margin
        self.assertAlmostEqual(estimate_full, n, delta=n * 0.25)

        # Test saturated filter (more items than expected)
        bf_over = BloomFilter(expected_items=n, false_positive_rate=p)
        for i in range(n * 2):
            bf_over.update(f"item-{i}")
        estimate_over = bf_over.estimate_cardinality()
        print(f"[Cardinality Test] Added: {n*2}, Estimated (over): {estimate_over}")
        # Estimate is unreliable when saturated, should cap reasonably (e.g., at items_processed)
        self.assertLessEqual(estimate_over, n * 2)
        # Check it doesn't return infinity or negative
        self.assertGreaterEqual(estimate_over, 0)

    def test_false_positive_probability(self):
        """Test calculation of the current false positive probability estimate."""
        n = 1000
        target_fpp = 0.01
        bf = BloomFilter(expected_items=n, false_positive_rate=target_fpp)

        # Empty filter should have FPP near 0
        self.assertAlmostEqual(bf.false_positive_probability(), 0.0, places=5)

        # Add items and check that FPP increases
        items_added_partial = 200
        for i in range(items_added_partial):
            bf.update(f"item-{i}")

        fpp_partially_filled = bf.false_positive_probability()
        print(
            f"\n[FPP Estimate Test] After {items_added_partial} items: {fpp_partially_filled:.5f}"
        )
        self.assertGreater(fpp_partially_filled, 0)
        # Theoretical FPP after n' items: (1 - (1 - 1/m)^(k*n'))^k ≈ (1 - exp(-k*n'/m))^k
        # Rough check: should be significantly less than target FPP
        self.assertLess(fpp_partially_filled, target_fpp)

        # Add more items (to capacity)
        for i in range(items_added_partial, n):
            bf.update(f"item-{i}")

        fpp_filled = bf.false_positive_probability()
        print(f"[FPP Estimate Test] After {n} items: {fpp_filled:.5f}")
        self.assertGreater(fpp_filled, fpp_partially_filled)
        # Should be close to the target FPP
        self.assertAlmostEqual(
            fpp_filled, target_fpp, delta=target_fpp * 0.5
        )  # Allow 50% margin

    def test_is_empty(self):
        """Test is_empty method."""
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)

        # New filter should be empty
        self.assertTrue(bf.is_empty())

        # Add an item, should not be empty
        bf.update("test")
        self.assertFalse(bf.is_empty())

        # Clear the filter, should be empty again
        bf.clear()
        self.assertTrue(bf.is_empty())
        self.assertEqual(bf._items_processed, 0)
        self.assertEqual(bf._approximate_count, 0)

    def test_clear(self):
        """Test clearing the filter."""
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)

        # Add some items
        items_added = {f"item-{i}" for i in range(50)}
        for item in items_added:
            bf.update(item)

        # Verify items are potentially in the filter and counts updated
        self.assertTrue(bf.contains("item-0"))
        self.assertTrue(bf.contains("item-49"))
        self.assertEqual(bf._items_processed, 50)
        self.assertGreater(bf._approximate_count, 0)
        self.assertFalse(bf.is_empty())

        # Clear the filter
        bf.clear()

        # Verify filter is empty and counts reset
        self.assertTrue(bf.is_empty())
        self.assertEqual(bf._items_processed, 0)
        self.assertEqual(bf._approximate_count, 0)

        # Verify items are not in the filter (no FPs expected right after clear)
        self.assertFalse(bf.contains("item-0"))
        self.assertFalse(bf.contains("item-49"))

    def test_create_from_memory_limit(self):
        """Test creating a filter from memory limit parameters."""
        # Create a filter with a memory limit
        memory_bytes = 1024 * 1  # 1KB
        target_fpp = 0.01

        bf = BloomFilter.create_from_memory_limit(
            memory_bytes=memory_bytes, false_positive_rate=target_fpp
        )

        # Check that the filter's memory usage is close to the limit
        estimated_size = bf.estimate_size()
        print(
            f"\n[Memory Limit Test] Limit: {memory_bytes}, Estimated Size: {estimated_size}"
        )
        # Allow some small slack due to estimation / allocation granularity
        # The fix aims to be <= limit, but python object sizes can fluctuate slightly.
        # Let's allow a small percentage overflow in the test for robustness.
        self.assertLessEqual(
            estimated_size,
            memory_bytes * 1.05,
            "Estimated size exceeds limit significantly",
        )
        # Ideally, it should be <= memory_bytes after the fix. Let's assert that directly.
        self.assertLessEqual(
            estimated_size,
            memory_bytes,
            "Estimated size should not exceed memory limit",
        )

        # The expected_items should be calculated based on the available memory
        # and the false positive rate
        self.assertGreater(bf._expected_items, 0)
        # Verify that the filter created actually targets the requested FPP
        self.assertEqual(bf._false_positive_rate, target_fpp)

        # Try with very small memory limit (should raise ValueError)
        small_limit = 50  # Likely too small for base object + array overhead
        with self.assertRaises(
            ValueError, msg="Should raise ValueError for too small memory limit"
        ):
            BloomFilter.create_from_memory_limit(memory_bytes=small_limit)

        # Test edge case memory limit = 0
        with self.assertRaises(ValueError):
            BloomFilter.create_from_memory_limit(memory_bytes=0)

    def test_different_data_types(self):
        """Test that the filter works with different data types by hashing their string representation."""
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)

        # Add items of different types
        items = [
            "string",
            123,
            3.14,
            (1, 2, 3),
            # Dicts and lists are unhashable by default, but our hash functions
            # work on their string representation via item_bytes = str(item).encode()
            {"key": "value", "num": 1},
            [1, 2, 3, 4],
            True,
            False,
            None,
            b"byte_string",  # Bytes should work directly
        ]

        # Add each item
        for item in items:
            bf.update(item)

        # Check each item (no false negatives allowed)
        for item in items:
            self.assertTrue(
                bf.contains(item), f"Item {item} (type {type(item)}) not found."
            )

        # Check items that are different but might stringify similarly (less likely with good hash)
        self.assertFalse(bf.contains(1234))  # Different number
        self.assertFalse(bf.contains("string "))  # Different string
        # Be cautious testing dicts/lists this way - order matters for str()
        self.assertFalse(bf.contains([1, 2, 3]))  # Different list length


if __name__ == "__main__":
    unittest.main()
