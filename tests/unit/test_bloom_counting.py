"""
Unit tests for Counting Bloom Filter implementation.
"""

import json
import random
import unittest
import sys  # For size estimation checks

# Adjust import path based on your project structure
from tiny_sift.algorithms.bloom.counting import CountingBloomFilter
from tiny_sift.algorithms.bloom.base import BloomFilter  # For type checking merge


class TestCountingBloomFilter(unittest.TestCase):
    """Test cases for Counting Bloom Filter."""

    def test_init(self):
        """Test initialization with valid and invalid parameters."""
        # Valid initialization
        cbf = CountingBloomFilter(
            expected_items=1000, false_positive_rate=0.01, counter_bits=4
        )
        self.assertEqual(cbf._counter_bits, 4)
        self.assertEqual(cbf._counter_max, 15)  # 2^4 - 1
        # Check that underlying size calculation was done
        self.assertGreater(
            cbf._bit_size, 9000
        )  # Should be same num_counters as BF init
        self.assertGreaterEqual(cbf._hash_count, 6)
        # Check that byte array is larger than equivalent BF
        bf_equiv = BloomFilter(expected_items=1000, false_positive_rate=0.01)
        expected_bytes = (cbf._bit_size * cbf._counter_bits + 7) // 8
        self.assertEqual(len(cbf._bytes), expected_bytes)
        self.assertGreater(len(cbf._bytes), len(bf_equiv._bytes))

        # Test different counter bits
        cbf2 = CountingBloomFilter(counter_bits=2)
        self.assertEqual(cbf2._counter_bits, 2)
        self.assertEqual(cbf2._counter_max, 3)  # 2^2 - 1
        expected_bytes_2 = (cbf2._bit_size * 2 + 7) // 8
        self.assertEqual(len(cbf2._bytes), expected_bytes_2)

        cbf8 = CountingBloomFilter(counter_bits=8)
        self.assertEqual(cbf8._counter_bits, 8)
        self.assertEqual(cbf8._counter_max, 255)  # 2^8 - 1
        expected_bytes_8 = (cbf8._bit_size * 8 + 7) // 8
        self.assertEqual(len(cbf8._bytes), expected_bytes_8)

        # --- Invalid parameter tests ---
        # Invalid counter_bits (out of range -> ValueError)
        with self.assertRaises(ValueError):
            CountingBloomFilter(counter_bits=1)  # Below range
        with self.assertRaises(ValueError):
            CountingBloomFilter(counter_bits=9)  # Above range
        with self.assertRaises(ValueError):
            CountingBloomFilter(counter_bits=0)

        # Invalid counter_bits (wrong type -> TypeError)
        with self.assertRaises(TypeError):  # <<< FIX IS HERE
            CountingBloomFilter(counter_bits=3.5)  # Not int
        with self.assertRaises(TypeError):
            CountingBloomFilter(counter_bits="4")  # String not int

        # Invalid base params (should raise ValueError)
        with self.assertRaises(ValueError):
            CountingBloomFilter(expected_items=0)
        with self.assertRaises(ValueError):
            CountingBloomFilter(false_positive_rate=1.1)

    def test_update_and_contains(self):
        """Test adding items and checking for membership."""
        cbf = CountingBloomFilter(expected_items=100, false_positive_rate=0.01)

        # Add some items
        items_to_add = ["apple", "banana", "cherry", 123]
        items_not_added = ["orange", "grape", 456]

        for item in items_to_add:
            cbf.update(item)
            self.assertTrue(cbf.contains(item), f"Item {item} missing after add")

        # Check items that should be in the filter
        for item in items_to_add:
            self.assertTrue(cbf.contains(item), f"Item {item} should be present")

        # Check items that should not be in the filter (might be FP)
        fp_count = 0
        for item in items_not_added:
            if cbf.contains(item):
                fp_count += 1
        # Cannot assert False due to FPs

        # Test adding the same item multiple times
        cbf.update("apple")  # Add again
        self.assertTrue(cbf.contains("apple"), "Item 'apple' missing after second add")
        # Verify counter increased (requires internal check or remove test)
        # Use try-except as _get_bit_positions might return empty if filter size is 0 (unlikely here)
        try:
            initial_pos = cbf._get_bit_positions("apple")[0]  # Get one position
            counter_val = cbf._get_counter(initial_pos)
            self.assertGreaterEqual(
                counter_val, 2, "Counter should be >= 2 after adding twice"
            )
        except IndexError:
            self.fail("Could not get bit positions for 'apple'")

    def test_remove(self):
        """Test removing items."""
        cbf = CountingBloomFilter(
            expected_items=100, false_positive_rate=0.01, counter_bits=4
        )

        # Add an item
        cbf.update("apple")
        self.assertTrue(cbf.contains("apple"))
        pos = cbf._get_bit_positions("apple")
        initial_counts = [cbf._get_counter(p) for p in pos]
        self.assertTrue(all(c > 0 for c in initial_counts))

        # Remove the item
        result = cbf.remove("apple")
        self.assertTrue(result, "remove() should return True for existing item")
        self.assertFalse(
            cbf.contains("apple"), "Item should be gone after single remove"
        )
        final_counts = [cbf._get_counter(p) for p in pos]
        self.assertTrue(
            all(c == 0 for c in final_counts), "Counters should be zero after removal"
        )

        # Try removing an item that wasn't added
        result = cbf.remove("orange")
        self.assertFalse(result, "remove() should return False for non-existent item")
        self.assertFalse(cbf.contains("orange"))  # Should still be false

        # Test multi-add / multi-remove
        cbf.update("banana")
        cbf.update("banana")
        cbf.update("banana")  # Added 3 times
        self.assertTrue(cbf.contains("banana"))
        pos_banana = cbf._get_bit_positions("banana")
        counts_after_3_adds = [cbf._get_counter(p) for p in pos_banana]
        # Use >=3 because collisions could increase count beyond expected
        self.assertTrue(
            all(c >= 3 for c in counts_after_3_adds),
            f"Counters expected >=3, got {counts_after_3_adds}",
        )

        # First removal should keep the item in the filter
        result1 = cbf.remove("banana")
        self.assertTrue(result1)
        self.assertTrue(
            cbf.contains("banana"), "Banana should still be present after 1st remove"
        )
        counts_after_1_remove = [cbf._get_counter(p) for p in pos_banana]
        self.assertTrue(
            all(c >= 2 for c in counts_after_1_remove),
            f"Counters expected >=2, got {counts_after_1_remove}",
        )

        # Second removal should keep the item in the filter
        result2 = cbf.remove("banana")
        self.assertTrue(result2)
        self.assertTrue(
            cbf.contains("banana"), "Banana should still be present after 2nd remove"
        )
        counts_after_2_removes = [cbf._get_counter(p) for p in pos_banana]
        self.assertTrue(
            all(c >= 1 for c in counts_after_2_removes),
            f"Counters expected >=1, got {counts_after_2_removes}",
        )

        # Third removal should remove the item
        result3 = cbf.remove("banana")
        self.assertTrue(result3)
        self.assertFalse(
            cbf.contains("banana"), "Banana should be gone after 3rd remove"
        )
        counts_after_3_removes = [cbf._get_counter(p) for p in pos_banana]
        self.assertTrue(
            all(c == 0 for c in counts_after_3_removes),
            f"Counters expected 0, got {counts_after_3_removes}",
        )

        # Try removing again (should fail and counters stay at 0)
        result4 = cbf.remove("banana")
        self.assertFalse(result4, "Removing already removed item should return False")
        self.assertFalse(cbf.contains("banana"))
        counts_after_4_removes = [cbf._get_counter(p) for p in pos_banana]
        self.assertTrue(
            all(c == 0 for c in counts_after_4_removes),
            f"Counters should remain 0, got {counts_after_4_removes}",
        )

    def test_counter_limits(self):
        """Test that counters cap at the maximum value and don't underflow."""
        # Use a small counter_bits value for easier testing
        counter_bits = 2
        max_val = (1 << counter_bits) - 1  # Should be 3
        cbf = CountingBloomFilter(
            expected_items=100, false_positive_rate=0.01, counter_bits=counter_bits
        )
        self.assertEqual(cbf._counter_max, max_val)

        item = "test_saturation"
        positions = cbf._get_bit_positions(item)

        # Add an item multiple times, exceeding counter max
        num_adds = max_val + 5  # Add more times than max_val
        for _ in range(num_adds):
            cbf.update(item)

        # Check that the item is still in the filter
        self.assertTrue(cbf.contains(item))
        # Check that counters are capped at max_val
        counts_after_adds = [cbf._get_counter(p) for p in positions]
        # Allow for >= max_val due to potential collisions, but should mostly be == max_val
        # Check that *at least one* counter hit max_val (stronger check is hard without knowing exact positions)
        # A better check: all counters must be exactly max_val *if there were no collisions*
        # Let's stick to checking they are ALL equal to max_val, assuming low collision for test item
        self.assertTrue(
            all(c == max_val for c in counts_after_adds),
            f"Counters should be capped at {max_val}, got {counts_after_adds}",
        )

        # Remove the item exactly max_val times
        for i in range(max_val):
            self.assertTrue(
                cbf.contains(item), f"Item should be present before remove #{i+1}"
            )
            remove_result = cbf.remove(item)
            self.assertTrue(remove_result, f"Remove #{i+1} should return True")
            counts_after_remove = [cbf._get_counter(p) for p in positions]
            expected_count = max_val - (i + 1)
            # Check if *all* counters match expected value (this assumes no collisions affecting counters)
            # A safer check is that all counters are >= expected_count for decrement
            # Let's check equality assuming no interference for this specific test item
            self.assertTrue(
                all(c == expected_count for c in counts_after_remove),
                f"Counters expected == {expected_count} after remove #{i+1}, got {counts_after_remove}",
            )

        # After max_val removals, item should be gone (counters at 0)
        self.assertFalse(
            cbf.contains(item), "Item should be removed after max_val removes"
        )
        final_counts = [cbf._get_counter(p) for p in positions]
        self.assertTrue(
            all(c == 0 for c in final_counts),
            f"Counters should be 0, got {final_counts}",
        )

        # Further removals should fail and counters stay at 0
        remove_result_extra = cbf.remove(item)
        self.assertFalse(remove_result_extra, "Removing again should fail")
        self.assertFalse(cbf.contains(item))
        extra_counts = [cbf._get_counter(p) for p in positions]
        self.assertTrue(
            all(c == 0 for c in extra_counts),
            f"Counters should remain 0, got {extra_counts}",
        )

    def test_serialization(self):
        """Test serialization and deserialization."""
        cbf = CountingBloomFilter(
            expected_items=100, false_positive_rate=0.01, counter_bits=4, seed=456
        )

        # Add some items, including duplicates
        items_added = [
            f"item-{i // 2}" for i in range(100)
        ]  # Add 50 unique items twice
        for item in items_added:
            cbf.update(item)

        # Serialize to dict
        data = cbf.to_dict()

        # Check dict contents specific to CBF
        self.assertEqual(data["type"], "CountingBloomFilter")
        self.assertEqual(data["counter_bits"], 4)
        self.assertEqual(data["counter_max"], 15)
        self.assertEqual(data["seed"], 456)
        # Check base class fields too
        self.assertEqual(data["expected_items"], 100)
        self.assertEqual(data["false_positive_rate"], 0.01)
        self.assertEqual(data["items_processed"], 100)

        # Deserialize from dict
        cbf2 = CountingBloomFilter.from_dict(data)

        # Check that the deserialized filter has the same core properties
        self.assertEqual(cbf2._counter_bits, cbf._counter_bits)
        self.assertEqual(cbf2._counter_max, cbf._counter_max)
        self.assertEqual(cbf2._seed, cbf._seed)
        self.assertEqual(cbf2._expected_items, cbf._expected_items)
        self.assertEqual(cbf2._false_positive_rate, cbf._false_positive_rate)
        self.assertEqual(cbf2._bit_size, cbf._bit_size)
        self.assertEqual(cbf2._hash_count, cbf._hash_count)
        self.assertEqual(cbf2._items_processed, cbf._items_processed)
        self.assertEqual(cbf2._approximate_count, cbf._approximate_count)
        # Compare byte arrays
        self.assertEqual(cbf2._bytes.tobytes(), cbf._bytes.tobytes())

        # Check functional equivalence (contains)
        unique_items = {f"item-{i}" for i in range(50)}
        for item in unique_items:
            self.assertTrue(
                cbf2.contains(item), f"Deserialized filter missing item {item}"
            )

        # Remove items from both and compare
        item_to_remove = "item-10"
        self.assertTrue(cbf.contains(item_to_remove))
        self.assertTrue(cbf2.contains(item_to_remove))

        cbf.remove(item_to_remove)  # Remove once (was added twice)
        cbf2.remove(item_to_remove)
        self.assertTrue(cbf.contains(item_to_remove))
        self.assertTrue(cbf2.contains(item_to_remove))
        self.assertEqual(
            cbf._bytes.tobytes(),
            cbf2._bytes.tobytes(),
            "Byte arrays differ after first remove",
        )

        cbf.remove(item_to_remove)  # Remove second time
        cbf2.remove(item_to_remove)
        self.assertFalse(cbf.contains(item_to_remove))
        self.assertFalse(cbf2.contains(item_to_remove))
        self.assertEqual(
            cbf._bytes.tobytes(),
            cbf2._bytes.tobytes(),
            "Byte arrays differ after second remove",
        )

        # Test JSON serialization via StreamSummary methods
        json_str = cbf.serialize(format="json")
        cbf3 = CountingBloomFilter.deserialize(json_str, format="json")

        # Check consistency again
        self.assertEqual(cbf3._bytes.tobytes(), cbf._bytes.tobytes())
        self.assertFalse(cbf3.contains(item_to_remove))

    def test_merge(self):
        """Test merging two Counting Bloom filters."""
        seed = 42
        counter_bits = 4
        max_val = (1 << counter_bits) - 1
        # Create two filters with the same parameters
        cbf1 = CountingBloomFilter(
            expected_items=100,
            false_positive_rate=0.01,
            counter_bits=counter_bits,
            seed=seed,
        )
        cbf2 = CountingBloomFilter(
            expected_items=100,
            false_positive_rate=0.01,
            counter_bits=counter_bits,
            seed=seed,
        )

        # Add different items to each, and some common items with different counts
        items1_only = {f"filter1-{i}" for i in range(30)}
        items2_only = {f"filter2-{i}" for i in range(20)}
        common_items = {f"common-{i}" for i in range(10)}
        saturation_item = "saturate"

        for item in items1_only:
            cbf1.update(item)
        for item in common_items:
            cbf1.update(item)
            cbf1.update(item)  # Add common twice to cbf1
        cbf1.update(saturation_item)  # Add once

        for item in items2_only:
            cbf2.update(item)
        for item in common_items:
            cbf2.update(item)  # Add common once to cbf2
        # Add saturation item enough times to hit max in cbf2
        for _ in range(max_val):
            cbf2.update(saturation_item)

        # Merge the filters
        merged = cbf1.merge(cbf2)

        # Check parameters of merged filter
        self.assertEqual(merged._bit_size, cbf1._bit_size)
        self.assertEqual(merged._hash_count, cbf1._hash_count)
        self.assertEqual(merged._counter_bits, cbf1._counter_bits)
        self.assertEqual(merged._seed, cbf1._seed)

        # Check that the merged filter contains items from both filters
        all_items = (
            items1_only.union(items2_only).union(common_items).union({saturation_item})
        )
        missing_count = 0
        for item in all_items:
            if not merged.contains(item):
                missing_count += 1
                print(f"Merged filter missing item: {item}")
        self.assertEqual(
            missing_count, 0, "Merged filter failed No-False-Negatives check"
        )

        # Check counter values for specific items
        # Common items: added 2 times in cbf1, 1 time in cbf2 -> merged should have count >= 3
        pos_common = merged._get_bit_positions("common-0")
        counts_common = [merged._get_counter(p) for p in pos_common]
        self.assertTrue(
            all(c >= 3 for c in counts_common),
            f"Common item counts should be >= 3, got {counts_common}",
        )

        # Saturation item: added 1 time in cbf1, max_val times in cbf2 -> merged should be capped at max_val
        pos_saturate = merged._get_bit_positions(saturation_item)
        counts_saturate = [merged._get_counter(p) for p in pos_saturate]
        self.assertTrue(
            all(c == max_val for c in counts_saturate),
            f"Saturation item counts should be capped at {max_val}, got {counts_saturate}",
        )

        # Check counts
        self.assertEqual(
            merged._items_processed, cbf1._items_processed + cbf2._items_processed
        )

        # --- Test Incompatible Merges ---
        cbf3_diff_bits = CountingBloomFilter(counter_bits=counter_bits - 1, seed=seed)
        cbf4_diff_size = CountingBloomFilter(
            expected_items=200, counter_bits=counter_bits, seed=seed
        )
        cbf5_diff_seed = CountingBloomFilter(counter_bits=counter_bits, seed=seed + 1)
        bf_plain = BloomFilter(expected_items=100, false_positive_rate=0.01, seed=seed)

        with self.assertRaises(
            ValueError, msg="Merge should fail: different counter_bits"
        ):
            cbf1.merge(cbf3_diff_bits)
        # Check if expected_items difference leads to size difference before asserting
        if (
            cbf1._bit_size != cbf4_diff_size._bit_size
            or cbf1._hash_count != cbf4_diff_size._hash_count
        ):
            with self.assertRaises(
                ValueError, msg="Merge should fail: different bit_size or hash_count"
            ):
                cbf1.merge(cbf4_diff_size)
        else:
            print("\nSkipping merge size mismatch test, params yielded same size/hash")

        # Check seed difference
        if (
            cbf1._bit_size != cbf5_diff_seed._bit_size
            or cbf1._hash_count != cbf5_diff_seed._hash_count
            or cbf1._seed != cbf5_diff_seed._seed
        ):
            with self.assertRaises(
                ValueError, msg="Merge should fail: different seed or derived params"
            ):
                cbf1.merge(cbf5_diff_seed)
        else:
            # This case shouldn't happen if seeds differ, but check anyway
            print("\nSkipping merge seed mismatch test, params yielded same config")

        with self.assertRaises(TypeError, msg="Merge should fail: different types"):
            cbf1.merge(bf_plain)  # Merging CBF with BF

    def test_clear(self):
        """Test clearing the filter."""
        cbf = CountingBloomFilter(
            expected_items=100, false_positive_rate=0.01, counter_bits=3
        )

        # Add some items
        items = {f"item-{i}" for i in range(50)}
        for item in items:
            cbf.update(item)
            cbf.update(item)  # Add twice

        # Verify items are in the filter and state is non-empty
        self.assertTrue(cbf.contains("item-10"))
        self.assertFalse(cbf.is_empty())  # is_empty checks if all bytes are 0
        self.assertEqual(cbf._items_processed, 100)
        self.assertGreater(cbf._approximate_count, 0)

        # Clear the filter
        cbf.clear()

        # Verify filter is empty and counts reset
        self.assertTrue(cbf.is_empty())
        self.assertEqual(cbf._items_processed, 0)
        self.assertEqual(cbf._approximate_count, 0)

        # Verify items are not in the filter
        self.assertFalse(cbf.contains("item-10"))

        # Check that internal counters are zero
        # Pick a random position to check
        if cbf._bit_size > 0:
            # Ensure random range is valid
            pos_to_check = random.randint(0, max(0, cbf._bit_size - 1))
            self.assertEqual(
                cbf._get_counter(pos_to_check), 0, "Counter not zero after clear"
            )

    def test_counter_operations_optimized(self):
        """Test the optimized counter get/set operations thoroughly."""
        # Use class attribute directly now
        for (
            bits
        ) in CountingBloomFilter.SUPPORTED_COUNTER_BITS:  # Test all supported sizes
            max_val = (1 << bits) - 1
            # Use minimal capacity for easier position calculation if needed
            cbf = CountingBloomFilter(expected_items=10, counter_bits=bits, seed=bits)
            num_counters = cbf._bit_size
            array_len = len(cbf._bytes)

            # Skip if filter becomes too small
            if num_counters <= 0:
                continue

            print(
                f"\n[Counter Ops Test] bits={bits}, max_val={max_val}, num_counters={num_counters}, array_len={array_len}"
            )

            # Ensure test positions are within the valid range [0, num_counters - 1]
            test_positions = [0]
            if num_counters > 1:
                test_positions.append(1)
            if num_counters > 2:
                test_positions.append(num_counters // 2)
            if num_counters > 1:
                test_positions.append(num_counters - 1)

            # Add positions near byte boundaries if possible
            if num_counters > 10:
                bits_per_byte = 8
                counters_per_byte_approx = bits_per_byte / bits
                # Calculate potential boundary position (e.g., end of first byte)
                boundary_pos = int(counters_per_byte_approx)
                # Ensure it's valid and not already included
                if (
                    0 < boundary_pos < num_counters
                    and boundary_pos not in test_positions
                ):
                    test_positions.append(boundary_pos)
                # Example: second boundary
                boundary_pos2 = int(counters_per_byte_approx * 2)
                if (
                    boundary_pos < boundary_pos2 < num_counters
                    and boundary_pos2 not in test_positions
                ):
                    test_positions.append(boundary_pos2)

            # Ensure positions are unique and sorted for clarity
            test_positions = sorted(list(set(test_positions)))

            test_values = [0, 1, max_val // 2, max_val - 1, max_val]
            # Ensure test values are valid (remove duplicates like max_val=1)
            test_values = sorted(list(set(v for v in test_values if 0 <= v <= max_val)))

            for pos in test_positions:
                # This check should be redundant now, but keep for safety
                if not (0 <= pos < num_counters):
                    continue

                # 1. Test initial state (should be 0)
                self.assertEqual(
                    cbf._get_counter(pos),
                    0,
                    f"Initial get failed: bits={bits}, pos={pos}",
                )

                for val in test_values:
                    # 2. Test set and get
                    cbf._set_counter(pos, val)
                    read_val = cbf._get_counter(pos)
                    self.assertEqual(
                        read_val,
                        val,
                        f"Set/Get failed: bits={bits}, pos={pos}, set={val}, got={read_val}",
                    )

                    # 3. Test increment (unless at max)
                    if val < max_val:
                        cbf._set_counter(pos, val)  # Reset before increment
                        cbf._increment_counter(pos)
                        read_inc = cbf._get_counter(pos)
                        self.assertEqual(
                            read_inc,
                            val + 1,
                            f"Increment failed: bits={bits}, pos={pos}, start={val}, got={read_inc}",
                        )
                        # Decrement back to test decrement
                        cbf._decrement_counter(pos)
                        read_dec_back = cbf._get_counter(pos)
                        self.assertEqual(
                            read_dec_back,
                            val,
                            f"Decrement back failed: bits={bits}, pos={pos}, start={val+1}, got={read_dec_back}",
                        )
                    else:  # If val == max_val
                        # 4. Test increment at max (should cap)
                        cbf._set_counter(pos, max_val)
                        cbf._increment_counter(pos)
                        read_inc_max = cbf._get_counter(pos)
                        self.assertEqual(
                            read_inc_max,
                            max_val,
                            f"Increment@max failed: bits={bits}, pos={pos}, got={read_inc_max}",
                        )

                    # 5. Test decrement (unless at 0)
                    if val > 0:
                        cbf._set_counter(pos, val)  # Reset state before decrement
                        cbf._decrement_counter(pos)
                        read_dec = cbf._get_counter(pos)
                        self.assertEqual(
                            read_dec,
                            val - 1,
                            f"Decrement failed: bits={bits}, pos={pos}, start={val}, got={read_dec}",
                        )
                    else:  # If val == 0
                        # 6. Test decrement at 0 (should stay 0)
                        cbf._set_counter(pos, 0)
                        cbf._decrement_counter(pos)
                        read_dec_zero = cbf._get_counter(pos)
                        self.assertEqual(
                            read_dec_zero,
                            0,
                            f"Decrement@zero failed: bits={bits}, pos={pos}, got={read_dec_zero}",
                        )

                # Reset position to 0 for next position test
                cbf._set_counter(pos, 0)

    # Optional: Test cardinality and FPP estimates (know they are inaccurate for CBF)
    def test_cbf_estimates_inherited(self):
        """Check that inherited estimate methods run without error."""
        cbf = CountingBloomFilter(expected_items=100, counter_bits=4)
        cbf.update("test1")
        cbf.update("test2")
        try:
            card = cbf.estimate_cardinality()
            self.assertIsInstance(card, int)
            fpp = cbf.false_positive_probability()
            self.assertIsInstance(fpp, float)
            size = cbf.estimate_size()
            self.assertIsInstance(size, int)
            # No assertions on accuracy, just that they run.
        except Exception as e:
            self.fail(f"Inherited estimate method failed: {e}")


if __name__ == "__main__":
    unittest.main()
