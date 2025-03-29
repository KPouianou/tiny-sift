"""
Unit tests for HyperLogLog algorithm.
"""

import json
import math
import random
import unittest
from collections import Counter
from typing import List, Set

from tiny_sift.algorithms.hyperloglog import HyperLogLog


class TestHyperLogLog(unittest.TestCase):
    """Test cases for HyperLogLog cardinality estimator."""

    def test_init(self):
        """Test initialization with valid and invalid parameters."""
        # Valid initialization
        hll = HyperLogLog(precision=12)
        self.assertEqual(hll._precision, 12)
        self.assertEqual(hll._m, 4096)  # 2^12
        self.assertEqual(len(hll._registers), 4096)

        # Test minimum precision
        hll = HyperLogLog(precision=4)
        self.assertEqual(hll._precision, 4)
        self.assertEqual(hll._m, 16)  # 2^4

        # Test maximum precision
        hll = HyperLogLog(precision=16)
        self.assertEqual(hll._precision, 16)
        self.assertEqual(hll._m, 65536)  # 2^16

        # Invalid precision - too low
        with self.assertRaises(ValueError):
            HyperLogLog(precision=3)

        # Invalid precision - too high
        with self.assertRaises(ValueError):
            HyperLogLog(precision=17)

    def test_update_and_estimate_small(self):
        """Test updating with a small stream and estimating cardinality."""
        # Use precision 8 for faster tests
        hll = HyperLogLog(precision=8, seed=42)

        # Add 100 unique items
        unique_items = set()
        for i in range(100):
            item = f"item-{i}"
            unique_items.add(item)
            hll.update(item)

        # Check estimate is reasonably close
        estimate = hll.estimate_cardinality()
        true_count = len(unique_items)

        # Calculate relative error
        rel_error = abs(estimate - true_count) / true_count

        # The error should be within theoretical bounds for 68% of test runs
        # For p=8, the standard error is ~1.04/sqrt(2^8) ≈ 7.3%
        # We'll allow a bit more leeway for the test to be reliable
        self.assertLessEqual(
            rel_error,
            0.15,
            f"Estimate {estimate} too far from true count {true_count}, "
            f"relative error {rel_error:.2%}",
        )

    def test_update_with_duplicates(self):
        """Test that duplicate items don't affect the cardinality estimate."""
        hll1 = HyperLogLog(precision=10, seed=42)
        hll2 = HyperLogLog(precision=10, seed=42)

        # Add 1000 unique items to both estimators
        for i in range(1000):
            item = f"unique-{i}"
            hll1.update(item)
            hll2.update(item)

        # Add the same 1000 items again to hll2 (duplicates)
        for i in range(1000):
            item = f"unique-{i}"
            hll2.update(item)

        # Add more duplicates to hll2 (repeat 10 times)
        for _ in range(10):
            for i in range(100):
                item = f"unique-{i}"
                hll2.update(item)

        # Estimates should be close despite the duplicates
        estimate1 = hll1.estimate_cardinality()
        estimate2 = hll2.estimate_cardinality()

        # Calculate relative difference
        rel_diff = abs(estimate2 - estimate1) / estimate1

        # The difference should be minimal since they contain the same unique items
        self.assertLessEqual(
            rel_diff,
            0.01,
            f"Estimates differ too much: {estimate1} vs {estimate2}, "
            f"relative difference {rel_diff:.2%}",
        )

    def test_large_cardinality(self):
        """Test with a larger number of unique items."""
        # Use precision 12 for better accuracy
        hll = HyperLogLog(precision=12, seed=42)

        # Add 100,000 unique items
        n = 100000
        for i in range(n):
            hll.update(f"item-{i}")

        # Check estimate
        estimate = hll.estimate_cardinality()

        # Calculate relative error
        rel_error = abs(estimate - n) / n

        # For p=12, the standard error is ~1.04/sqrt(2^12) ≈ 1.62%
        # We'll use 5% as a conservative bound for the test
        self.assertLessEqual(
            rel_error,
            0.05,
            f"Estimate {estimate} too far from true count {n}, "
            f"relative error {rel_error:.2%}",
        )

    def test_statistical_properties(self):
        """Test the statistical properties with multiple runs."""
        # We'll run multiple trials and check the distribution of errors
        precision = 10  # Smaller for faster tests
        trials = 20
        n = 10000  # Number of unique items

        rel_errors = []

        for trial in range(trials):
            hll = HyperLogLog(precision=precision, seed=trial)

            # Add n unique items
            for i in range(n):
                hll.update(f"trial-{trial}-item-{i}")

            estimate = hll.estimate_cardinality()
            rel_error = (estimate - n) / n  # Can be positive or negative
            rel_errors.append(rel_error)

        # Calculate statistics on the relative errors
        abs_errors = [abs(e) for e in rel_errors]
        mean_abs_error = sum(abs_errors) / len(abs_errors)

        # The expected standard error for this precision
        expected_std_error = 1.04 / (2 ** (precision / 2))

        # The mean absolute error should be close to the expected standard error
        # We'll use a generous bound for the test to be reliable
        self.assertLessEqual(
            mean_abs_error,
            expected_std_error * 1.5,
            f"Mean absolute error {mean_abs_error:.4f} too high "
            f"compared to expected {expected_std_error:.4f}",
        )

        # Check bias - the mean error should be close to zero
        mean_error = sum(rel_errors) / len(rel_errors)
        self.assertLessEqual(
            abs(mean_error), 0.02, f"Estimator shows bias: mean error {mean_error:.4f}"
        )

    def test_empty_estimate(self):
        """Test cardinality estimate for an empty estimator."""
        hll = HyperLogLog(precision=10)

        # Empty estimator should return 0
        self.assertEqual(hll.estimate_cardinality(), 0)

    def test_small_cardinality_correction(self):
        """Test the small cardinality correction (linear counting)."""
        # When many registers are empty, the algorithm should switch to linear counting
        hll = HyperLogLog(
            precision=14, seed=42
        )  # Large precision to have many empty registers

        # Add just a few unique items
        for i in range(10):
            hll.update(f"item-{i}")

        # Get the estimate
        estimate = hll.estimate_cardinality()

        # The estimate should be reasonably close to 10
        # With p=14, linear counting should give excellent results for small sets
        self.assertGreaterEqual(estimate, 8)
        self.assertLessEqual(estimate, 12)

    def test_serialization(self):
        """Test serialization and deserialization."""
        hll = HyperLogLog(precision=8, seed=42)

        # Add some items
        for i in range(1000):
            hll.update(f"item-{i}")

        # Serialize to dict
        data = hll.to_dict()

        # Check dict contents
        self.assertEqual(data["type"], "HyperLogLog")
        self.assertEqual(data["precision"], 8)
        self.assertEqual(data["items_processed"], 1000)
        self.assertEqual(len(data["registers"]), 256)  # 2^8

        # Deserialize from dict
        hll2 = HyperLogLog.from_dict(data)

        # Check that the deserialized object matches the original
        self.assertEqual(hll2._precision, hll._precision)
        self.assertEqual(hll2._items_processed, hll._items_processed)
        self.assertEqual(hll2._registers, hll._registers)
        self.assertEqual(hll2.estimate_cardinality(), hll.estimate_cardinality())

        # Test JSON serialization
        json_str = hll.serialize(format="json")
        hll3 = HyperLogLog.deserialize(json_str, format="json")

        # Check that the deserialized object matches the original
        self.assertEqual(hll3.estimate_cardinality(), hll.estimate_cardinality())

    def test_merge(self):
        """Test merging two HyperLogLog estimators."""
        # Create two estimators with different data
        hll1 = HyperLogLog(precision=10, seed=42)
        for i in range(1000):
            hll1.update(f"set1-{i}")

        hll2 = HyperLogLog(precision=10, seed=42)
        for i in range(500, 1500):
            hll2.update(f"set1-{i}")  # 500 items overlap with hll1

        # Merge the estimators
        merged = hll1.merge(hll2)

        # The merged estimator should estimate the cardinality of the union
        estimate = merged.estimate_cardinality()
        true_count = 1500  # 0-999 from hll1 + 500-1499 from hll2 = 0-1499

        # Calculate relative error
        rel_error = abs(estimate - true_count) / true_count

        # For p=10, the standard error is ~1.04/sqrt(2^10) ≈ 3.25%
        # We'll use 10% as a conservative bound for the test
        self.assertLessEqual(
            rel_error,
            0.1,
            f"Merged estimate {estimate} too far from true count {true_count}, "
            f"relative error {rel_error:.2%}",
        )

        # Test merging with incompatible precision
        hll3 = HyperLogLog(precision=12)
        with self.assertRaises(ValueError):
            hll1.merge(hll3)

    def test_create_from_error_rate(self):
        """Test creating a sketch from error rate parameters."""
        # Create an estimator with specific error rate
        rel_error = 0.01  # 1% error

        hll = HyperLogLog.create_from_error_rate(rel_error, seed=42)

        # Check precision is calculated correctly
        # For 1% error, precision should be ~14 (1.04/sqrt(2^p) = 0.01 => p ≈ 13.7)
        self.assertEqual(hll._precision, 14)

        # Test with very small error (too small for p<=16)
        with self.assertRaises(ValueError):
            HyperLogLog.create_from_error_rate(0.001)  # 0.1% error would need p=18

        # Test with invalid error rate
        with self.assertRaises(ValueError):
            HyperLogLog.create_from_error_rate(-0.01)

        with self.assertRaises(ValueError):
            HyperLogLog.create_from_error_rate(1.5)

    def test_create_from_memory_limit(self):
        """Test creating an estimator from memory constraints."""
        # Create an estimator with a memory limit
        memory_bytes = 8192  # 8KB

        hll = HyperLogLog.create_from_memory_limit(memory_bytes, seed=42)

        # Check that the memory usage is within the limit
        self.assertLessEqual(hll.estimate_size(), memory_bytes)

        # Check that we got the highest precision that fits
        # An 8KB memory limit should fit at least p=12 (4KB for registers)
        self.assertGreaterEqual(hll._precision, 12)

        # Test with too small memory limit
        with self.assertRaises(ValueError):
            HyperLogLog.create_from_memory_limit(10)  # Too small

    def test_clear(self):
        """Test clearing the estimator."""
        hll = HyperLogLog(precision=8, seed=42)

        # Add some items
        for i in range(1000):
            hll.update(f"item-{i}")

        # Verify non-zero estimate
        self.assertGreater(hll.estimate_cardinality(), 0)

        # Clear the estimator
        hll.clear()

        # Verify registers are cleared
        for register in hll._registers:
            self.assertEqual(register, 0)

        # Verify estimate is reset to 0
        self.assertEqual(hll.estimate_cardinality(), 0)
        self.assertEqual(hll._items_processed, 0)
        self.assertTrue(hll._is_empty)

    def test_error_bound(self):
        """Test the error bound calculations."""
        # Test with different precision values
        precisions = [4, 8, 12, 16]

        for p in precisions:
            hll = HyperLogLog(precision=p)
            bounds = hll.error_bound()

            # Check expected standard error: 1.04/sqrt(2^p)
            expected_std_error = 1.04 / math.sqrt(2**p)
            self.assertAlmostEqual(
                bounds["relative_error"], expected_std_error, places=10
            )

            # Check confidence intervals
            self.assertAlmostEqual(
                bounds["confidence_68pct"], expected_std_error, places=10
            )
            self.assertAlmostEqual(
                bounds["confidence_95pct"], expected_std_error * 1.96, places=10
            )
            self.assertAlmostEqual(
                bounds["confidence_99pct"], expected_std_error * 2.58, places=10
            )

    def test_accuracy_with_increasing_precision(self):
        """Test how accuracy improves with precision."""
        # Create datasets with different cardinalities
        cardinalities = [100, 1000, 10000]
        precisions = [6, 8, 10, 12]  # Test different precision values

        # Instead of checking if errors decrease with precision for a single seed,
        # let's run multiple trials with different seeds and check if the *average*
        # errors decrease, which is what theory predicts
        num_trials = 5

        for n in cardinalities:
            # Generate a dataset with n unique items
            dataset = [f"item-{i}" for i in range(n)]

            # Track average errors across trials for each precision
            avg_errors = [0.0] * len(precisions)

            # Run multiple trials with different seeds
            for trial in range(num_trials):
                for i, p in enumerate(precisions):
                    # Use a different seed for each trial
                    hll = HyperLogLog(precision=p, seed=42 + trial)

                    # Process the dataset
                    for item in dataset:
                        hll.update(item)

                    # Measure relative error
                    estimate = hll.estimate_cardinality()
                    rel_error = abs(estimate - n) / n

                    # Accumulate error for averaging later
                    avg_errors[i] += rel_error / num_trials

            # Calculate theoretical error bounds for each precision
            theoretical_errors = [1.04 / math.sqrt(2**p) for p in precisions]

            # Now check if error generally decreases with precision
            # We'll check if higher precisions have lower average error
            half_idx = len(precisions) // 2
            avg_error_first_half = sum(avg_errors[:half_idx]) / half_idx
            avg_error_second_half = sum(avg_errors[half_idx:]) / (
                len(precisions) - half_idx
            )

            self.assertLessEqual(
                avg_error_second_half,
                avg_error_first_half * 1.2,  # Allow some margin
                f"Average error did not generally decrease with precision: {avg_error_first_half:.4f} -> {avg_error_second_half:.4f}",
            )

            # Also check that errors are roughly within theoretical bounds
            # Allow for a factor of 3 over theoretical error (common in practice)
            for i, p in enumerate(precisions):
                self.assertLessEqual(
                    avg_errors[i],
                    theoretical_errors[i] * 3.0,
                    f"Error at precision {p} exceeds 3x theoretical bound: {avg_errors[i]:.4f} vs {theoretical_errors[i]:.4f}",
                )

    def test_hash_distribution(self):
        """Test that hash values are well distributed across registers."""
        hll = HyperLogLog(precision=8, seed=42)  # 256 registers

        # Add a large number of items
        n = 100000
        for i in range(n):
            hll.update(f"item-{i}")

        # Get register values
        register_values = hll.get_register_values()

        # Check that most registers have non-zero values
        non_zero = sum(1 for v in register_values if v > 0)
        self.assertGreater(
            non_zero,
            0.9 * len(register_values),
            f"Too many empty registers: {len(register_values) - non_zero}",
        )

        # Check distribution of register values
        # For a well-distributed hash function, the distribution should follow
        # a geometric distribution with values roughly between 0 and 20 for 32-bit hashes
        counter = Counter(register_values)
        max_value = max(counter.keys())

        # Register values shouldn't exceed ~30 for 32-bit hash and p=8
        self.assertLessEqual(
            max_value, 30, f"Maximum register value {max_value} is suspiciously high"
        )

    def test_get_stats(self):
        """Test the get_stats method."""
        hll = HyperLogLog(precision=10, seed=42)

        # Add some items
        for i in range(1000):
            hll.update(f"item-{i}")

        # Get stats
        stats = hll.get_stats()

        # Check stats
        self.assertEqual(stats["precision"], 10)
        self.assertEqual(stats["num_registers"], 1024)
        self.assertLessEqual(stats["empty_registers"], 1024)
        self.assertGreaterEqual(stats["max_register_value"], 1)
        self.assertGreater(stats["estimate"], 0)
        self.assertAlmostEqual(
            stats["relative_error"], 1.04 / math.sqrt(1024), places=10
        )
        self.assertGreater(stats["memory_usage"], 0)

    def test_different_data_types(self):
        """Test that HyperLogLog works with different data types."""
        hll = HyperLogLog(precision=10, seed=42)

        # Add items of different types
        items = [
            "string",
            123,
            3.14,
            (1, 2, 3),
            {"key": "value"},
            [1, 2, 3],
            True,
            None,
        ]

        # Update with each item
        for item in items:
            hll.update(item)

        # Check cardinality estimate
        estimate = hll.estimate_cardinality()

        # Should be approximately the number of unique items
        self.assertGreaterEqual(estimate, len(items) * 0.8)
        self.assertLessEqual(estimate, len(items) * 1.2)


if __name__ == "__main__":
    unittest.main()
