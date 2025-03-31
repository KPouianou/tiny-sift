"""
Basic Bloom Filter Demo for TinySift.

This example demonstrates how to use the standard Bloom Filter
for space-efficient set membership testing. It highlights its probabilistic
nature (false positives) and its guarantee of no false negatives.
"""

import random
import sys
import time
import math  # Needed for theoretical FPP calculation if desired

# Adjust import path based on your project structure
from tiny_sift.algorithms.bloom.base import BloomFilter


def demonstrate_basic_usage():
    """Demonstrate Bloom Filter initialization, adding, and checking."""
    print("\n=== Basic Bloom Filter Demo ===")

    # Create a Bloom filter
    # Expecting ~10,000 items with a 1% (0.01) false positive rate
    bf = BloomFilter(expected_items=10000, false_positive_rate=0.01, seed=42)

    print(f"Bloom Filter parameters:")
    print(f"  Expected items: {bf._expected_items:,}")
    print(f"  Target false positive rate: {bf._false_positive_rate:.1%}")
    print(f"  Calculated filter size (bits): {bf._bit_size:,} bits")
    print(f"  Calculated number of hashes: {bf._hash_count}")
    print(f"  Estimated memory usage: {bf.estimate_size():,} bytes")

    # Add some items
    print("\nAdding items to the filter...")
    items_to_add = [
        "apple",
        "banana",
        "cherry",
        "date",
        "fig",
        "grape",
        "kiwi",
        "lemon",
        "mango",
        "nectarine",
    ]
    for item in items_to_add:
        bf.update(item)
        print(f"  Added '{item}'")

    print(
        f"\nFilter state: {bf._items_processed} items processed, approx {bf._approximate_count} unique items added."
    )

    # Check membership
    print("\nChecking membership:")
    print("  (Note: 'False' means DEFINITELY NOT present)")
    print("  (Note: 'True' means POSSIBLY present - could be a false positive)")
    items_to_test = items_to_add + [
        "orange",
        "pear",
        "plum",
        "lime",
        "papaya",
    ]  # Includes added and non-added

    for item in items_to_test:
        is_present = bf.contains(item)
        print(f"  '{item}' in filter? {is_present}")

    # Demonstrate NO False Negatives
    print("\nVerifying NO False Negatives:")
    missing_count = 0
    for item in items_to_add:
        if not bf.contains(item):
            missing_count += 1
            print(f"  ERROR: False negative for added item '{item}'!")
    if missing_count == 0:
        print("  Confirmed: All items originally added were found.")
    else:
        print(f"  ERROR: {missing_count} added items were NOT found!")

    # Demonstrate False Positives
    print("\nChecking for False Positives:")
    items_not_added = [
        "orange",
        "pear",
        "plum",
        "lime",
        "papaya",
        "grapefruit",
        "apricot",
        "peach",
        "blueberry",
        "raspberry",
    ]
    fp_count = 0
    for item in items_not_added:
        if bf.contains(item):
            fp_count += 1
            print(f"  False Positive: '{item}' (was not added, but contains() is True)")

    if fp_count == 0:
        print("  No false positives detected among tested items.")
    else:
        print(
            f"  Detected {fp_count} false positive(s) among {len(items_not_added)} non-added items tested."
        )
        # Note: The actual FPP depends on how full the filter is.


def demonstrate_fpp_and_fill_ratio():
    """Show how fill ratio affects the actual False Positive Probability."""
    print("\n=== FPP vs. Fill Ratio Demo ===")

    n = 1000
    target_fpp = 0.05  # Use a slightly higher FPP for easier demo
    bf = BloomFilter(expected_items=n, false_positive_rate=target_fpp, seed=123)

    print(f"Filter initialized for {n} items, target FPP: {target_fpp:.1%}")
    print(f"Initial estimated FPP: {bf.false_positive_probability():.4f}")

    items = [f"item_{i}" for i in range(n * 2)]  # Generate items to potentially add

    # Add items incrementally and check estimated FPP
    steps = [0, int(n * 0.1), int(n * 0.5), n, int(n * 1.5)]
    items_added_count = 0
    for step_target in steps:
        items_to_add_now = items[items_added_count:step_target]
        for item in items_to_add_now:
            bf.update(item)
        items_added_count = step_target

        if items_added_count > 0:
            estimated_fpp = bf.false_positive_probability()
            fill_ratio = sum(bin(byte).count("1") for byte in bf._bytes) / bf._bit_size
            print(f"\nAfter adding {items_added_count} items:")
            print(f"  Approximate unique items added: {bf._approximate_count}")
            print(f"  Filter fill ratio: {fill_ratio:.2%}")
            print(f"  Estimated current FPP: {estimated_fpp:.4f}")
            # Theoretical FPP = (1 - exp(-k*n_actual/m))^k
            # Using estimated cardinality n_est = bf.estimate_cardinality()
            n_est = bf.estimate_cardinality()
            k = bf._hash_count
            m = bf._bit_size
            if m > 0 and k > 0 and n_est >= 0:
                try:
                    theoretic_fpp = (1.0 - math.exp(-k * n_est / m)) ** k
                    print(
                        f"  Theoretical FPP (using est. cardinality {n_est}): {theoretic_fpp:.4f}"
                    )
                except (OverflowError, ValueError):
                    print("  Could not calculate theoretical FPP.")

    print("\nNote: As the filter fills (especially beyond expected capacity),")
    print("the actual false positive probability increases above the target rate.")


def demonstrate_merging():
    """Demonstrate merging (union) of two Bloom Filters."""
    print("\n=== Bloom Filter Merging (Union) Demo ===")

    seed = 88
    params = {"expected_items": 500, "false_positive_rate": 0.02, "seed": seed}

    bf_a = BloomFilter(**params)
    bf_b = BloomFilter(**params)

    items_a = {f"set_a_{i}" for i in range(300)}
    items_common = {f"common_{i}" for i in range(100)}
    items_b = {f"set_b_{i}" for i in range(250)}

    print("Populating Filter A...")
    for item in items_a.union(items_common):
        bf_a.update(item)
    print(f"  Filter A has approx {bf_a._approximate_count} items.")

    print("Populating Filter B...")
    for item in items_b.union(items_common):
        bf_b.update(item)
    print(f"  Filter B has approx {bf_b._approximate_count} items.")

    print("\nMerging Filter A and Filter B...")
    try:
        merged_bf = bf_a.merge(bf_b)
        print("  Merge successful.")
        print(f"  Merged filter estimated size: {merged_bf.estimate_size()} bytes")
        print(f"  Merged filter approx count: {merged_bf._approximate_count}")
        print(f"  Merged filter total processed: {merged_bf._items_processed}")

        # Verify union property (no false negatives)
        all_items = items_a.union(items_b).union(items_common)
        print(
            f"\nVerifying merged filter contains all {len(all_items)} unique items..."
        )
        missing_in_merge = 0
        for item in all_items:
            if not merged_bf.contains(item):
                missing_in_merge += 1
        if missing_in_merge == 0:
            print(
                "  Confirmed: All items from A and B are present in the merged filter."
            )
        else:
            print(f"  ERROR: {missing_in_merge} items missing in the merged filter!")

    except ValueError as e:
        print(f"  ERROR during merge: {e}")
    except TypeError as e:
        print(f"  ERROR during merge (likely wrong type): {e}")


def demonstrate_cardinality_estimation():
    """Show the cardinality (unique item count) estimation feature."""
    print("\n=== Cardinality Estimation Demo ===")

    n = 5000
    bf = BloomFilter(expected_items=n, false_positive_rate=0.01, seed=7)
    print(f"Filter initialized for {n} items.")

    actual_added = 0
    estimates = {}

    # Add items in chunks and estimate
    chunk_size = n // 4
    for i in range(6):  # Add up to 1.5 * n
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size
        items_in_chunk = {f"card_item_{j}" for j in range(start_index, end_index)}
        for item in items_in_chunk:
            bf.update(item)
        actual_added += len(items_in_chunk)

        estimated_cardinality = bf.estimate_cardinality()
        estimates[actual_added] = estimated_cardinality
        print(
            f"  Added {actual_added} unique items => Estimated Cardinality: {estimated_cardinality}"
        )

    print("\nNote: Cardinality estimation is an approximation.")
    print("Accuracy decreases as the filter becomes more saturated (more items added).")


if __name__ == "__main__":
    demonstrate_basic_usage()
    demonstrate_fpp_and_fill_ratio()
    demonstrate_merging()
    demonstrate_cardinality_estimation()
