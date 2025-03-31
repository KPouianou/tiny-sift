"""
Counting Bloom Filter Demo for TinySift.

This example demonstrates how to use the Counting Bloom Filter
for efficient set membership testing with support for item deletion.
"""

import random
import sys
import time
from collections import Counter

from tiny_sift.algorithms.bloom import CountingBloomFilter


def demonstrate_counting_bloom_filter():
    """Demonstrate Counting Bloom Filter with deletion capability."""
    print("\n=== Counting Bloom Filter Demo ===")

    # Create a Counting Bloom filter
    cbf = CountingBloomFilter(
        expected_items=1000,
        false_positive_rate=0.01,
        counter_bits=4,  # 4 bits per counter (values 0-15)
        seed=42,
    )

    print(f"Counting Filter parameters:")
    print(f"  Expected items: 1,000")
    print(f"  Target false positive rate: 1%")
    print(
        f"  Counter bits: {cbf._counter_bits} bits per counter (max value: {cbf._counter_max})"
    )
    print(f"  Memory usage: {cbf.estimate_size():,} bytes")

    # Add some items
    print("\nAdding items to the filter...")
    words = [
        "apple",
        "banana",
        "cherry",
        "date",
        "elderberry",
        "fig",
        "grape",
        "honeydew",
    ]

    for word in words:
        cbf.update(word)
        print(f"  Added '{word}' to filter")

    # Check membership
    print("\nChecking membership:")
    test_words = words + ["kiwi", "lemon", "mango"]
    for word in test_words:
        print(f"  '{word}' in filter: {cbf.contains(word)}")

    # Demonstrate removal
    print("\nRemoving items from the filter...")
    words_to_remove = ["apple", "cherry", "grape", "kiwi"]
    for word in words_to_remove:
        result = cbf.remove(word)
        print(f"  Removed '{word}' from filter: {result}")

    # Check membership after removal
    print("\nChecking membership after removal:")
    for word in test_words:
        print(f"  '{word}' in filter: {cbf.contains(word)}")

    # Demonstrate multiple insertions and removals
    print("\nDemonstrating counter behavior with multiple insertions/removals:")

    # Add "test" 3 times
    print("  Adding 'test' three times...")
    for _ in range(3):
        cbf.update("test")

    print(f"  'test' in filter: {cbf.contains('test')}")

    # Remove once
    print("  Removing 'test' once...")
    cbf.remove("test")
    print(f"  'test' in filter: {cbf.contains('test')}")  # Should still be true

    # Remove again
    print("  Removing 'test' again...")
    cbf.remove("test")
    print(f"  'test' in filter: {cbf.contains('test')}")  # Should still be true

    # Remove a third time
    print("  Removing 'test' a third time...")
    cbf.remove("test")
    print(f"  'test' in filter: {cbf.contains('test')}")  # Should be false now


def demonstrate_cache_application():
    """Demonstrate using CountingBloomFilter for a cache application."""
    print("\n=== Cache Filtering with Counting Bloom Filter ===")
    print("A Counting Bloom Filter can be used to implement an efficient cache")
    print("that can both add and remove items.")

    # Create a counting Bloom filter for caching
    cache_filter = CountingBloomFilter(
        expected_items=1000, false_positive_rate=0.01, counter_bits=4
    )

    # Simulate cache operations
    print("\nSimulating cache operations:")

    # Add some items to the cache
    cache_items = ["user:1001", "product:2345", "session:a1b2c3", "settings:dark-mode"]
    for item in cache_items:
        cache_filter.update(item)
        print(f"  Added to cache: {item}")

    # Check if items are in the cache
    check_items = cache_items + ["user:9999", "product:8888"]
    print("\nChecking cache membership:")
    for item in check_items:
        cached = cache_filter.contains(item)
        print(f"  {item}: {'Cached' if cached else 'Not cached'}")

    # Remove some items from the cache
    print("\nRemoving items from cache:")
    for item in ["user:1001", "session:a1b2c3", "user:9999"]:
        result = cache_filter.remove(item)
        if result:
            print(f"  Removed from cache: {item}")
        else:
            print(f"  Not in cache: {item}")

    # Check again after removal
    print("\nChecking cache membership after removals:")
    for item in check_items:
        cached = cache_filter.contains(item)
        print(f"  {item}: {'Cached' if cached else 'Not cached'}")

    print("\nCountingBloomFilter memory usage:")
    print(f"  {cache_filter.estimate_size():,} bytes")


def demonstrate_counting_vs_standard():
    """Compare standard Bloom filter with Counting Bloom filter."""
    print("\n=== Comparing Standard vs. Counting Bloom Filters ===")

    # Parameters for comparison
    num_items = 10000
    false_positive_rate = 0.01

    # Create both types of filters
    std_filter = CountingBloomFilter(
        expected_items=num_items,
        false_positive_rate=false_positive_rate,
        counter_bits=1,  # This mimics a standard Bloom filter (1 bit)
    )

    counting_filter = CountingBloomFilter(
        expected_items=num_items,
        false_positive_rate=false_positive_rate,
        counter_bits=4,  # 4 bits per counter
    )

    # Compare memory usage
    std_size = std_filter.estimate_size()
    counting_size = counting_filter.estimate_size()

    print("Memory usage comparison:")
    print(f"  Standard Bloom filter: {std_size:,} bytes")
    print(f"  Counting Bloom filter: {counting_size:,} bytes")
    print(f"  Size ratio: {counting_size / std_size:.2f}x larger")

    # Insert items and measure time
    items = [f"item-{i}" for i in range(num_items)]

    print("\nInsertion time comparison:")

    # Standard filter
    start_time = time.time()
    for item in items:
        std_filter.update(item)
    std_insert_time = time.time() - start_time
    print(f"  Standard Bloom filter: {std_insert_time:.6f} seconds")

    # Counting filter
    start_time = time.time()
    for item in items:
        counting_filter.update(item)
    counting_insert_time = time.time() - start_time
    print(f"  Counting Bloom filter: {counting_insert_time:.6f} seconds")
    print(f"  Time ratio: {counting_insert_time / std_insert_time:.2f}x slower")

    # Query time
    print("\nQuery time comparison:")

    # Standard filter
    start_time = time.time()
    for item in items:
        std_filter.contains(item)
    std_query_time = time.time() - start_time
    print(f"  Standard Bloom filter: {std_query_time:.6f} seconds")

    # Counting filter
    start_time = time.time()
    for item in items:
        counting_filter.contains(item)
    counting_query_time = time.time() - start_time
    print(f"  Counting Bloom filter: {counting_query_time:.6f} seconds")
    print(f"  Time ratio: {counting_query_time / std_query_time:.2f}x slower")

    # Demonstrate removal capability (unique to counting filter)
    print("\nDemonstrating removal capability:")
    print("  Removing 100 items from counting filter...")

    for item in items[:100]:
        counting_filter.remove(item)

    # Check that items were removed
    removed_count = 0
    for item in items[:100]:
        if not counting_filter.contains(item):
            removed_count += 1

    print(f"  Successfully removed {removed_count} out of 100 items")

    # Summary
    print("\nSummary:")
    print("  Standard Bloom Filter:")
    print("    - More memory efficient")
    print("    - Faster operations")
    print("    - No support for element removal")
    print("  Counting Bloom Filter:")
    print("    - Supports element removal")
    print(f"    - Requires {counting_size / std_size:.1f}x more memory")
    print(f"    - Operations are {counting_query_time / std_query_time:.1f}x slower")
    print("    - Better choice when item removal is required")


def demonstrate_set_operations():
    """Demonstrate set operations with Counting Bloom filters."""
    print("\n=== Set Operations with Bloom Filters ===")

    # Create two Counting Bloom filters
    filter_a = CountingBloomFilter(
        expected_items=100, false_positive_rate=0.01, seed=42
    )
    filter_b = CountingBloomFilter(
        expected_items=100, false_positive_rate=0.01, seed=42
    )

    # Add different items to each filter with some overlap
    items_a = [f"a-{i}" for i in range(50)] + [f"common-{i}" for i in range(20)]
    items_b = [f"b-{i}" for i in range(40)] + [f"common-{i}" for i in range(20)]

    for item in items_a:
        filter_a.update(item)

    for item in items_b:
        filter_b.update(item)

    print(f"Filter A contains {len(items_a)} items")
    print(f"Filter B contains {len(items_b)} items")
    print(f"They share 20 common items")

    # Demonstrate union via merging
    print("\nDemonstrating union operation (A ∪ B):")
    union_filter = filter_a.merge(filter_b)

    # Count matches in the union
    all_items = set(items_a + items_b)
    match_count = sum(1 for item in all_items if union_filter.contains(item))

    print(f"  Union should contain {len(all_items)} unique items")
    print(f"  Items detected in union: {match_count}")
    print(f"  Union memory usage: {union_filter.estimate_size():,} bytes")

    # Demonstrate how to simulate intersection (A ∩ B)
    print("\nSimulating intersection operation (A ∩ B):")
    print(
        "  (Note: This is an approximation since Bloom filters don't support direct intersection)"
    )

    # Check which items are in both filters
    common_items = []
    potential_common = set(items_a + items_b)

    for item in potential_common:
        if filter_a.contains(item) and filter_b.contains(item):
            common_items.append(item)

    print(f"  Found {len(common_items)} items in the intersection")
    print(f"  Actual number of common items: 20")

    # If there are false positives, explain them
    false_positives = len(common_items) - 20
    if false_positives > 0:
        print(f"  False positives in intersection: {false_positives}")
        print("  These are items incorrectly identified as being in both filters")

    # Demonstrate how to simulate difference (A - B)
    print("\nSimulating difference operation (A - B):")
    print(
        "  (Note: This is an approximation since Bloom filters don't support direct difference)"
    )

    # Check which items are in A but not in B
    a_minus_b = []

    for item in items_a:
        if not filter_b.contains(item):
            a_minus_b.append(item)

    print(f"  Found {len(a_minus_b)} items in A - B")
    print(f"  Expected number of items in A - B: {len(items_a) - 20}")

    # Summary about set operations
    print("\nSummary of set operations with Bloom filters:")
    print("  - Union (∪): Directly supported through merging filters")
    print("  - Intersection (∩): Approximate by checking items in both filters")
    print(
        "  - Difference (−): Approximate by checking items in one filter but not the other"
    )
    print(
        "  - All approximate operations may have false positives, but no false negatives"
    )


if __name__ == "__main__":
    demonstrate_counting_bloom_filter()
    demonstrate_cache_application()
    demonstrate_counting_vs_standard()
    demonstrate_set_operations()
