"""
Space-Saving Algorithm Demo for TinySift.

This example demonstrates how to use the Space-Saving algorithm
for finding the most frequent items in a data stream with limited memory.
"""

import random
import time
from collections import Counter
import math

from tiny_sift.algorithms.spacesaving import SpaceSaving


def demonstrate_basic_usage():
    """Demonstrate basic Space-Saving algorithm usage."""
    print("\n=== Basic Space-Saving Demo ===")

    # Create a Space-Saving counter with capacity for 10 items
    ss = SpaceSaving(capacity=10)

    # Add some items
    items = [
        "apple",
        "banana",
        "apple",
        "cherry",
        "apple",
        "banana",
        "apple",
        "date",
        "banana",
        "apple",
        "cherry",
    ]

    print(f"Processing {len(items)} items...")
    for item in items:
        ss.update(item)

    # Get frequencies
    print("\nEstimated frequencies:")
    for item in set(items):
        freq = ss.estimate_frequency(item)
        print(f"  {item}: {freq}")

    # Get top-k items
    print("\nTop-3 items:")
    top_items = ss.get_top_k(3)
    for item, freq in top_items:
        print(f"  {item}: {freq}")

    # Show total items processed
    print(f"\nTotal items processed: {ss.items_processed}")


def demonstrate_performance_on_zipf():
    """Demonstrate Space-Saving on a Zipf distribution (power law)."""
    print("\n=== Space-Saving with Zipf Distribution ===")
    print("This demonstrates finding frequent items in a power-law distribution")
    print("where some items are much more common than others.")

    # Create a Space-Saving counter with limited capacity
    capacity = 50
    ss = SpaceSaving(capacity=capacity)

    # Generate stream following Zipf-like distribution
    n_items = 10000
    n_distinct = 1000

    # Create weights for Zipf distribution
    zipf_exponent = 1.5  # Controls how skewed the distribution is
    weights = [1.0 / (i + 1) ** zipf_exponent for i in range(n_distinct)]

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Generate stream and track exact frequencies for comparison
    stream = []
    exact_counter = Counter()

    print(
        f"Generating a stream of {n_items} items from {n_distinct} distinct values..."
    )
    print(f"Using Zipf exponent {zipf_exponent} (higher means more skewed)")

    # Use a deterministic seed for reproducible results
    random.seed(42)

    for _ in range(n_items):
        item = random.choices(range(n_distinct), weights=weights, k=1)[0]
        stream.append(item)
        exact_counter[item] += 1

    # Track time for processing the stream
    start_time = time.time()

    # Process the stream with Space-Saving
    for item in stream:
        ss.update(item)

    processing_time = time.time() - start_time

    print(f"\nProcessed {n_items} items in {processing_time:.4f} seconds")
    print(
        f"Space-Saving used {capacity} counters ({capacity/n_distinct:.1%} of distinct items)"
    )

    # Compare top results with exact counts
    top_k = 20
    print(f"\nComparing top {top_k} results:")
    space_saving_top = ss.get_top_k(top_k)
    exact_top = exact_counter.most_common(top_k)

    # Calculate error metrics
    total_error = 0
    max_error = 0

    print("\n    Item   |  Exact   | Estimated | Error  ")
    print("-----------|----------|-----------|--------")

    for i in range(top_k):
        ss_item, ss_freq = space_saving_top[i]
        exact_item, exact_freq = exact_top[i]

        # Verify it's the same item (should match for truly frequent items)
        if ss_item == exact_item:
            error = abs(ss_freq - exact_freq)
            error_pct = 100 * error / exact_freq if exact_freq > 0 else 0

            print(f"  {ss_item:8} | {exact_freq:8} | {ss_freq:9} | {error_pct:5.1f}%")

            total_error += error
            max_error = max(max_error, error)

    avg_error = total_error / top_k

    # Calculate theoretical error bound
    theoretical_bound = n_items / capacity

    print(f"\nError statistics:")
    print(
        f"  Average error: {avg_error:.2f} ({avg_error/n_items*100:.4f}% of stream size)"
    )
    print(
        f"  Maximum error: {max_error:.2f} ({max_error/n_items*100:.4f}% of stream size)"
    )
    print(
        f"  Theoretical maximum error: {theoretical_bound:.2f} "
        + f"({theoretical_bound/n_items*100:.4f}% of stream size)"
    )


def demonstrate_heavy_hitters():
    """Demonstrate using Space-Saving for heavy hitter detection."""
    print("\n=== Heavy Hitter Detection ===")

    # Threshold for heavy hitters (proportion of total stream)
    phi = 0.05  # 5%

    # Tolerable error rate
    epsilon = 0.01  # 1%

    # Calculate required capacity based on theory
    capacity = math.ceil(1.0 / (phi * epsilon))

    print(f"Creating Space-Saving sketch configured for:")
    print(f"  Heavy hitter threshold: {phi:.1%} of total stream")
    print(f"  Error rate: {epsilon:.1%}")
    print(f"  Required capacity: {capacity} counters")

    # Create sketch using utility method
    ss = SpaceSaving.create_from_error_rate(phi, epsilon)

    # Verify capacity
    print(f"  Created sketch with capacity: {ss._capacity}")

    # Generate stream with known heavy hitters
    n_items = 100000
    print(f"\nGenerating stream with {n_items} items...")

    # Create 3 heavy hitters
    heavy_a = n_items * 0.20  # 20% of stream
    heavy_b = n_items * 0.15  # 15% of stream
    heavy_c = n_items * 0.08  # 8% of stream

    # Counts for items that are NOT heavy hitters
    remaining = n_items - heavy_a - heavy_b - heavy_c
    other_items = 1000

    # Create and process the stream
    exact_counts = Counter()

    # Add heavy hitters
    for _ in range(int(heavy_a)):
        ss.update("A")
        exact_counts["A"] += 1

    for _ in range(int(heavy_b)):
        ss.update("B")
        exact_counts["B"] += 1

    for _ in range(int(heavy_c)):
        ss.update("C")
        exact_counts["C"] += 1

    # Add remaining items with uniform distribution
    for _ in range(int(remaining)):
        item = f"item-{random.randint(0, other_items-1)}"
        ss.update(item)
        exact_counts[item] += 1

    # Verify total count
    print(f"Total items processed: {ss.items_processed}")

    # Get heavy hitters using threshold
    heavy_hitters = ss.get_heavy_hitters(phi)

    print(f"\nDetected heavy hitters (threshold = {phi:.1%}):")
    print("   Item   | Estimated | Exact  | Percentage | Error")
    print("----------|-----------|--------|------------|------")

    # Calculate actual heavy hitters for verification
    actual_heavy = {
        item: count for item, count in exact_counts.items() if count >= phi * n_items
    }

    # True positives
    for item, count in sorted(heavy_hitters.items(), key=lambda x: x[1], reverse=True):
        exact = exact_counts[item]
        percent = exact / n_items * 100
        error = abs(count - exact)
        error_pct = error / exact * 100 if exact > 0 else 0

        print(
            f"  {item:7} | {count:9.0f} | {exact:6.0f} | {percent:8.2f}% | {error_pct:.2f}%"
        )

        # Remove from actual heavy hitters to identify false negatives
        if item in actual_heavy:
            del actual_heavy[item]

    # Check for false negatives (items that should be heavy hitters but weren't detected)
    if actual_heavy:
        print("\nMissed heavy hitters (false negatives):")
        for item, count in actual_heavy.items():
            percent = count / n_items * 100
            print(f"  {item}: {count} ({percent:.2f}%)")
    else:
        print("\nNo false negatives - detected all true heavy hitters!")

    # Check for false positives
    false_positives = [
        item for item in heavy_hitters if exact_counts[item] < phi * n_items
    ]

    if false_positives:
        print("\nFalse positives (items incorrectly identified as heavy hitters):")
        for item in false_positives:
            count = exact_counts[item]
            percent = count / n_items * 100
            print(f"  {item}: {count} ({percent:.2f}%)")
    else:
        print("\nNo false positives - all detected items are genuine heavy hitters!")


def demonstrate_merging():
    """Demonstrate merging two Space-Saving sketches."""
    print("\n=== Merging Space-Saving Sketches ===")

    # Create two sketches with the same capacity
    capacity = 20
    ss1 = SpaceSaving(capacity=capacity)
    ss2 = SpaceSaving(capacity=capacity)

    print(f"Creating two sketches with capacity {capacity}")

    # Add items to first sketch
    stream1 = ["A"] * 100 + ["B"] * 80 + ["C"] * 60 + ["D"] * 40
    random.shuffle(stream1)

    print(f"Processing stream 1 with {len(stream1)} items...")
    for item in stream1:
        ss1.update(item)

    # Add different items to second sketch
    stream2 = ["B"] * 120 + ["C"] * 100 + ["E"] * 80 + ["F"] * 60
    random.shuffle(stream2)

    print(f"Processing stream 2 with {len(stream2)} items...")
    for item in stream2:
        ss2.update(item)

    # Print top items from each sketch
    print("\nTop items from sketch 1:")
    for item, freq in ss1.get_top_k(4):
        print(f"  {item}: {freq}")

    print("\nTop items from sketch 2:")
    for item, freq in ss2.get_top_k(4):
        print(f"  {item}: {freq}")

    # Merge the sketches
    merged = ss1.merge(ss2)

    print("\nTop items from merged sketch:")
    for item, freq in merged.get_top_k(6):
        print(f"  {item}: {freq}")

    # Calculate exact counts for comparison
    exact = Counter()
    for item in stream1:
        exact[item] += 1
    for item in stream2:
        exact[item] += 1

    print("\nExact counts for comparison:")
    for item, count in exact.most_common(6):
        print(f"  {item}: {count}")


if __name__ == "__main__":
    demonstrate_basic_usage()
    demonstrate_performance_on_zipf()
    demonstrate_heavy_hitters()
    demonstrate_merging()
