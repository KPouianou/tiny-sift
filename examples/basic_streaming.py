"""
Basic example of using TinySift for stream processing.

This example demonstrates how to use the Reservoir Sampling algorithm
to maintain a random sample of a data stream.
"""

import random
import sys
import time
from collections import Counter

from tiny_sift.algorithms.reservoir import ReservoirSampling, WeightedReservoirSampling


def demonstrate_basic_reservoir():
    """Demonstrate basic reservoir sampling on a simulated data stream."""
    print("\n=== Basic Reservoir Sampling Demo ===")

    # Create a reservoir with size 10
    reservoir = ReservoirSampling(size=10, seed=42)

    # Process a stream of 1000 integers (simulated stream)
    print("Processing 1000 integers...")
    for i in range(1000):
        reservoir.update(i)

        # Print progress occasionally
        if i % 200 == 0:
            print(f"  Processed {i} items")

    # Get the final sample
    sample = reservoir.get_sample()
    print(f"\nFinal sample (size {len(sample)}):")
    print(sample)

    print(f"\nTotal items processed: {reservoir.items_processed}")
    print(f"Approximate memory usage: {reservoir.estimate_size()} bytes")

    # Serialize the reservoir
    serialized = reservoir.serialize(format="json")
    print(f"\nSerialized size: {len(serialized)} bytes")

    # Deserialize to a new reservoir
    new_reservoir = ReservoirSampling.deserialize(serialized, format="json")
    print(f"Deserialized sample: {new_reservoir.get_sample()}")


def demonstrate_weighted_reservoir():
    """Demonstrate weighted reservoir sampling with user-defined weights."""
    print("\n=== Weighted Reservoir Sampling Demo ===")

    # Create a weighted reservoir with size 20
    reservoir = WeightedReservoirSampling(size=20, seed=42)

    # Define some categories with different weights
    categories = {
        "A": 10.0,  # High weight (10x more likely than C)
        "B": 5.0,  # Medium weight (5x more likely than C)
        "C": 1.0,  # Low weight
    }

    # Process a stream with weighted items
    print("Processing 1000 weighted items...")
    for i in range(1000):
        # Choose a random category for this item
        category = random.choice(list(categories.keys()))

        # Add to reservoir with its weight
        reservoir.update(f"{category}-{i}", weight=categories[category])

        # Print progress occasionally
        if i % 200 == 0:
            print(f"  Processed {i} items")

    # Get the final sample
    sample = reservoir.get_sample()

    # Count categories in the sample
    category_counts = Counter()
    for item in sample:
        category = item.split("-")[0]
        category_counts[category] += 1

    # Print the results
    print(f"\nFinal sample (size {len(sample)}):")
    for category, count in sorted(category_counts.items()):
        print(f"  Category {category} (weight {categories[category]}): {count} items")

    print(f"\nTotal items processed: {reservoir.items_processed}")
    print(f"Approximate memory usage: {reservoir.estimate_size()} bytes")


def simulate_integration_with_gigq():
    """Simulate how TinySift could integrate with GigQ for job processing."""
    print("\n=== GigQ Integration Simulation ===")

    # Simulate a data stream job with GigQ
    print("Simulating a data stream processing job...")

    # Create a reservoir
    reservoir = ReservoirSampling(size=100)

    # Process chunks of data as they would arrive in a job
    for chunk_id in range(5):
        print(f"\nProcessing chunk {chunk_id + 1}...")

        # Simulate processing a chunk of data
        chunk_size = 1000
        start_time = time.time()

        for i in range(chunk_size):
            # Generate a simulated data point
            data_point = f"item-{chunk_id}-{i}"

            # Update the reservoir
            reservoir.update(data_point)

        # Print chunk processing stats
        elapsed = time.time() - start_time
        print(f"  Processed {chunk_size} items in {elapsed:.3f} seconds")
        print(f"  Items processed so far: {reservoir.items_processed}")
        print(f"  Current reservoir size: {len(reservoir.get_sample())}")

        # In a real GigQ job, we would serialize the reservoir
        # and save it to the database for checkpointing
        serialized = reservoir.serialize()
        print(f"  Checkpoint saved: {len(serialized)} bytes")

    # Get a sample of results from the final state
    sample = reservoir.get_sample()
    print(f"\nFinal sample contains {len(sample)} items")

    # Analyze sample - count items from each chunk
    chunk_counts = Counter()
    for item in sample:
        chunk_id = int(item.split("-")[1])
        chunk_counts[chunk_id] += 1

    print("Items per chunk in the sample:")
    for chunk_id, count in sorted(chunk_counts.items()):
        print(f"  Chunk {chunk_id}: {count} items")


if __name__ == "__main__":
    demonstrate_basic_reservoir()
    demonstrate_weighted_reservoir()
    simulate_integration_with_gigq()
