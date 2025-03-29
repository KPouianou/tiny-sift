"""
HyperLogLog Example for TinySift.

This example demonstrates how to use the HyperLogLog algorithm
for cardinality estimation on data streams.
"""

import random
import sys
import time
from collections import Counter

from tiny_sift.algorithms.hyperloglog import HyperLogLog


def demonstrate_basic_hyperloglog():
    """Demonstrate basic HyperLogLog cardinality estimation on a simulated data stream."""
    print("\n=== Basic HyperLogLog Demo ===")

    # Create a HyperLogLog estimator with default precision (p=12)
    hll = HyperLogLog(seed=42)

    print(
        f"Using precision p={hll._precision} (error ~{hll.error_bound()['relative_error']:.2%})"
    )
    print(f"Memory usage: ~{hll.estimate_size()} bytes")

    # Process a stream of 100,000 integers (simulated stream)
    print("\nProcessing 100,000 integers...")
    unique_count = 0

    for i in range(100000):
        # Generate a value (use i directly for fully unique data)
        hll.update(i)
        unique_count += 1

        # Print progress occasionally
        if i % 20000 == 0:
            print(
                f"  Processed {i} items, current estimate: {hll.estimate_cardinality()}"
            )

    # Get the final cardinality estimate
    final_estimate = hll.estimate_cardinality()
    print(f"\nFinal cardinality estimate: {final_estimate} (true: {unique_count})")
    print(f"Relative error: {abs(final_estimate - unique_count) / unique_count:.2%}")

    # Print detailed stats
    stats = hll.get_stats()
    print("\nEstimator statistics:")
    print(f"  Number of registers: {stats['num_registers']}")
    print(
        f"  Empty registers: {stats['empty_registers']} ({stats['empty_registers']/stats['num_registers']:.2%})"
    )
    print(f"  Maximum register value: {stats['max_register_value']}")
    print(f"  Theoretical standard error: {stats['relative_error']:.2%}")


def demonstrate_zipf_distribution():
    """Demonstrate HyperLogLog on a more realistic Zipf distribution."""
    print("\n=== Zipf Distribution Demo ===")

    # Create a HyperLogLog estimator
    hll = HyperLogLog(precision=14, seed=42)
    print(
        f"Using precision p={hll._precision} (error ~{hll.error_bound()['relative_error']:.2%})"
    )

    # Generate a stream with Zipf distribution (power law)
    # to simulate more realistic internet or commercial data
    print("\nProcessing 1,000,000 items with Zipf distribution...")

    # Parameters for the Zipf distribution
    n_unique = 100000  # Number of possible unique values
    zipf_exponent = 1.2  # Control parameter (higher means more skewed)
    stream_size = 1000000  # Total stream size

    # Create a Zipf distribution (approximation)
    weights = [1.0 / (i + 1) ** zipf_exponent for i in range(n_unique)]
    total = sum(weights)
    weights = [w / total for w in weights]

    # Keep track of true unique count for comparison
    true_uniques = set()

    # Process the stream with progress tracking
    start_time = time.time()
    last_report = 0

    for i in range(stream_size):
        # Generate a value following Zipf distribution
        # (choose an item based on the weights)
        value = random.choices(range(n_unique), weights=weights, k=1)[0]

        # Update the estimator
        hll.update(value)

        # Track true unique count
        true_uniques.add(value)

        # Report progress every 10% or 100,000 items
        if i - last_report >= 100000:
            elapsed = time.time() - start_time
            progress = i / stream_size * 100
            estimated = hll.estimate_cardinality()
            true_count = len(true_uniques)
            rel_error = abs(estimated - true_count) / true_count

            print(f"  {progress:.0f}% complete ({i} items in {elapsed:.1f}s)")
            print(
                f"    Current estimate: {estimated:,} (true: {true_count:,}, error: {rel_error:.2%})"
            )

            last_report = i

    # Final results
    final_estimate = hll.estimate_cardinality()
    true_count = len(true_uniques)
    rel_error = abs(final_estimate - true_count) / true_count

    print("\nFinal results:")
    print(f"  Stream size: {stream_size:,} items")
    print(f"  True unique count: {true_count:,}")
    print(f"  HyperLogLog estimate: {final_estimate:,}")
    print(f"  Relative error: {rel_error:.4%}")
    print(f"  Theoretical error bound: {hll.error_bound()['relative_error']:.4%}")
    print(f"  Memory usage: {hll.estimate_size():,} bytes")

    # For comparison, show the memory that would be required for an exact count
    exact_bytes = sys.getsizeof(true_uniques)
    print(f"  Memory for exact storage (set): {exact_bytes:,} bytes")
    print(f"  Memory ratio: 1:{exact_bytes/hll.estimate_size():.1f}")


def demonstrate_precision_comparison():
    """Compare different precision values for HyperLogLog."""
    print("\n=== Precision Comparison Demo ===")

    # Test multiple precision values
    precisions = [4, 6, 8, 10, 12, 14, 16]

    results = []

    for p in precisions:
        # Create a HyperLogLog with this precision
        hll = HyperLogLog(precision=p, seed=42)

        # Generate deterministic test data
        n_unique = 100000
        for i in range(n_unique):
            hll.update(f"item-{i}")

        # Get results
        estimate = hll.estimate_cardinality()
        rel_error = abs(estimate - n_unique) / n_unique
        memory = hll.estimate_size()
        theoretical_error = hll.error_bound()["relative_error"]

        results.append((p, estimate, rel_error, memory, theoretical_error))

    # Print results table
    print("\nPrecision  Estimate  Actual Error  Memory Usage  Theoretical Error")
    print("----------------------------------------------------------------")
    for p, est, err, mem, t_err in results:
        print(
            f"p = {p:2d}     {est:8,}    {err:.4%}      {mem:7,} bytes       {t_err:.4%}"
        )


def demonstrate_streaming_integration():
    """
    Demonstrate how to use HyperLogLog in a streaming context with checkpointing.
    This simulates how it would integrate with GigQ.
    """
    print("\n=== Streaming Integration Demo ===")

    # Create an estimator for tracking unique users over time
    hll = HyperLogLog(precision=12, seed=42)
    print("Simulating a real-time unique user counting scenario...")

    # Simulate processing data in chunks (e.g., hourly batches)
    time_periods = 24  # Simulate 24 hours of data
    batch_size = 50000  # Average users per hour

    # Parameters for simulating user activity
    total_users = 500000  # Total user pool
    new_user_ratio = 0.05  # 5% chance of a completely new user per request

    # Track true uniques for validation
    all_users = set()
    period_users = []  # Users seen in each period

    print("\nProcessing data in batches...")
    for period in range(time_periods):
        period_start = time.time()

        # Reset tracking for this period
        period_unique = set()

        # Process data for this period
        for i in range(batch_size):
            # Simulate user behavior: mix of returning users and new users
            if random.random() < new_user_ratio:
                # New user
                user_id = f"user-{len(all_users) + 1}"
            else:
                # Returning user (if any exist)
                if all_users:
                    user_id = random.choice(list(all_users))
                else:
                    user_id = f"user-1"

            # Update the estimator and tracking sets
            hll.update(user_id)
            all_users.add(user_id)
            period_unique.add(user_id)

        # Store this period's users
        period_users.append(period_unique)

        # Print results for this period
        period_time = time.time() - period_start
        estimate = hll.estimate_cardinality()
        true_count = len(all_users)
        rel_error = abs(estimate - true_count) / true_count

        print(f"\nPeriod {period + 1} (e.g., Hour {period + 1}):")
        print(f"  Users in this period: {len(period_unique):,}")
        print(f"  Total unique users so far: {true_count:,}")
        print(f"  HyperLogLog estimate: {estimate:,} (error: {rel_error:.2%})")
        print(f"  Processing time: {period_time:.3f} seconds")

        # Simulate serializing/deserializing the estimator for persistence
        # (e.g., to store in GigQ's SQLite database between jobs)
        if period % 6 == 5:  # Every 6 hours
            print("  Checkpointing estimator state...")
            serialized = hll.serialize()
            print(f"    Serialized size: {len(serialized):,} bytes")

            # In a real scenario, this would be stored and loaded from
            # GigQ's database between job runs

            # Simulate loading from checkpoint
            hll = HyperLogLog.deserialize(serialized)

    # Final summary
    print("\nFinal summary:")
    print(f"  Total periods: {time_periods}")
    print(f"  Total events processed: {time_periods * batch_size:,}")
    print(f"  True unique users: {len(all_users):,}")
    print(f"  Final estimate: {hll.estimate_cardinality():,}")
    print(
        f"  Relative error: {abs(hll.estimate_cardinality() - len(all_users)) / len(all_users):.4%}"
    )

    # Calculate metrics for each period
    print("\nUnique user count by period:")
    for i, period_set in enumerate(period_users):
        print(f"  Period {i+1}: {len(period_set):,} users")

    # Calculate overlap between adjacent periods
    for i in range(len(period_users) - 1):
        overlap = len(period_users[i] & period_users[i + 1])
        pct = overlap / len(period_users[i]) * 100
        print(
            f"  Overlap between periods {i+1} and {i+2}: {overlap:,} users ({pct:.1f}%)"
        )


if __name__ == "__main__":
    demonstrate_basic_hyperloglog()
    demonstrate_zipf_distribution()
    demonstrate_precision_comparison()
    demonstrate_streaming_integration()
