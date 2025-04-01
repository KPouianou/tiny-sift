"""
Exponential Histogram Demo for TinySift.

This example demonstrates how to use the Exponential Histogram algorithm
for sliding window statistics on data streams.
"""

import random
import sys
import time
from collections import deque
from datetime import datetime, timedelta

from tiny_sift.algorithms.histogram import ExponentialHistogram


def demonstrate_basic_histogram():
    """Demonstrate basic Exponential Histogram on a simulated data stream."""
    print("\n=== Basic Exponential Histogram Demo ===")

    # Create a histogram with window size of 100 and 1% error bound
    histogram = ExponentialHistogram(window_size=100, error_bounds=0.01)

    print(f"Window size: {histogram._window_size} items")
    print(f"Error bound: {histogram._error_bound:.1%}")
    print(f"Initial memory usage: {histogram.estimate_size()} bytes")

    # Process a stream of 500 values (simulated stream)
    print("\nProcessing 500 values...")

    for i in range(500):
        # Generate a value (just use i for simplicity)
        histogram.update(value=float(i))

        # Print progress occasionally
        if i % 100 == 0:
            stats = histogram.get_window_stats()
            print(f"  Processed {i} items, current window stats:")
            print(f"    Count: {stats['count']}")
            print(f"    Sum: {stats['sum']:.1f}")
            print(f"    Average: {stats['average']:.1f}")

    # Get the final window statistics
    final_stats = histogram.get_window_stats()
    print(f"\nFinal window statistics:")
    print(f"  Count: {final_stats['count']}")
    print(f"  Sum: {final_stats['sum']:.1f}")
    print(f"  Average: {final_stats['average']:.1f}")
    print(f"  Memory usage: {histogram.estimate_size()} bytes")

    # The window should contain the most recent 100 items (400-499)
    expected_sum = sum(range(400, 500))
    expected_avg = expected_sum / 100
    print(f"\nExpected statistics for items 400-499:")
    print(f"  Count: 100")
    print(f"  Sum: {expected_sum}")
    print(f"  Average: {expected_avg:.1f}")

    # Calculate error
    sum_error = abs(final_stats["sum"] - expected_sum) / expected_sum
    print(f"\nRelative error in sum: {sum_error:.2%}")
    print(f"Error bound guarantee: {histogram._error_bound:.2%}")


def demonstrate_time_based_window():
    """Demonstrate time-based sliding window statistics."""
    print("\n=== Time-Based Window Demo ===")

    # Create a histogram with 60 second window and 1% error
    window_seconds = 60
    histogram = ExponentialHistogram(
        window_size=window_seconds, error_bounds=0.01, is_time_based=True
    )

    print(f"Window size: {window_seconds} seconds")
    print(f"Error bound: {histogram._error_bound:.1%}")

    # Simulate a stream of measurements over 3 minutes
    # Backdate to start 3 minutes ago
    start_time = int(time.time()) - 180

    print("\nSimulating 3 minutes of data with 1 measurement per second...")

    # For comparison, we'll track exact values with a deque
    exact_values = deque()

    # Generate one measurement per second for 3 minutes
    for second in range(180):
        # Calculate timestamp for this measurement
        timestamp = start_time + second

        # Generate a random value (e.g., temperature between 15-25°C)
        value = 20.0 + random.uniform(-5.0, 5.0)

        # Update the histogram
        histogram.update(value=value, timestamp=timestamp)

        # Update exact tracking
        exact_values.append((timestamp, value))

        # Remove expired values from exact tracking
        while exact_values and exact_values[0][0] < timestamp - window_seconds:
            exact_values.popleft()

        # Print stats every 30 seconds
        if second % 30 == 29:
            # Calculate exact statistics
            exact_count = len(exact_values)
            exact_sum = sum(v for _, v in exact_values)
            exact_avg = exact_sum / exact_count if exact_count else 0

            # Get histogram statistics
            approx_stats = histogram.get_window_stats()

            # Print comparison
            time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            print(f"\nAt time {time_str} (second {second+1}):")
            print(f"  Exact count: {exact_count}")
            print(f"  Exact sum: {exact_sum:.2f}")
            print(f"  Exact average: {exact_avg:.2f}")
            print(f"  Approx count: {approx_stats['count']}")
            print(f"  Approx sum: {approx_stats['sum']:.2f}")
            print(f"  Approx average: {approx_stats['average']:.2f}")

            # Calculate error
            if exact_sum:
                sum_error = abs(approx_stats["sum"] - exact_sum) / exact_sum
                print(f"  Relative error: {sum_error:.2%}")

    print(f"\nFinal memory usage: {histogram.estimate_size()} bytes")

    # For comparison, calculate memory usage of exact approach
    exact_size = sys.getsizeof(exact_values)
    for timestamp, value in exact_values:
        exact_size += sys.getsizeof(timestamp) + sys.getsizeof(value)

    print(f"Memory for exact storage: {exact_size} bytes")
    print(f"Memory ratio: 1:{exact_size/histogram.estimate_size():.1f}")


def demonstrate_min_max_tracking():
    """Demonstrate tracking minimum and maximum values in the window."""
    print("\n=== Min/Max Tracking Demo ===")

    # Create a histogram with min/max tracking
    histogram = ExponentialHistogram(
        window_size=100, error_bounds=0.05, track_min_max=True
    )

    print("Processing stream with random values...")

    # Generate sequence with some extreme values
    seq = [random.uniform(10, 50) for _ in range(95)]
    # Add some extreme values
    seq.extend([5.0, 7.0, 60.0, 55.0, 6.0])

    for i, value in enumerate(seq):
        histogram.update(value=value)

        if i % 20 == 19:
            stats = histogram.get_window_stats()
            print(f"\nAfter {i+1} items:")
            print(f"  Min: {stats['min']:.1f}")
            print(f"  Max: {stats['max']:.1f}")
            print(f"  Average: {stats['average']:.1f}")

    # Final stats
    final_stats = histogram.get_window_stats()
    print("\nFinal window statistics:")
    print(f"  Min: {final_stats['min']:.1f}")
    print(f"  Max: {final_stats['max']:.1f}")
    print(f"  Average: {final_stats['average']:.1f}")

    # Calculate exact min/max for comparison
    exact_min = min(seq)
    exact_max = max(seq)
    print("\nExact statistics:")
    print(f"  Min: {exact_min:.1f}")
    print(f"  Max: {exact_max:.1f}")


def demonstrate_serialization():
    """Demonstrate serializing and deserializing a histogram."""
    print("\n=== Serialization Demo ===")

    # Create and populate a histogram
    histogram = ExponentialHistogram(
        window_size=3600, error_bounds=0.01, is_time_based=True
    )

    # Add some data
    base_time = int(time.time()) - 3000  # Backdate to 50 minutes ago
    for i in range(0, 3000, 10):  # One measurement every 10 seconds for 50 minutes
        timestamp = base_time + i
        value = 100.0 + i / 100.0
        histogram.update(value=value, timestamp=timestamp)

    # Get statistics before serialization
    before_stats = histogram.get_window_stats()
    print("Statistics before serialization:")
    print(f"  Count: {before_stats['count']}")
    print(f"  Sum: {before_stats['sum']:.1f}")
    print(f"  Average: {before_stats['average']:.1f}")

    # Serialize to JSON
    serialized = histogram.serialize(format="json")
    print(f"\nSerialized size: {len(serialized)} bytes")

    # Deserialize to a new histogram
    new_histogram = ExponentialHistogram.deserialize(serialized, format="json")

    # Get statistics after deserialization
    after_stats = new_histogram.get_window_stats()
    print("\nStatistics after deserialization:")
    print(f"  Count: {after_stats['count']}")
    print(f"  Sum: {after_stats['sum']:.1f}")
    print(f"  Average: {after_stats['average']:.1f}")

    # Verify they match
    count_match = before_stats["count"] == after_stats["count"]
    sum_match = abs(before_stats["sum"] - after_stats["sum"]) < 0.001

    print(f"\nCount matches: {count_match}")
    print(f"Sum matches: {sum_match}")


def demonstrate_gigq_integration():
    """
    Demonstrate how to use Exponential Histogram in a GigQ workflow.
    This simulates how it would integrate with GigQ.
    """
    print("\n=== GigQ Integration Demo ===")

    print("Simulating a stream processing workflow with GigQ...")
    print("Each job processes a batch of sensor data and updates statistics.")

    # Create a histogram for tracking sensor statistics
    histogram = ExponentialHistogram(
        window_size=24 * 3600,  # 24 hour window
        error_bounds=0.01,
        is_time_based=True,
        track_min_max=True,
    )

    # Simulate processing 24 hourly jobs (representing 1 day)
    # In a real scenario, each job would load the histogram from the database,
    # process new data, and save the updated histogram back
    for hour in range(24):
        print(f"\nRunning job for hour {hour}...")

        # In a real job, we'd load the histogram from the database
        # histogram = load_from_database()

        # Simulate the current batch timestamp
        batch_time = time.time() - (24 - hour) * 3600  # Backdate to appropriate hour

        # Process sensor readings for this hour
        # Assume 60 readings (one per minute)
        readings_count = 0
        readings_sum = 0
        readings_min = float("inf")
        readings_max = float("-inf")

        for minute in range(60):
            # Generate a reading value that varies by hour of day
            # Simulating temperature that peaks in afternoon
            hour_factor = 1.0 - abs(hour - 12) / 12.0  # Highest at noon
            reading = 15 + 10 * hour_factor + random.uniform(-2, 2)

            # Update min/max for this batch
            readings_min = min(readings_min, reading)
            readings_max = max(readings_max, reading)

            # Update counts for this batch
            readings_count += 1
            readings_sum += reading

            # Add to the histogram
            timestamp = int(batch_time + minute * 60)
            histogram.update(value=reading, timestamp=timestamp)

        # Calculate batch statistics
        batch_avg = readings_sum / readings_count

        # Get current window statistics (last 24 hours)
        window_stats = histogram.get_window_stats()

        # Print batch results
        print(f"  Batch statistics (hour {hour}):")
        print(f"    Count: {readings_count}")
        print(f"    Average: {batch_avg:.2f}°C")
        print(f"    Min: {readings_min:.2f}°C")
        print(f"    Max: {readings_max:.2f}°C")

        # Print window results (cumulative)
        print(f"  Window statistics (last {hour+1} hours):")
        print(f"    Count: {window_stats['count']}")
        print(f"    Average: {window_stats['average']:.2f}°C")
        print(f"    Min: {window_stats['min']:.2f}°C")
        print(f"    Max: {window_stats['max']:.2f}°C")

        # In a real job, we'd save the histogram back to the database
        # save_to_database(histogram)

        # Simulate serializing for storage in SQLite
        serialized = histogram.serialize()
        print(f"  Checkpoint size: {len(serialized)} bytes")

        # For simulation, we'll continue with the same histogram
        # In reality, each job would deserialize from the database

    print("\nStream processing workflow completed!")
    print(f"Final histogram memory usage: {histogram.estimate_size()} bytes")
    print(f"Total items processed: {histogram._items_processed}")


def demonstrate_pandas_integration():
    """
    Demonstrate how to use Exponential Histogram with pandas DataFrames.
    This shows how TinySift can complement pandas for streaming analytics.
    """
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("\n=== Pandas Integration Demo ===")
        print("Skipping demo: pandas not installed")
        return

    print("\n=== Pandas Integration Demo ===")
    print("Creating a time series DataFrame with sample data...")

    # Create a sample time series DataFrame
    # Simulating website traffic data over 48 hours
    start_date = datetime.now() - timedelta(days=2)
    dates = [
        start_date + timedelta(minutes=15 * i) for i in range(192)
    ]  # 15-min intervals for 48 hours

    # Generate traffic pattern with daily cycle and some randomness
    def traffic_pattern(dt):
        hour = dt.hour
        # Daily pattern: lowest at 3 AM, highest at 3 PM
        hour_factor = 1.0 - abs((hour - 15) % 24) / 12.0
        base_traffic = 100 + 900 * hour_factor
        # Add some random noise
        return int(base_traffic * random.uniform(0.8, 1.2))

    traffic = [traffic_pattern(d) for d in dates]

    # Create DataFrame
    df = pd.DataFrame({"timestamp": dates, "visitors": traffic})

    print(f"DataFrame shape: {df.shape}")
    print("\nData sample:")
    print(df.head())

    print("\nUsing Exponential Histogram to analyze the last 24 hours of data...")

    # Create histogram with 24-hour window
    histogram = ExponentialHistogram(
        window_size=24 * 60 * 60,  # 24 hours in seconds
        error_bounds=0.01,
        is_time_based=True,
    )

    # Process the DataFrame in chronological order
    print("Processing data chronologically...")
    for i, row in df.iterrows():
        # Convert timestamp to epoch seconds
        timestamp = int(row["timestamp"].timestamp())
        # Update histogram with visitor count
        histogram.update(value=row["visitors"], timestamp=timestamp)

        # Show periodic updates
        if i % 32 == 31:  # Every ~8 hours in our data
            window_stats = histogram.get_window_stats()
            time_str = row["timestamp"].strftime("%Y-%m-%d %H:%M")
            print(f"\nStatistics at {time_str} (processed {i+1} rows):")
            print(f"  Window count: {window_stats['count']}")
            print(f"  Avg visitors: {window_stats['average']:.1f}")

    # Compare with pandas calculations for the last 24 hours
    print("\nComparing with pandas calculations...")

    # Get the last 24 hours from the DataFrame
    cutoff_time = df["timestamp"].iloc[-1] - timedelta(hours=24)
    last_day = df[df["timestamp"] >= cutoff_time]

    pandas_count = len(last_day)
    pandas_sum = last_day["visitors"].sum()
    pandas_avg = last_day["visitors"].mean()

    # Get histogram calculations
    hist_stats = histogram.get_window_stats()

    print(f"Pandas calculations (last 24 hours):")
    print(f"  Count: {pandas_count}")
    print(f"  Sum: {pandas_sum}")
    print(f"  Average: {pandas_avg:.1f}")

    print(f"Histogram calculations (sliding window):")
    print(f"  Count: {hist_stats['count']}")
    print(f"  Sum: {hist_stats['sum']}")
    print(f"  Average: {hist_stats['average']:.1f}")

    # Calculate difference
    count_diff = abs(hist_stats["count"] - pandas_count)
    sum_diff_pct = abs(hist_stats["sum"] - pandas_sum) / pandas_sum * 100
    avg_diff_pct = abs(hist_stats["average"] - pandas_avg) / pandas_avg * 100

    print(f"\nDifferences:")
    print(f"  Count: {count_diff} readings")
    print(f"  Sum: {sum_diff_pct:.2f}%")
    print(f"  Average: {avg_diff_pct:.2f}%")

    print(f"\nMemory comparison:")
    print(f"  Pandas DataFrame: {sys.getsizeof(df)} bytes")
    print(f"  Histogram: {histogram.estimate_size()} bytes")


if __name__ == "__main__":
    demonstrate_basic_histogram()
    demonstrate_time_based_window()
    demonstrate_min_max_tracking()
    demonstrate_serialization()
    demonstrate_gigq_integration()
    demonstrate_pandas_integration()
