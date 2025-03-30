"""
GigQ Integration Patterns for HyperLogLog

This example demonstrates patterns for using HyperLogLog with GigQ
while maintaining zero dependencies between the packages.

Note: This requires both TinySift and GigQ to be installed.
"""

# Import note: These examples use conditional imports to avoid
# hard dependencies between packages
try:
    from gigq import Job, JobQueue

    GIGQ_AVAILABLE = True
except ImportError:
    GIGQ_AVAILABLE = False
    print("GigQ not available. Install with: pip install gigq")

from tiny_sift.algorithms.hyperloglog import HyperLogLog


def hyperloglog_processor(data, estimator_state=None, precision=12):
    """
    Example processor function for use in GigQ jobs.

    Args:
        data: The data to process (list or iterable)
        estimator_state: Optional serialized HyperLogLog state from previous run
        precision: Precision to use if creating a new estimator

    Returns:
        Dict with serialized estimator and results
    """
    # Create or restore the HyperLogLog estimator
    if estimator_state:
        estimator = HyperLogLog.deserialize(estimator_state)
    else:
        estimator = HyperLogLog(precision=precision)

    # Process the data
    for item in data:
        estimator.update(item)

    # Return serialized state and results
    return {
        "state": estimator.serialize(),
        "cardinality": estimator.estimate_cardinality(),
        "items_processed": estimator.items_processed,
        "memory_bytes": estimator.estimate_size(),
    }


def demonstrate_gigq_integration():
    """
    Demonstrate GigQ integration patterns using HyperLogLog.
    """
    if not GIGQ_AVAILABLE:
        print("This example requires GigQ to be installed")
        return

    print("\n=== GigQ Integration Patterns ===")

    # Create a temporary in-memory database for the example
    queue = JobQueue(":memory:")

    # Create example data
    data_chunks = [
        [f"user-{i}" for i in range(100)],
        [f"user-{i}" for i in range(50, 150)],  # 50 overlap with chunk 1
        [f"user-{i}" for i in range(120, 220)],  # 30 overlap with chunk 2
    ]

    print(f"Processing {len(data_chunks)} data chunks")

    # Pattern 1: Sequential Processing with State Passing
    print("\nPattern 1: Sequential Processing with State Passing")

    # Initial job with no previous state
    initial_job = Job(
        name="hyperloglog_initial",
        function=hyperloglog_processor,
        params={"data": data_chunks[0]},
    )

    job_id = queue.submit(initial_job)
    queue.process_one()  # Simulate worker processing the job

    # Get result from first job
    result = queue.get_result(job_id)
    print(f"  Chunk 1: {result['cardinality']} unique items")

    # Process next chunk with previous state
    for i, chunk in enumerate(data_chunks[1:], 2):
        next_job = Job(
            name=f"hyperloglog_chunk_{i}",
            function=hyperloglog_processor,
            params={"data": chunk, "estimator_state": result["state"]},
        )

        job_id = queue.submit(next_job)
        queue.process_one()  # Simulate worker processing the job

        # Get updated result
        result = queue.get_result(job_id)
        print(f"  Chunk {i}: {result['cardinality']} unique items")

    # Pattern 2: Parallel Processing with Merge Job
    print("\nPattern 2: Parallel Processing with Merge Job")

    # Process chunks in parallel
    parallel_job_ids = []
    for i, chunk in enumerate(data_chunks, 1):
        job = Job(
            name=f"hyperloglog_parallel_{i}",
            function=hyperloglog_processor,
            params={"data": chunk},
        )

        job_id = queue.submit(job)
        queue.process_one()  # Simulate worker processing the job
        parallel_job_ids.append(job_id)

        result = queue.get_result(job_id)
        print(f"  Parallel chunk {i}: {result['cardinality']} unique items")

    # Merging function for combining multiple estimators
    def merge_estimators(estimator_states):
        """Merge multiple serialized HyperLogLog estimators."""
        if not estimator_states:
            return {"error": "No estimators to merge"}

        # Deserialize the first estimator
        merged = HyperLogLog.deserialize(estimator_states[0])

        # Merge the rest
        for state in estimator_states[1:]:
            other = HyperLogLog.deserialize(state)
            merged = merged.merge(other)

        return {
            "state": merged.serialize(),
            "cardinality": merged.estimate_cardinality(),
            "memory_bytes": merged.estimate_size(),
        }

    # Get all results
    estimator_states = [
        queue.get_result(job_id)["state"] for job_id in parallel_job_ids
    ]

    # Create merge job
    merge_job = Job(
        name="hyperloglog_merge",
        function=merge_estimators,
        params={"estimator_states": estimator_states},
    )

    merge_job_id = queue.submit(merge_job)
    queue.process_one()  # Simulate worker processing the job

    # Get merged result
    merged_result = queue.get_result(merge_job_id)
    print(f"  Merged result: {merged_result['cardinality']} unique items")

    # Calculate true unique count for verification
    all_items = set()
    for chunk in data_chunks:
        all_items.update(chunk)

    print(f"  True unique count: {len(all_items)}")

    # Pattern 3: Time-based Estimation with Results Table
    print("\nPattern 3: Time-based Estimation with Results Table")

    # In a real application, you would create a table to store the results
    # Here we'll simulate this with a simple dictionary
    tracking_results = {}

    # Create data for 3 time periods
    time_periods = ["2025-01-01", "2025-01-02", "2025-01-03"]

    for period in time_periods:
        # Get or create the estimator for this period
        period_state = tracking_results.get(period, {}).get("state")

        # Create a job to process this period's data
        period_job = Job(
            name=f"hyperloglog_{period}",
            function=hyperloglog_processor,
            params={
                "data": [f"{period}_user_{i}" for i in range(100)],
                "estimator_state": period_state,
            },
        )

        job_id = queue.submit(period_job)
        queue.process_one()  # Simulate worker processing the job

        # Store the result
        tracking_results[period] = queue.get_result(job_id)
        print(
            f"  Period {period}: {tracking_results[period]['cardinality']} unique items"
        )

    print(
        "\nThis example demonstrates three key patterns for using HyperLogLog with GigQ:"
    )
    print("1. Sequential processing with state passing between jobs")
    print("2. Parallel processing with a final merge job")
    print("3. Time-based tracking of unique items")
    print("\nThese patterns can be adapted for your specific use cases while")
    print("maintaining zero dependencies between TinySift and GigQ.")


if __name__ == "__main__":
    demonstrate_gigq_integration()
