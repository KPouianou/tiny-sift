"""
Count-Min Sketch implementation for TinySift with enhanced benchmarking hooks.

This module provides the implementation of Count-Min Sketch, a probabilistic
data structure used for frequency estimation in data streams with bounded
memory usage.

The Count-Min Sketch provides the following guarantees:
1. Space Complexity: O(width * depth) where width and depth are parameters
2. Update Time: O(depth)
3. Query Time: O(depth)
4. Error Bound: With probability at least 1-delta, the error is at most epsilon * N,
   where N is the sum of all frequencies and epsilon and delta are functions of width and depth.

References:
    - Cormode, G., & Muthukrishnan, S. (2005). An improved data stream summary:
      The count-min sketch and its applications. Journal of Algorithms, 55(1), 58-75.
"""

import array
import collections
import math
import sys
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union, cast

from tiny_sift.core.base import FrequencyEstimator
from tiny_sift.core.hash import murmurhash3_32

T = TypeVar("T")  # Type for the items being processed


class CountMinSketch(FrequencyEstimator[T]):
    """
    Count-Min Sketch for frequency estimation in data streams.

    This implementation uses a 2D array of counters and multiple hash functions
    to estimate item frequencies with a bounded memory footprint. It provides
    an estimate that is always greater than or equal to the true frequency.

    The implementation includes comprehensive benchmarking hooks for monitoring
    the performance, accuracy, and internal state of the sketch.

    References:
        - Cormode, G., & Muthukrishnan, S. (2005). An improved data stream summary:
          The count-min sketch and its applications. Journal of Algorithms, 55(1), 58-75.
    """

    def __init__(
        self,
        width: int = 1024,
        depth: int = 5,
        memory_limit_bytes: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize a new Count-Min Sketch.

        Args:
            width: The number of counters per hash function (columns).
                  Larger values reduce the collision rate and improve accuracy.
                  Should be a large enough number to reduce hash collisions.
            depth: The number of hash functions (rows).
                  Larger values reduce the probability of error by using more hash functions.
                  Typically 5-10 is sufficient for most applications.
            memory_limit_bytes: Optional maximum memory usage in bytes.
            seed: Optional random seed for hash functions.

        Raises:
            ValueError: If width or depth is less than 1.
        """
        super().__init__(memory_limit_bytes)

        if width < 1:
            raise ValueError("Width must be at least 1")
        if depth < 1:
            raise ValueError("Depth must be at least 1")

        self._width = width
        self._depth = depth
        self._seed = seed if seed is not None else 0

        # Initialize the counter matrix as a 2D array
        # Using 64-bit unsigned integers (typecode 'L') to prevent overflow on large streams
        self._counters = [array.array("L", [0] * width) for _ in range(depth)]

        # Track total frequency for calculating heavy hitters
        self._total_frequency = 0

        # Calculate error bounds
        # epsilon: The error factor (with high probability, errors are less than epsilon * total_frequency)
        # delta: The probability of exceeding the error bound is at most delta
        self._epsilon = math.e / width
        self._delta = math.exp(-depth)

        # Initialize variables for tracking hash statistics
        self._enable_hash_tracking = False
        self._hash_position_counts = None
        self._hash_collision_stats = None
        self._recent_items_for_tests = None
        self._max_tracking_items = 1000  # Limit tracking to avoid memory issues

        # Initialize cached statistics
        self._cached_stats = None

    def update(self, item: T, count: int = 1) -> None:
        """
        Update the sketch with a new item from the stream.

        This method increments counters in each row at indices determined by
        hash functions applied to the item.

        Args:
            item: The item to add to the sketch.
            count: How many occurrences to add (default is 1).
                  Supports batch increments.

        Raises:
            ValueError: If count is negative.
        """
        if count < 0:
            raise ValueError("Count must be non-negative")

        if count == 0:
            return

        # Increment the items processed counter
        super().update(item)

        # Increment the total frequency count
        self._total_frequency += count

        # Update counters for each hash function
        for i in range(self._depth):
            # Get the bucket index for this hash function
            index = self._hash(item, i)

            # Track hash positions if enabled
            if self._enable_hash_tracking and self._hash_position_counts is not None:
                position_key = (i, index)
                self._hash_position_counts[position_key] = (
                    self._hash_position_counts.get(position_key, 0) + 1
                )

            # Increment the counter by count
            self._counters[i][index] += count

        # Store recent items for hash quality assessment if tracking is enabled
        if (
            self._enable_hash_tracking
            and self._recent_items_for_tests is not None
            and len(self._recent_items_for_tests) < self._max_tracking_items
        ):
            self._recent_items_for_tests.append(item)

        # Invalidate cached statistics
        self._cached_stats = None

    def _hash(self, item: T, row: int) -> int:
        """
        Hash function for determining counter indices.

        This method uses MurmurHash3 and combines the item with the row index and seed
        to create different hash functions for each row.

        Args:
            item: The item to hash.
            row: The row index (0 to depth-1).

        Returns:
            The counter index for the given item and row.
        """
        # Use the row index to create a different seed for each hash function
        row_seed = self._seed ^ (row * 0xBC9F1D34)

        # Apply MurmurHash3 to the item with the row-specific seed
        # This gives us a different hash function for each row
        item_hash = murmurhash3_32(item, seed=row_seed)

        # Map to the range [0, width-1]
        return item_hash % self._width

    def query(self, item: T) -> float:
        """
        Query the sketch for an item's estimated frequency.

        This is a convenience method that calls estimate_frequency.

        Args:
            item: The item to query.

        Returns:
            The estimated frequency of the item.
        """
        return self.estimate_frequency(item)

    def estimate_frequency(self, item: T) -> float:
        """
        Estimate the frequency of an item in the stream.

        This method applies the same hash functions used during updates
        and returns the minimum counter value as the estimate.

        Args:
            item: The item to estimate the frequency for.

        Returns:
            The estimated frequency of the item.
            This is guaranteed to be at least the true frequency,
            and with high probability close to the true frequency.
        """
        # The key insight of the Count-Min Sketch:
        # Take the minimum of all counter values to reduce the impact of hash collisions
        min_count = float("inf")

        for i in range(self._depth):
            # Get the bucket index for this hash function
            index = self._hash(item, i)

            # Update the minimum counter value seen
            min_count = min(min_count, self._counters[i][index])

        return min_count

    def estimate_frequency_error(self, item: T) -> Tuple[float, float]:
        """
        Estimate frequency with error bounds for an item.

        Args:
            item: The item to estimate the frequency for.

        Returns:
            A tuple of (estimated_frequency, max_error) where max_error is
            the maximum expected overestimation (epsilon * total_frequency).
        """
        frequency = self.estimate_frequency(item)

        # The maximum error is epsilon * total_frequency with probability 1-delta
        max_error = self._epsilon * self._total_frequency

        return (frequency, max_error)

    def get_heavy_hitters(self, threshold: float) -> Dict[T, float]:
        """
        Get items that appear more frequently than the threshold.

        Note: This method requires maintaining a list of all unique items seen,
        which is not typically done in a streaming context. This implementation
        returns an approximation by querying a set of items provided.

        Args:
            threshold: The minimum frequency ratio (0.0 to 1.0) to include.
                      For example, 0.01 means items appearing in at least 1% of the stream.

        Returns:
            A dictionary mapping items to their estimated frequencies.
            Only includes items whose estimated frequency exceeds the threshold * total_frequency.

        Raises:
            ValueError: If threshold is not between 0 and 1.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        # This method typically needs a set of candidate items to check
        # In a real implementation, one would pass in these candidates
        # or use an additional sketch to track them
        # For this implementation, we'll log a warning
        raise NotImplementedError(
            "This implementation does not track unique items in the stream. "
            "Use the 'get_heavy_hitters_from_candidates' method instead."
        )

    def get_heavy_hitters_from_candidates(
        self, candidates: List[T], threshold: float
    ) -> Dict[T, float]:
        """
        Get heavy hitters from a list of candidate items.

        Args:
            candidates: List of candidate items to check.
            threshold: The minimum frequency ratio (0.0 to 1.0) to include.

        Returns:
            A dictionary mapping heavy hitter items to their estimated frequencies.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        # Calculate the minimum count threshold
        min_count = threshold * self._total_frequency

        # Check each candidate item
        heavy_hitters = {}
        for item in candidates:
            freq = self.estimate_frequency(item)
            if freq >= min_count:
                heavy_hitters[item] = freq

        return heavy_hitters

    def merge(self, other: "CountMinSketch[T]") -> "CountMinSketch[T]":
        """
        Merge this sketch with another Count-Min Sketch.

        Args:
            other: Another Count-Min Sketch.

        Returns:
            A new merged Count-Min Sketch.

        Raises:
            TypeError: If other is not a CountMinSketch.
            ValueError: If sketches have different dimensions.
        """
        self._check_same_type(other)

        # Check compatible dimensions
        if self._width != other._width or self._depth != other._depth:
            raise ValueError(
                f"Cannot merge sketches with different dimensions: "
                f"({self._width}x{self._depth}) and ({other._width}x{other._depth})"
            )

        # Create a new sketch with the same dimensions
        result = CountMinSketch[T](
            width=self._width,
            depth=self._depth,
            memory_limit_bytes=self._memory_limit_bytes,
            seed=self._seed,
        )

        # Combine the counters by adding them
        for i in range(self._depth):
            for j in range(self._width):
                result._counters[i][j] = self._counters[i][j] + other._counters[i][j]

        # Update total frequency and items processed
        result._total_frequency = self._total_frequency + other._total_frequency
        result._items_processed = self._combine_items_processed(other)

        # Invalidate cached statistics
        result._cached_stats = None

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the sketch to a dictionary for serialization.

        Returns:
            A dictionary representation of the sketch.
        """
        data = self._base_dict()
        data.update(
            {
                "width": self._width,
                "depth": self._depth,
                "seed": self._seed,
                "counters": [list(row) for row in self._counters],
                "total_frequency": self._total_frequency,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CountMinSketch[T]":
        """
        Create a sketch from a dictionary representation.

        Args:
            data: The dictionary containing the sketch state.

        Returns:
            A new Count-Min Sketch initialized with the given state.
        """
        # Create a new sketch
        sketch = cls(
            width=data["width"],
            depth=data["depth"],
            memory_limit_bytes=data.get("memory_limit_bytes"),
            seed=data.get("seed", 0),
        )

        # Restore counter values
        for i, row in enumerate(data["counters"]):
            sketch._counters[i] = array.array(
                "L", row
            )  # Use "L" for unsigned long, matching initialization

        # Restore other state
        sketch._total_frequency = data["total_frequency"]
        sketch._items_processed = data["items_processed"]

        # Invalidate cached statistics
        sketch._cached_stats = None

        return sketch

    def estimate_size(self) -> int:
        """
        Estimate the current memory usage of this sketch in bytes.

        This method provides a detailed breakdown of the memory used by
        the Count-Min Sketch, including the counter arrays, tracking structures,
        and overhead.

        Returns:
            Estimated memory usage in bytes.
        """
        # Base size of the object
        size = sys.getsizeof(self)

        # Size of the counter arrays
        counters_size = 0
        for row in self._counters:
            counters_size += sys.getsizeof(row)

        # Size of the list containing the rows
        counters_list_size = sys.getsizeof(self._counters)

        # Size of tracking structures if enabled
        tracking_size = 0
        if self._enable_hash_tracking:
            if self._hash_position_counts is not None:
                tracking_size += sys.getsizeof(self._hash_position_counts)
                # Add estimated size of keys and values
                tracking_size += len(self._hash_position_counts) * (
                    sys.getsizeof((0, 0)) + sys.getsizeof(0)
                )

            if self._hash_collision_stats is not None:
                tracking_size += sys.getsizeof(self._hash_collision_stats)

            if self._recent_items_for_tests is not None:
                tracking_size += sys.getsizeof(self._recent_items_for_tests)
                # Add estimated size of stored items (approximate)
                if self._recent_items_for_tests:
                    avg_item_size = sum(
                        sys.getsizeof(item)
                        for item in self._recent_items_for_tests[
                            : min(10, len(self._recent_items_for_tests))
                        ]
                    ) / min(10, len(self._recent_items_for_tests))
                    tracking_size += len(self._recent_items_for_tests) * avg_item_size

        # Size of cached statistics if present
        cached_stats_size = (
            sys.getsizeof(self._cached_stats) if self._cached_stats is not None else 0
        )

        # Total size
        total_size = (
            size
            + counters_size
            + counters_list_size
            + tracking_size
            + cached_stats_size
        )

        return total_size

    def error_bounds(self) -> Dict[str, float]:
        """
        Calculate the theoretical and observed error bounds for this sketch.

        This method provides detailed error bound information specific to Count-Min Sketch,
        including the epsilon and delta parameters, as well as observed error metrics
        based on the current state of the sketch.

        Returns:
            A dictionary with the error bounds:
            - epsilon: The error factor (errors are less than epsilon * total_frequency)
            - delta: The probability of exceeding the error bound
            - max_absolute_error: The maximum absolute error (epsilon * total_frequency)
            - max_relative_error: The maximum relative error as a fraction of total items
            - observed_saturation: The proportion of counters that are non-zero
            - estimated_collision_rate: Estimated hash collision rate
        """
        # Calculate theoretical error bounds
        bounds = {
            "epsilon": self._epsilon,
            "delta": self._delta,
            "max_absolute_error": self._epsilon * self._total_frequency,
        }

        # Add relative error as percentage of total frequency
        if self._total_frequency > 0:
            bounds["max_relative_error"] = self._epsilon
        else:
            bounds["max_relative_error"] = 0.0

        # Calculate observed saturation (proportion of non-zero counters)
        non_zero_counters = 0
        total_counters = self._width * self._depth

        for row in self._counters:
            non_zero_counters += sum(1 for val in row if val > 0)

        bounds["observed_saturation"] = (
            non_zero_counters / total_counters if total_counters > 0 else 0.0
        )

        # Estimate collision rate based on occupancy (Birthday paradox approximation)
        # For a hash function with width slots and n items, probability of no collision is approximately:
        # p ≈ exp(-n(n-1)/(2*width))
        if self._items_processed > 0:
            n = min(
                self._items_processed, self._width
            )  # Cap at width for realistic estimation
            no_collision_prob = math.exp(-n * (n - 1) / (2 * self._width))
            bounds["estimated_collision_rate"] = 1.0 - no_collision_prob
        else:
            bounds["estimated_collision_rate"] = 0.0

        # Calculate expected error based on current load and theoretical properties
        if self._total_frequency > 0:
            # Expected error increases with saturation
            load_factor = min(1.0, self._items_processed / self._width)
            bounds["expected_error_multiplier"] = (
                1.0 + load_factor
            )  # Adjust based on load
        else:
            bounds["expected_error_multiplier"] = 1.0

        return bounds

    def get_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the current state of the Count-Min Sketch.

        This method extends the base class implementation to include sketch-specific
        statistics such as counter distribution, collision rates, and saturation levels.

        Returns:
            A dictionary containing various statistics about the sketch.
        """
        # Use cached stats if available
        if self._cached_stats is not None:
            return self._cached_stats.copy()

        # Start with base class statistics
        stats = super().get_stats()

        # Add Count-Min Sketch specific parameters
        stats.update(
            {
                "width": self._width,
                "depth": self._depth,
                "seed": self._seed,
                "total_frequency": self._total_frequency,
            }
        )

        # Add error bounds
        error_bounds = self.error_bounds()
        stats.update(error_bounds)

        # Calculate counter statistics
        counter_stats = self._calculate_counter_stats()
        stats.update(counter_stats)

        # Add estimated memory breakdown
        memory_stats = self._calculate_memory_breakdown()
        stats["memory_breakdown"] = memory_stats

        # Add hash quality metrics if hash tracking is enabled
        if self._enable_hash_tracking and self._hash_position_counts is not None:
            hash_quality = self._assess_hash_quality()
            stats["hash_quality"] = hash_quality

        # Cache the stats
        self._cached_stats = stats.copy()

        return stats

    def _calculate_counter_stats(self) -> Dict[str, Any]:
        """
        Calculate statistics about the counter values.

        Returns:
            A dictionary with counter statistics.
        """
        stats = {}

        # Initialize counters
        zero_counters = 0
        total_counters = self._width * self._depth
        counter_sum = 0
        max_counter = 0
        min_counter = float("inf") if total_counters > 0 else 0

        # Counter value distribution
        counter_distribution = {}

        # Process each counter
        for row in self._counters:
            for val in row:
                if val == 0:
                    zero_counters += 1
                else:
                    # Update min/max
                    max_counter = max(max_counter, val)
                    min_counter = min(min_counter, val)

                # Track distribution (bin into ranges for efficiency)
                bin_key = self._get_distribution_bin(val)
                counter_distribution[bin_key] = counter_distribution.get(bin_key, 0) + 1

                # Update sum
                counter_sum += val

        # Calculate statistics
        stats["total_counters"] = total_counters
        stats["zero_counters"] = zero_counters
        stats["non_zero_counters"] = total_counters - zero_counters
        stats["saturation"] = (
            (total_counters - zero_counters) / total_counters
            if total_counters > 0
            else 0.0
        )
        stats["max_counter"] = max_counter
        stats["min_non_zero_counter"] = (
            min_counter if min_counter != float("inf") else 0
        )
        stats["average_counter"] = (
            counter_sum / total_counters if total_counters > 0 else 0.0
        )
        stats["counter_distribution"] = counter_distribution

        # Estimate number of distinct items based on the number of non-zero counters
        # This is a very rough estimate assuming random distribution
        if total_counters > 0:
            occupancy_rate = (total_counters - zero_counters) / total_counters
            # This is an approximation based on the coupon collector's problem
            if occupancy_rate > 0 and occupancy_rate < 1:
                stats["estimated_distinct_items"] = int(
                    -self._width * math.log(1 - occupancy_rate)
                )
            else:
                stats["estimated_distinct_items"] = self._items_processed
        else:
            stats["estimated_distinct_items"] = 0

        return stats

    def _get_distribution_bin(self, value: int) -> str:
        """
        Group counter values into distribution bins.

        Args:
            value: The counter value.

        Returns:
            A string representing the bin label.
        """
        if value == 0:
            return "0"
        elif value == 1:
            return "1"
        elif value <= 5:
            return "2-5"
        elif value <= 10:
            return "6-10"
        elif value <= 50:
            return "11-50"
        elif value <= 100:
            return "51-100"
        elif value <= 500:
            return "101-500"
        elif value <= 1000:
            return "501-1000"
        else:
            # For very large values, use logarithmic binning
            power = math.floor(math.log10(value))
            lower = 10**power
            upper = 10 ** (power + 1) - 1
            return f"{lower}-{upper}"

    def _calculate_memory_breakdown(self) -> Dict[str, int]:
        """
        Calculate a detailed breakdown of memory usage.

        Returns:
            A dictionary with memory usage by component.
        """
        # Base object size
        base_size = sys.getsizeof(self)

        # Counter arrays
        counters_size = 0
        for row in self._counters:
            counters_size += sys.getsizeof(row)

        # Counter list container
        counter_list_size = sys.getsizeof(self._counters)

        # Dictionary and other attributes
        attributes_size = 0
        for attr in vars(self):
            if (
                attr != "_counters"
                and not attr.startswith("_hash_")
                and attr != "_cached_stats"
            ):
                value = getattr(self, attr)
                attributes_size += sys.getsizeof(value)

        # Tracking structures
        tracking_size = 0
        if self._enable_hash_tracking:
            if self._hash_position_counts is not None:
                tracking_size += sys.getsizeof(self._hash_position_counts)
                # Approximation for dictionary entries
                if self._hash_position_counts:
                    # Take a sample of keys and values to estimate average size
                    sample_keys = list(self._hash_position_counts.keys())[
                        : min(10, len(self._hash_position_counts))
                    ]
                    if sample_keys:
                        key_size = sum(sys.getsizeof(k) for k in sample_keys) / len(
                            sample_keys
                        )
                        value_size = sum(
                            sys.getsizeof(self._hash_position_counts[k])
                            for k in sample_keys
                        ) / len(sample_keys)
                        tracking_size += len(self._hash_position_counts) * (
                            key_size + value_size
                        )

            if self._hash_collision_stats is not None:
                tracking_size += sys.getsizeof(self._hash_collision_stats)

            if self._recent_items_for_tests is not None:
                tracking_size += sys.getsizeof(self._recent_items_for_tests)
                # Add estimated size of contained items
                if self._recent_items_for_tests:
                    sample_count = min(10, len(self._recent_items_for_tests))
                    if sample_count > 0:
                        sample_size = sum(
                            sys.getsizeof(item)
                            for item in self._recent_items_for_tests[:sample_count]
                        )
                        avg_item_size = sample_size / sample_count
                        tracking_size += (
                            len(self._recent_items_for_tests) * avg_item_size
                        )

        # Cached stats size
        cached_stats_size = 0
        if self._cached_stats is not None:
            cached_stats_size = sys.getsizeof(self._cached_stats)
            # Approximation for dictionary entries
            if self._cached_stats:
                # Take a sample of keys and values to estimate average size
                sample_keys = list(self._cached_stats.keys())[
                    : min(10, len(self._cached_stats))
                ]
                if sample_keys:
                    key_size = sum(sys.getsizeof(k) for k in sample_keys) / len(
                        sample_keys
                    )
                    # Values can be complex objects, just use a rough estimate
                    value_size = 100  # Rough estimate per value
                    cached_stats_size += len(self._cached_stats) * (
                        key_size + value_size
                    )

        total = (
            base_size
            + counters_size
            + counter_list_size
            + attributes_size
            + tracking_size
            + cached_stats_size
        )

        return {
            "base_object": base_size,
            "counter_arrays": counters_size,
            "counter_container": counter_list_size,
            "attributes": attributes_size,
            "tracking_structures": tracking_size,
            "cached_stats": cached_stats_size,
            "total": total,
        }

    def clear(self) -> None:
        """
        Reset the sketch to its initial state.

        This method clears all counters but keeps the same dimensions.
        """
        for i in range(self._depth):
            for j in range(self._width):
                self._counters[i][j] = 0

        self._total_frequency = 0
        self._items_processed = 0

        # Reset hash tracking if enabled
        if self._enable_hash_tracking:
            if self._hash_position_counts is not None:
                self._hash_position_counts.clear()
            if self._hash_collision_stats is not None:
                self._hash_collision_stats.clear()
            if self._recent_items_for_tests is not None:
                self._recent_items_for_tests.clear()

        # Invalidate cached statistics
        self._cached_stats = None

    def enable_hash_tracking(
        self, track_items: bool = True, max_items: int = 1000
    ) -> None:
        """
        Enable tracking of hash function behavior for quality assessment.

        This adds some memory and performance overhead, so it should be used
        primarily for debugging and benchmarking.

        Args:
            track_items: Whether to track recent items for detailed analysis.
            max_items: Maximum number of items to track.
        """
        self._enable_hash_tracking = True
        self._hash_position_counts = {}
        self._hash_collision_stats = {}

        if track_items:
            self._recent_items_for_tests = []
            self._max_tracking_items = max_items

        # Invalidate cached statistics
        self._cached_stats = None

    def disable_hash_tracking(self) -> None:
        """
        Disable hash tracking to reduce overhead.
        """
        self._enable_hash_tracking = False
        self._hash_position_counts = None
        self._hash_collision_stats = None
        self._recent_items_for_tests = None

        # Invalidate cached statistics
        self._cached_stats = None

    def _assess_hash_quality(self) -> Dict[str, Any]:
        """
        Assess the quality of the hash functions based on tracking data.

        Returns:
            A dictionary with hash quality metrics.
        """
        if not self._enable_hash_tracking or self._hash_position_counts is None:
            return {"tracking_enabled": False}

        stats = {"tracking_enabled": True}

        # Calculate collision metrics
        if self._hash_position_counts:
            # Analyze position utilization
            positions_hit = len(self._hash_position_counts)
            total_positions = self._width * self._depth
            utilization = positions_hit / total_positions if total_positions > 0 else 0

            # Analyze position distribution
            position_counts = list(self._hash_position_counts.values())
            max_hits = max(position_counts) if position_counts else 0
            avg_hits = (
                sum(position_counts) / len(position_counts) if position_counts else 0
            )

            # Calculate variance and coefficient of variation
            variance = (
                sum((x - avg_hits) ** 2 for x in position_counts) / len(position_counts)
                if position_counts
                else 0
            )
            std_dev = math.sqrt(variance) if variance > 0 else 0
            cv = std_dev / avg_hits if avg_hits > 0 else 0

            # Add metrics to stats
            stats["positions_hit"] = positions_hit
            stats["total_positions"] = total_positions
            stats["position_utilization"] = utilization
            stats["max_position_hits"] = max_hits
            stats["avg_position_hits"] = avg_hits
            stats["position_hit_stddev"] = std_dev
            stats["position_hit_cv"] = (
                cv  # Coefficient of variation - lower is more uniform
            )

            # Calculate per-row statistics if there's enough data
            row_stats = {}
            for row in range(self._depth):
                row_hits = {
                    pos: count
                    for (r, pos), count in self._hash_position_counts.items()
                    if r == row
                }
                if row_hits:
                    positions_used = len(row_hits)
                    coverage = positions_used / self._width if self._width > 0 else 0
                    row_stats[f"row_{row}"] = {
                        "positions_used": positions_used,
                        "coverage": coverage,
                        "max_hits": max(row_hits.values()) if row_hits else 0,
                    }

            stats["row_statistics"] = row_stats

        # Assess hash uniformity with chi-square test if we have stored items
        if self._recent_items_for_tests and len(self._recent_items_for_tests) >= 10:
            uniformity_stats = self._calculate_uniformity_stats()
            stats.update(uniformity_stats)

        return stats

    def _calculate_uniformity_stats(self) -> Dict[str, Any]:
        """
        Calculate statistics about hash function uniformity.

        Returns:
            A dictionary with uniformity statistics.
        """
        stats = {}

        # Get sample of items for analysis
        items = self._recent_items_for_tests[: self._max_tracking_items]

        # Generate hash values for each item and row
        hash_distributions = [[] for _ in range(self._depth)]
        for item in items:
            for row in range(self._depth):
                hash_distributions[row].append(self._hash(item, row))

        # Calculate uniformity metrics for each row
        row_uniformity = {}
        for row, hash_values in enumerate(hash_distributions):
            # Calculate bin frequencies (simplified bin count)
            bin_count = min(20, self._width)
            bin_size = self._width // bin_count

            # Count items in each bin
            bins = [0] * bin_count
            for val in hash_values:
                bin_idx = min(bin_count - 1, val // bin_size)
                bins[bin_idx] += 1

            # Calculate chi-square statistic
            expected = len(hash_values) / bin_count
            chi_square = sum((observed - expected) ** 2 / expected for observed in bins)

            # Calculate p-value (approximate) - degrees of freedom = bin_count - 1
            # Lower p-value indicates less uniform distribution
            # Note: This is a simplified calculation and should not be used for formal hypothesis testing
            df = bin_count - 1
            p_value = 1.0  # Default high p-value (good uniformity)

            if chi_square > df:
                # Simple approximation: p-value decreases as chi-square increases above df
                p_value = math.exp(-(chi_square - df) / df)

            # Calculate entropy of the distribution (higher is more uniform)
            probs = [count / len(hash_values) for count in bins if count > 0]
            entropy = -sum(p * math.log2(p) for p in probs)
            max_entropy = math.log2(bin_count)  # Maximum possible entropy
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

            row_uniformity[f"row_{row}"] = {
                "chi_square": chi_square,
                "uniformity_p_value": p_value,
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
                "num_samples": len(hash_values),
            }

        stats["uniformity_by_row"] = row_uniformity

        # Overall uniformity assessment
        stats["uniformity_assessment"] = {
            "entropy_avg": sum(
                info["normalized_entropy"] for info in row_uniformity.values()
            )
            / self._depth,
            "samples_analyzed": len(items),
        }

        return stats

    def analyze_performance(self) -> Dict[str, Any]:
        """
        Perform a detailed analysis of the Count-Min Sketch's performance.

        This method provides insights into the sketch's efficiency, accuracy,
        and collision characteristics based on its current state.

        Returns:
            A dictionary containing performance analysis metrics.
        """
        analysis = {
            "algorithm": "Count-Min Sketch",
            "memory_efficiency": {
                "bits_per_item": (self.estimate_size() * 8)
                / max(1, self._items_processed),
                "total_bytes": self.estimate_size(),
                "counter_bytes": sum(sys.getsizeof(row) for row in self._counters),
                "bits_per_counter": 64,  # Using 64-bit counters (unsigned long)
                "counters_per_item": (self._width * self._depth)
                / max(1, self._items_processed),
            },
            "accuracy": {
                "width": self._width,
                "depth": self._depth,
                "epsilon": self._epsilon,
                "delta": self._delta,
                "error_factor": self._epsilon * self._total_frequency,
                "confidence": 1.0 - self._delta,
            },
            "saturation": {
                "total_counters": self._width * self._depth,
                "non_zero_counters": sum(
                    sum(1 for c in row if c > 0) for row in self._counters
                ),
                "saturation_ratio": (
                    sum(sum(1 for c in row if c > 0) for row in self._counters)
                    / (self._width * self._depth)
                    if (self._width * self._depth) > 0
                    else 0
                ),
            },
        }

        # Calculate counter value distribution
        counter_values = [c for row in self._counters for c in row]
        counter_stats = {
            "min": min(counter_values) if counter_values else 0,
            "max": max(counter_values) if counter_values else 0,
            "mean": sum(counter_values) / len(counter_values) if counter_values else 0,
        }

        if counter_values:
            # Calculate standard deviation
            mean = counter_stats["mean"]
            variance = sum((x - mean) ** 2 for x in counter_values) / len(
                counter_values
            )
            counter_stats["stddev"] = math.sqrt(variance)

            # Calculate median
            sorted_values = sorted(counter_values)
            mid = len(sorted_values) // 2
            if len(sorted_values) % 2 == 0:
                counter_stats["median"] = (
                    sorted_values[mid - 1] + sorted_values[mid]
                ) / 2
            else:
                counter_stats["median"] = sorted_values[mid]

        analysis["counter_statistics"] = counter_stats

        # Add recommendations based on the analysis
        analysis["recommendations"] = []

        # Check if sketch is too saturated
        if analysis["saturation"]["saturation_ratio"] > 0.8:
            analysis["recommendations"].append(
                "High saturation detected (>80%). Consider increasing width to reduce collisions."
            )

        # Check if width is too small relative to items
        if self._items_processed > 0 and (self._width / self._items_processed) < 0.5:
            analysis["recommendations"].append(
                f"Width ({self._width}) may be too small for the number of items processed ({self._items_processed}). "
                f"Consider width ≥ 2*items for better accuracy."
            )

        # Check if depth might be excessive
        if self._depth > 10 and analysis["saturation"]["saturation_ratio"] < 0.3:
            analysis["recommendations"].append(
                f"Depth ({self._depth}) may be unnecessarily high given the current saturation. "
                f"Consider reducing depth to improve performance."
            )

        # Check if memory usage is efficient
        bits_per_item = analysis["memory_efficiency"]["bits_per_item"]
        if bits_per_item > 100 and self._items_processed > 1000:
            analysis["recommendations"].append(
                f"High memory usage per item ({bits_per_item:.1f} bits/item). "
                f"Consider adjusting width/depth ratio for better memory efficiency."
            )

        return analysis

    @classmethod
    def create_from_error_rate(
        cls,
        epsilon: float,
        delta: float,
        memory_limit_bytes: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> "CountMinSketch[T]":
        """
        Create a Count-Min Sketch with the desired error guarantees.

        Args:
            epsilon: The error factor (errors will be less than epsilon * total_frequency)
            delta: The probability of exceeding the error bound
            memory_limit_bytes: Optional maximum memory usage in bytes
            seed: Optional random seed for hash functions

        Returns:
            A new Count-Min Sketch configured for the specified error bounds

        Raises:
            ValueError: If epsilon or delta are out of range (must be between 0 and 1)
        """
        if not (0 < epsilon < 1):
            raise ValueError("Epsilon must be between 0 and 1")
        if not (0 < delta < 1):
            raise ValueError("Delta must be between 0 and 1")

        # Calculate width and depth from epsilon and delta
        # Width is e/epsilon (where e is the mathematical constant)
        # Depth is ln(1/delta)
        width = math.ceil(math.e / epsilon)
        depth = math.ceil(math.log(1 / delta))

        return cls(
            width=width, depth=depth, memory_limit_bytes=memory_limit_bytes, seed=seed
        )

    @classmethod
    def create_from_memory_limit(
        cls,
        memory_bytes: int,
        epsilon_delta_ratio: float = 0.1,
        seed: Optional[int] = None,
    ) -> "CountMinSketch[T]":
        """
        Create a Count-Min Sketch optimized for a given memory limit.

        This method calculates the optimal width and depth parameters to maximize
        accuracy within the specified memory constraint.

        Args:
            memory_bytes: Maximum memory usage in bytes
            epsilon_delta_ratio: Ratio between epsilon and delta for error tradeoff
                               Lower values prioritize width over depth
            seed: Optional random seed for hash functions

        Returns:
            A new Count-Min Sketch optimized for the memory constraint

        Raises:
            ValueError: If memory_bytes is too small for a useful sketch
        """
        if memory_bytes <= 0:
            raise ValueError("Memory limit must be positive")

        # Estimate base size and overhead (conservative estimate)
        base_size = sys.getsizeof(object()) + 300  # Add extra buffer for overhead

        # Calculate space available for counters (in bytes)
        # Be conservative and reserve some space for object overhead that's hard to estimate
        available_bytes = max(1, memory_bytes - base_size - 500)

        # Each counter is an unsigned long (typically 8 bytes on 64-bit systems)
        counter_size = 8  # bytes

        # Maximum number of counters we can fit
        max_counters = available_bytes // counter_size

        if max_counters < 2:
            raise ValueError(
                f"Memory limit {memory_bytes} bytes is too small for a useful sketch"
            )

        # Determine optimal width/depth ratio based on epsilon/delta ratio
        # For Count-Min Sketch:
        # - width determines epsilon (error magnitude): width = e/epsilon
        # - depth determines delta (error probability): depth = ln(1/delta)

        # The total number of counters = width * depth
        # We want to optimize the width and depth such that:
        # 1. width * depth <= max_counters
        # 2. We achieve the best error bounds

        # Simplistically, if we want the ratio epsilon/delta = r:
        # ln(1/delta) = r * epsilon = r * e/width
        # depth = r * e/width

        # So width * depth = width * (r * e/width) = r * e
        # This isn't accurate for actual memory optimization, so we'll solve numerically

        # Start with an initial depth estimate
        depth = max(1, int(math.sqrt(max_counters * epsilon_delta_ratio / math.e)))

        # Calculate corresponding width
        # Be conservative and use 95% of max_counters to ensure we stay within limit
        width = int(0.95 * max_counters) // depth

        # Fine-tune for optimal error bounds within constraint
        best_width = width
        best_depth = depth
        best_error = float("inf")

        # Try a few depth values around the initial estimate
        for d in range(max(1, depth - 2), depth + 3):
            w = int(0.95 * max_counters) // d
            if w * d > int(0.95 * max_counters):
                continue

            # Calculate error bounds for this configuration
            epsilon = math.e / w
            delta = math.exp(-d)

            # Combined error metric (weighted by epsilon_delta_ratio)
            error = epsilon + delta * epsilon_delta_ratio

            if error < best_error:
                best_error = error
                best_width = w
                best_depth = d

        # Create the optimized sketch
        return cls(
            width=best_width,
            depth=best_depth,
            memory_limit_bytes=memory_bytes,
            seed=seed,
        )
