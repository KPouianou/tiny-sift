"""
Exponential Histogram implementation for TinySift.

This module provides an implementation of the Exponential Histogram algorithm,
a probabilistic data structure for maintaining statistics over a sliding window
in a data stream using limited memory.

The implementation follows the DGIM (Datar-Gionis-Indyk-Motwani) algorithm with
deterministic guarantees and supports both time-based and count-based windows.

References:
    - Datar, M., Gionis, A., Indyk, P., & Motwani, R. (2002).
      Maintaining stream statistics over sliding windows.
      SIAM Journal on Computing, 31(6), 1794-1813.
"""

import math
import sys
import time
from bisect import insort
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple, TypeVar, Union, cast
from typing import Union

from tiny_sift.core.base import WindowAggregator

T = TypeVar("T")  # Type for the items being processed


@dataclass
class Bucket:
    """
    A bucket in the exponential histogram.

    Attributes:
        timestamp: The oldest timestamp in the bucket (for time-based windows)
                  or the sequence number (for count-based windows)
        size: The number of items in the bucket
        sum_value: The sum of the values in the bucket (used for sum estimation)
        min_value: The minimum value in the bucket (if tracking min)
        max_value: The maximum value in the bucket (if tracking max)
    """

    timestamp: Union[int, float]  # Allow float for time-based
    size: int = 1
    sum_value: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class ExponentialHistogram(WindowAggregator[T]):
    """
    Exponential Histogram for sliding window statistics.

    Uses buckets of exponentially increasing sizes to maintain approximate
    counts, sums, and other statistics over a sliding window. Smaller buckets
    represent more recent data for better precision, while older data is
    stored in larger buckets to save memory.

    Guarantees that all estimates are within a factor of (1 ± ε) of the true value,
    where ε is the configured error bound.
    """

    def __init__(
        self,
        window_size: int = 10000,
        error_bounds: float = 0.01,
        memory_limit_bytes: Optional[int] = None,
        is_time_based: bool = False,
        timestamp_unit: str = "s",
        track_min_max: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize a new Exponential Histogram.

        Args:
            window_size: Size of the sliding window (items for count-based, duration for time-based)
            error_bounds: Maximum relative error in estimates (epsilon), must be between 0 and 1
            memory_limit_bytes: Optional maximum memory usage in bytes
            is_time_based: If True, window_size is in time units; otherwise, it's in items
            timestamp_unit: Time unit for is_time_based ("s" for seconds, "ms" for milliseconds)
            track_min_max: Whether to track min and max values in each bucket
            seed: Optional random seed (not used in this deterministic implementation)
        """
        super().__init__(memory_limit_bytes)

        if not (0 < error_bounds < 1):
            raise ValueError("Error bound must be between 0 and 1")

        if is_time_based:
            if window_size <= 0:
                raise ValueError(
                    "Window size must be greater than 0 for time-based windows"
                )
        else:
            if window_size < 1:
                raise ValueError(
                    "Window size must be at least 1 for count-based windows"
                )

        self._window_size = window_size
        self._error_bound = error_bounds
        self._is_time_based = is_time_based
        self._track_min_max = track_min_max

        # Maximum buckets per size - calculated as k = ⌈1/ε⌉ to ensure the error bound
        self._k = math.ceil(1 / error_bounds)

        # For timestamps
        self._timestamp_unit = timestamp_unit
        self._time_multiplier = 1000 if timestamp_unit == "ms" else 1

        # Initialize the buckets: mapping from bucket size to list of buckets of that size
        self._buckets: Dict[int, List[Bucket]] = defaultdict(list)

        # For count-based windows: track sequence number
        self._sequence_number = 0

        # Window boundary tracking (oldest timestamp/sequence we should keep)
        self._window_start = 0

        # Statistics tracking
        self._total_count = 0
        self._total_sum = 0.0

        # Cached window stats
        self._cached_stats: Optional[Dict[str, float]] = None

    def update(
        self,
        item: Optional[T] = None,
        value: float = 1.0,
        timestamp: Optional[int] = None,
    ) -> None:
        """
        Update the histogram with a new item from the stream.

        Args:
            item: The new item (not used in this implementation, but required by interface)
            value: The value associated with the item (default is 1.0 for simple counting)
            timestamp: Optional timestamp for time-based windows (current time used if None)
        """
        # Increment the items processed counter from base class
        super().update(item)

        # Invalidate cached stats
        self._cached_stats = None

        # Get the current timestamp
        current_time = self._get_current_timestamp(timestamp)

        # For time-based window, remove expired buckets first
        if self._is_time_based:
            self._remove_expired(current_time)

        # Create a new bucket
        new_bucket = Bucket(timestamp=current_time, size=1, sum_value=value)

        # If tracking min/max, set them to the current value
        if self._track_min_max:
            new_bucket.min_value = value
            new_bucket.max_value = value

        # Add the new bucket to the smallest size level
        self._buckets[1].append(new_bucket)

        # Update totals
        self._total_count += 1
        self._total_sum += value

        # Merge buckets to maintain our invariant (at most k buckets of each size)
        self._merge_buckets()

        # For count-based windows, check if we exceed the window size
        if not self._is_time_based:
            # Increment sequence number
            self._sequence_number += 1
            if self._total_count > self._window_size:
                # Remove oldest bucket if needed
                self._remove_oldest()

    def _get_current_timestamp(
        self, timestamp: Optional[Union[int, float]] = None
    ) -> Union[int, float]:
        """
        Get the current timestamp for the item.

        Returns current time, provided timestamp, or sequence number based on window type.
        """
        if self._is_time_based:
            if timestamp is None:
                # Use current time as float
                return time.time() * self._time_multiplier
            return timestamp
        else:
            # For count-based windows, use sequence number
            return self._sequence_number

    def _merge_buckets(self) -> None:
        """
        Merge buckets when we exceed k buckets of a given size.

        This maintains our size invariant and error bound by combining
        buckets of the same size into larger buckets when necessary.
        """
        size = 1
        while len(self._buckets[size]) > self._k:
            # Get the oldest two buckets of this size
            b1 = self._buckets[size].pop(0)
            b2 = self._buckets[size].pop(0)

            # Create a new bucket with combined size
            merged = Bucket(
                timestamp=min(b1.timestamp, b2.timestamp),
                size=b1.size + b2.size,
                sum_value=b1.sum_value + b2.sum_value,
            )

            # Update min/max if tracking
            if self._track_min_max:
                merged.min_value = (
                    min(x for x in [b1.min_value, b2.min_value] if x is not None)
                    if b1.min_value is not None or b2.min_value is not None
                    else None
                )

                merged.max_value = (
                    max(x for x in [b1.max_value, b2.max_value] if x is not None)
                    if b1.max_value is not None or b2.max_value is not None
                    else None
                )

            # Add the merged bucket to the next level
            self._buckets[size * 2].append(merged)

            # Move to the next level
            size *= 2

            # If there are not enough buckets at the next level, we're done
            if len(self._buckets[size]) <= self._k:
                break

    def _remove_expired(self, current_time: float) -> None:
        """
        Remove buckets that are outside the time window.
        """
        # Calculate the cutoff time
        window_duration = self._window_size * self._time_multiplier
        cutoff_time = current_time - window_duration
        self._window_start = cutoff_time

        # Remove buckets with timestamps before the cutoff
        expired_count = 0
        expired_sum = 0.0

        # Check each bucket size
        for size in list(self._buckets.keys()):
            # Find buckets to remove
            new_buckets = []
            for bucket in self._buckets[size]:
                if bucket.timestamp < cutoff_time:
                    expired_count += bucket.size
                    expired_sum += bucket.sum_value
                else:
                    new_buckets.append(bucket)

            # Replace the list or remove the key if empty
            if new_buckets:
                self._buckets[size] = new_buckets
            else:
                del self._buckets[size]

        # Update totals
        self._total_count -= expired_count
        self._total_sum -= expired_sum

        # Invalidate cached stats
        self._cached_stats = None

    def _remove_oldest(self) -> None:
        """
        Remove the oldest bucket to maintain the window size.
        For count-based windows.
        """
        # Find the smallest non-empty bucket size
        smallest_size = min(self._buckets.keys())

        # Update window start
        self._window_start = self._sequence_number - self._window_size

        # Find the oldest bucket across all sizes
        oldest_bucket = None
        oldest_size = None
        oldest_timestamp = float("inf")

        for size, buckets in self._buckets.items():
            if buckets and buckets[0].timestamp < oldest_timestamp:
                oldest_timestamp = buckets[0].timestamp
                oldest_bucket = buckets[0]
                oldest_size = size

        if oldest_bucket and oldest_size:
            # Remove the bucket
            self._buckets[oldest_size].pop(0)

            # If the list is now empty, remove the key
            if not self._buckets[oldest_size]:
                del self._buckets[oldest_size]

            # Update totals
            self._total_count -= oldest_bucket.size
            self._total_sum -= oldest_bucket.sum_value

    def query(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        """
        Query the current state of the histogram.

        Returns statistics for the current window via get_window_stats().
        """
        return self.get_window_stats()

    def get_window_stats(self) -> Dict[str, float]:
        """
        Get statistics for the current window.

        Returns:
            A dictionary with statistics like count, sum, average, etc.
        """
        # Use cached stats if available
        if self._cached_stats:
            return self._cached_stats.copy()

        # Calculate statistics
        stats = {
            "count": self.estimate_count(),
            "sum": self.estimate_sum(),
        }

        # Add average if count is non-zero
        if stats["count"] > 0:
            stats["average"] = stats["sum"] / stats["count"]
        else:
            stats["average"] = 0.0

        # Add min/max if tracking
        if self._track_min_max:
            min_val, max_val = self.estimate_min_max()
            if min_val is not None:
                stats["min"] = min_val
            if max_val is not None:
                stats["max"] = max_val

        # Add error bounds
        stats["relative_error"] = self._error_bound

        # Add window information
        stats["window_size"] = self._window_size
        stats["is_time_based"] = self._is_time_based

        # Cache the stats
        self._cached_stats = stats.copy()

        return stats

    def estimate_count(self) -> int:
        """
        Estimate the count of items in the current window.
        """
        return self._total_count

    def estimate_sum(self) -> float:
        """
        Estimate the sum of values in the current window.
        """
        return self._total_sum

    def estimate_min_max(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Estimate the minimum and maximum values in the current window.

        Returns:
            A tuple of (min, max), or (None, None) if no data or not tracking.
        """
        if not self._track_min_max or not self._buckets:
            return (None, None)

        min_val = float("inf")
        max_val = float("-inf")

        for size, buckets in self._buckets.items():
            for bucket in buckets:
                if bucket.min_value is not None and bucket.min_value < min_val:
                    min_val = bucket.min_value
                if bucket.max_value is not None and bucket.max_value > max_val:
                    max_val = bucket.max_value

        # Check for no data case
        if min_val == float("inf"):
            min_val = None
        if max_val == float("-inf"):
            max_val = None

        return (min_val, max_val)

    def merge(self, other: "ExponentialHistogram[T]") -> "ExponentialHistogram[T]":
        """
        Merge this histogram with another one.

        Creates a new histogram containing data from both inputs with
        the same error guarantees.

        Args:
            other: Another ExponentialHistogram with identical parameters.

        Returns:
            A new merged ExponentialHistogram.
        """
        self._check_same_type(other)

        # Check that parameters match
        if (
            self._window_size != other._window_size
            or self._error_bound != other._error_bound
            or self._is_time_based != other._is_time_based
            or self._timestamp_unit != other._timestamp_unit
        ):
            raise ValueError("Cannot merge histograms with different parameters")

        # Create a new histogram with the same parameters
        result = ExponentialHistogram[T](
            window_size=self._window_size,
            error_bounds=self._error_bound,
            memory_limit_bytes=self._memory_limit_bytes,
            is_time_based=self._is_time_based,
            timestamp_unit=self._timestamp_unit,
            track_min_max=self._track_min_max,
        )

        # Copy all buckets from both histograms
        for size in set(list(self._buckets.keys()) + list(other._buckets.keys())):
            result._buckets[size] = (
                self._buckets.get(size, []).copy() + other._buckets.get(size, []).copy()
            )

        # Sort buckets by timestamp for each size
        for size in result._buckets:
            result._buckets[size].sort(key=lambda b: b.timestamp)

        # Update totals
        result._total_count = self._total_count + other._total_count
        result._total_sum = self._total_sum + other._total_sum

        # Set the sequence number to the max of both
        result._sequence_number = max(self._sequence_number, other._sequence_number)

        # Merge buckets if needed
        size_keys = sorted(result._buckets.keys())
        for size in size_keys:
            while len(result._buckets[size]) > result._k:
                # Get the oldest two buckets of this size
                b1 = result._buckets[size].pop(0)
                b2 = result._buckets[size].pop(0)

                # Create a new bucket with combined size
                merged = Bucket(
                    timestamp=min(b1.timestamp, b2.timestamp),
                    size=b1.size + b2.size,
                    sum_value=b1.sum_value + b2.sum_value,
                )

                # Update min/max if tracking
                if result._track_min_max:
                    merged.min_value = (
                        min(x for x in [b1.min_value, b2.min_value] if x is not None)
                        if b1.min_value is not None or b2.min_value is not None
                        else None
                    )

                    merged.max_value = (
                        max(x for x in [b1.max_value, b2.max_value] if x is not None)
                        if b1.max_value is not None or b2.max_value is not None
                        else None
                    )

                # Add the merged bucket to the next level
                result._buckets[size * 2].append(merged)

                # Sort the new level's buckets
                result._buckets[size * 2].sort(key=lambda b: b.timestamp)

        # Remove expired buckets if time-based
        if result._is_time_based:
            current_time = int(time.time() * result._time_multiplier)
            result._remove_expired(current_time)
        else:
            # Keep removing oldest buckets until we're within the window size
            while result._total_count > result._window_size:
                result._remove_oldest()

        # Update the items_processed count
        result._items_processed = self._combine_items_processed(other)

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the histogram to a dictionary for serialization.
        """
        data = self._base_dict()

        # Convert defaultdict of lists to regular dict of lists for serialization
        buckets_dict = {}
        for size, buckets in self._buckets.items():
            buckets_dict[size] = [
                {
                    "timestamp": b.timestamp,
                    "size": b.size,
                    "sum_value": b.sum_value,
                    "min_value": b.min_value,
                    "max_value": b.max_value,
                }
                for b in buckets
            ]

        data.update(
            {
                "window_size": self._window_size,
                "error_bounds": self._error_bound,
                "is_time_based": self._is_time_based,
                "timestamp_unit": self._timestamp_unit,
                "track_min_max": self._track_min_max,
                "k": self._k,
                "sequence_number": self._sequence_number,
                "window_start": self._window_start,
                "total_count": self._total_count,
                "total_sum": self._total_sum,
                "buckets": buckets_dict,
            }
        )

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExponentialHistogram[T]":
        """
        Create a histogram from a dictionary representation.
        """
        # Create a new histogram
        hist = cls(
            window_size=data["window_size"],
            error_bounds=data["error_bounds"],
            memory_limit_bytes=data.get("memory_limit_bytes"),
            is_time_based=data["is_time_based"],
            timestamp_unit=data["timestamp_unit"],
            track_min_max=data["track_min_max"],
        )

        # Restore state
        hist._sequence_number = data["sequence_number"]
        hist._window_start = data["window_start"]
        hist._total_count = data["total_count"]
        hist._total_sum = data["total_sum"]
        hist._items_processed = data["items_processed"]

        # Restore buckets
        for size_str, buckets_data in data["buckets"].items():
            size = int(size_str)
            for bucket_data in buckets_data:
                bucket = Bucket(
                    timestamp=bucket_data["timestamp"],
                    size=bucket_data["size"],
                    sum_value=bucket_data["sum_value"],
                    min_value=bucket_data["min_value"],
                    max_value=bucket_data["max_value"],
                )
                hist._buckets[size].append(bucket)

        return hist

    def estimate_size(self) -> int:
        """
        Estimate the current memory usage of this histogram in bytes.
        """
        # Base size of the object
        size = sys.getsizeof(self)

        # Size of the buckets dictionary
        size += sys.getsizeof(self._buckets)

        # Size of each bucket
        for bucket_size, buckets in self._buckets.items():
            size += sys.getsizeof(bucket_size)
            size += sys.getsizeof(buckets)

            for bucket in buckets:
                size += sys.getsizeof(bucket)

        return size

    def error_bounds(self) -> Dict[str, float]:
        """
        Calculate the theoretical error bounds for this histogram.
        """
        return {
            "relative_error": self._error_bound,
        }

    def clear(self) -> None:
        """
        Reset the histogram to its initial state.
        """
        self._buckets.clear()
        self._sequence_number = 0
        self._window_start = 0
        self._total_count = 0
        self._total_sum = 0.0
        self._items_processed = 0
        self._cached_stats = None

    def compress(self) -> None:
        """
        Compress the histogram by aggressively merging buckets.

        This reduces memory usage at the cost of increased error.
        The error after compression may exceed the configured error_bounds.
        """
        # Skip if empty
        if not self._buckets:
            return

        # Start with the smallest bucket size and merge aggressively
        sizes = sorted(self._buckets.keys())

        for i in range(len(sizes) - 1):
            current_size = sizes[i]
            next_size = sizes[i + 1]

            # While we have pairs of buckets at this level, merge them
            while len(self._buckets[current_size]) >= 2:
                # Get the oldest two buckets
                b1 = self._buckets[current_size].pop(0)
                b2 = self._buckets[current_size].pop(0)

                # Merge them
                merged = Bucket(
                    timestamp=min(b1.timestamp, b2.timestamp),
                    size=b1.size + b2.size,
                    sum_value=b1.sum_value + b2.sum_value,
                )

                # Update min/max if tracking
                if self._track_min_max:
                    merged.min_value = (
                        min(x for x in [b1.min_value, b2.min_value] if x is not None)
                        if b1.min_value is not None or b2.min_value is not None
                        else None
                    )

                    merged.max_value = (
                        max(x for x in [b1.max_value, b2.max_value] if x is not None)
                        if b1.max_value is not None or b2.max_value is not None
                        else None
                    )

                # Add to next level
                self._buckets[next_size].append(merged)

                # Sort the buckets at the next level
                self._buckets[next_size].sort(key=lambda b: b.timestamp)

            # If no buckets left at this level, remove the key
            if not self._buckets[current_size]:
                del self._buckets[current_size]

        # Invalidate cached stats
        self._cached_stats = None

    @classmethod
    def create_from_error_rate(
        cls,
        relative_error: float,
        window_size: int = 10000,
        is_time_based: bool = False,
        memory_limit_bytes: Optional[int] = None,
    ) -> "ExponentialHistogram[T]":
        """
        Create an Exponential Histogram with the desired error guarantees.

        Factory method optimized for specified error tolerance.
        """
        if not (0 < relative_error < 1):
            raise ValueError("Relative error must be between 0 and 1")

        return cls(
            window_size=window_size,
            error_bounds=relative_error,
            memory_limit_bytes=memory_limit_bytes,
            is_time_based=is_time_based,
        )
