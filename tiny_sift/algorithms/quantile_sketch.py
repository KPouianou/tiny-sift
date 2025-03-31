# tiny_sift/algorithms/quantile_sketch.py

import math
import sys
from typing import List, Dict, Any, Tuple, Optional, Type, TypeVar
import bisect

from tiny_sift.core.base import StreamSummary

# Type variable for the class itself (for from_dict)
TDigestType = TypeVar("TDigestType", bound="TDigest")


class _Centroid:
    """Internal representation of a centroid in the T-Digest algorithm."""

    __slots__ = ["mean", "weight"]

    def __init__(self, mean: float, weight: float = 1.0):
        """Initialize a centroid with a mean value and weight."""
        if weight < 0:
            raise ValueError("Centroid weight cannot be negative")
        self.mean = float(mean)
        self.weight = float(weight)

    def __lt__(self, other: "_Centroid") -> bool:
        """Allow centroids to be sorted by mean value."""
        return self.mean < other.mean

    def __repr__(self) -> str:
        """Provide a readable representation of the centroid."""
        return f"Centroid(mean={self.mean:.4g}, weight={self.weight:.4g})"

    def to_dict(self) -> Dict[str, float]:
        """Serialize the centroid to a dictionary."""
        return {"mean": self.mean, "weight": self.weight}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "_Centroid":
        """Deserialize a centroid from a dictionary."""
        if "mean" not in data or "weight" not in data:
            raise ValueError("Centroid dictionary missing 'mean' or 'weight'")
        if data["weight"] < 0:
            raise ValueError(
                f"Invalid serialized data: Centroid weight cannot be negative ({data['weight']})"
            )
        return cls(mean=data["mean"], weight=data["weight"])


class TDigest(StreamSummary):
    """
    T-Digest for efficient and accurate quantile estimation over data streams.

    The T-Digest (Dunning, 2019) is a probabilistic data structure that provides
    accurate estimation of quantiles while using bounded memory. Key properties:

    1. Memory usage is controlled by the compression parameter, not data size
    2. Accuracy is non-uniform: extreme quantiles (near 0 or 1) are more precise
    3. Error bounds scale with quantile (1-q), making tails more accurate
    4. Mergeable: multiple digests from separate streams can be combined

    The algorithm clusters data points into centroids, with more centroids near
    the tails and fewer in the middle, creating a bias that improves accuracy at
    the extremes.
    """

    DEFAULT_COMPRESSION: int = 100
    DEFAULT_BUFFER_FACTOR: int = 5

    def __init__(self, compression: int = DEFAULT_COMPRESSION):
        """
        Initialize a TDigest sketch.

        Args:
            compression: Controls accuracy and memory usage. Higher values
                improve accuracy at the cost of more memory. The maximum
                number of centroids is typically proportional to this value.
                Must be ≥ 20. Default: 100.

        Raises:
            ValueError: If compression is less than 20 or not an integer.
        """
        super().__init__()
        if not isinstance(compression, int) or compression < 20:
            raise ValueError("Compression factor must be an integer >= 20")

        self.compression: int = compression
        self._centroids: List[_Centroid] = []
        self._unmerged_buffer: List[float] = []
        self._buffer_size: int = max(10, self.DEFAULT_BUFFER_FACTOR * self.compression)

        self.total_weight: float = 0.0
        self._min_val: Optional[float] = None
        self._max_val: Optional[float] = None

    def update(self, item: float) -> None:
        """
        Add a value to the sketch.

        The value is initially added to a buffer. When the buffer reaches
        a threshold size, values are processed into centroids.

        Args:
            item: Numeric value to add. Non-finite values (NaN, +/-Inf) are ignored.
        """
        if not isinstance(item, (int, float)) or not math.isfinite(item):
            return

        item = float(item)

        self._unmerged_buffer.append(item)
        self._items_processed += 1

        # Track min/max for precise quantile estimation at the extremes
        if self._min_val is None or item < self._min_val:
            self._min_val = item
        if self._max_val is None or item > self._max_val:
            self._max_val = item

        # Process the buffer if it has reached capacity
        if len(self._unmerged_buffer) >= self._buffer_size:
            self._process_buffer()

    def _process_buffer(self) -> None:
        """
        Process buffered values into centroids.

        Sorts the buffer, creates centroids, and triggers compression
        to maintain the size bound if needed.
        """
        if not self._unmerged_buffer:
            return

        # Sort for locality when creating centroids
        self._unmerged_buffer.sort()

        # Create centroids for buffer values
        new_centroids = [
            _Centroid(mean=val, weight=1.0) for val in self._unmerged_buffer
        ]

        self.total_weight += len(self._unmerged_buffer)
        self._unmerged_buffer = []

        # Add new centroids and maintain size bound
        self._centroids.extend(new_centroids)
        self._compress()

    def _compress(self) -> None:
        """
        Reduce the number of centroids to maintain the size bound.

        This is the core space-saving mechanism of T-Digest. It iteratively
        merges adjacent centroids with the smallest difference in means until
        the number of centroids is at most `compression`.
        """
        if len(self._centroids) <= self.compression:
            return

        # Sort centroids by mean
        self._centroids.sort()

        # Merge centroids until we reach the target size
        while len(self._centroids) > self.compression:
            # Find the adjacent pair with smallest mean difference
            min_mean_diff = float("inf")
            min_idx = -1

            for i in range(len(self._centroids) - 1):
                c1 = self._centroids[i]
                c2 = self._centroids[i + 1]

                mean_diff = c2.mean - c1.mean
                current_weight_sum = c1.weight + c2.weight

                # Prefer pairs with smaller mean difference, then smaller total weight
                is_better_candidate = False
                if mean_diff < min_mean_diff:
                    is_better_candidate = True
                elif mean_diff == min_mean_diff:
                    if min_idx != -1:
                        prev_best_weight = (
                            self._centroids[min_idx].weight
                            + self._centroids[min_idx + 1].weight
                        )
                        if current_weight_sum < prev_best_weight:
                            is_better_candidate = True

                if is_better_candidate:
                    min_mean_diff = mean_diff
                    min_idx = i

            if min_idx == -1:
                break  # Safety check

            # Merge the centroids
            c1 = self._centroids[min_idx]
            c2 = self._centroids[min_idx + 1]

            merged_weight = c1.weight + c2.weight
            if merged_weight > 0:
                # Calculate weighted mean of the two centroids
                merged_mean = (
                    (c1.mean * c1.weight) + (c2.mean * c2.weight)
                ) / merged_weight
                merged_centroid = _Centroid(mean=merged_mean, weight=merged_weight)

                # Replace the two originals with the merged centroid
                self._centroids[min_idx : min_idx + 2] = [merged_centroid]
            else:
                # Unexpected case: remove both zero-weight centroids
                del self._centroids[min_idx + 1]
                del self._centroids[min_idx]

    def query(self, quantile: float) -> float:
        """
        Estimate the value at the given quantile.

        Args:
            quantile: Target quantile between 0.0 and 1.0.
                      0.0 returns the minimum value.
                      0.5 returns the estimated median.
                      1.0 returns the maximum value.

        Returns:
            Estimated value at the specified quantile. Returns NaN
            if the sketch is empty.

        Raises:
            ValueError: If quantile is not between 0.0 and 1.0.
        """
        if not (0.0 <= quantile <= 1.0):
            raise ValueError("Quantile must be between 0.0 and 1.0")

        # Ensure all buffered data is incorporated
        self._process_buffer()

        # Handle empty sketch
        if not self._centroids or self.total_weight == 0:
            return float("nan")

        # Handle extreme quantiles precisely
        if quantile == 0.0:
            return self._min_val if self._min_val is not None else float("nan")
        if quantile == 1.0:
            return self._max_val if self._max_val is not None else float("nan")

        # Calculate the target weight based on quantile
        target_weight = quantile * self.total_weight

        # Find centroids bracketing the target weight
        cumulative_weight = 0.0
        for i, c in enumerate(self._centroids):
            # Calculate the cumulative weight at the center of this centroid
            centroid_mid_weight = cumulative_weight + c.weight / 2.0

            # Check if target falls at/before this centroid's midpoint
            if target_weight <= centroid_mid_weight and c.weight > 0:
                if i == 0:
                    # Target falls within first centroid: interpolate between
                    # minimum value and centroid mean
                    fraction = target_weight / (c.weight / 2.0)
                    fraction = max(0.0, min(1.0, fraction))

                    lower_bound = self._min_val if self._min_val is not None else c.mean
                    return lower_bound + fraction * (c.mean - lower_bound)
                else:
                    # Target falls between centroids: interpolate between
                    # current and previous centroid means
                    prev_c = self._centroids[i - 1]
                    prev_cumulative_start_weight = cumulative_weight - prev_c.weight
                    prev_centroid_mid_weight = (
                        prev_cumulative_start_weight + prev_c.weight / 2.0
                    )

                    weight_diff = centroid_mid_weight - prev_centroid_mid_weight
                    if weight_diff <= 1e-9:
                        return c.mean

                    # Calculate interpolation fraction
                    fraction = (target_weight - prev_centroid_mid_weight) / weight_diff
                    fraction = max(0.0, min(1.0, fraction))

                    # Perform linear interpolation
                    return prev_c.mean + fraction * (c.mean - prev_c.mean)

            # Add this centroid's weight to the cumulative total
            if c.weight > 0:
                cumulative_weight += c.weight

        # If we reach here, return the maximum value
        return self._max_val if self._max_val is not None else float("nan")

    def merge(self: TDigestType, other: TDigestType) -> TDigestType:
        """
        Merge this sketch with another T-Digest.

        Creates a new sketch that represents the combined data from both inputs.
        The original sketches are not modified. This enables parallel processing
        of distributed streams.

        Args:
            other: Another TDigest sketch with the same compression parameter.

        Returns:
            A new TDigest containing data from both inputs.

        Raises:
            TypeError: If 'other' is not a TDigest.
            ValueError: If the compression parameters don't match.
        """
        self._check_same_type(other)

        if self.compression != other.compression:
            raise ValueError(
                f"Cannot merge TDigest sketches with different compression factors: "
                f"{self.compression} != {other.compression}"
            )

        # Create a new sketch for the merged result
        merged_sketch = TDigest(compression=self.compression)

        # Process buffers to ensure complete states
        self._process_buffer()

        # Copy centroids and buffers (creates new lists)
        merged_sketch._centroids = self._centroids + other._centroids
        merged_sketch._unmerged_buffer = self._unmerged_buffer + other._unmerged_buffer

        # Combine scalar state
        merged_sketch.total_weight = self.total_weight + other.total_weight
        merged_sketch._items_processed = self.items_processed + other.items_processed

        # Combine min/max values
        merged_min = None
        if self._min_val is not None:
            merged_min = self._min_val
        if other._min_val is not None:
            merged_min = (
                min(merged_min, other._min_val)
                if merged_min is not None
                else other._min_val
            )
        merged_sketch._min_val = merged_min

        merged_max = None
        if self._max_val is not None:
            merged_max = self._max_val
        if other._max_val is not None:
            merged_max = (
                max(merged_max, other._max_val)
                if merged_max is not None
                else other._max_val
            )
        merged_sketch._max_val = merged_max

        # Process buffer and compress the combined centroids
        merged_sketch._process_buffer()

        return merged_sketch

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the T-Digest to a dictionary.

        Processes the buffer first to ensure a complete state snapshot.

        Returns:
            Dictionary containing the sketch configuration and internal state.
        """
        self._process_buffer()

        state = self._base_dict()
        state.update(
            {
                "compression": self.compression,
                "total_weight": self.total_weight,
                "min_val": self._min_val,
                "max_val": self._max_val,
                "centroids": [c.to_dict() for c in self._centroids],
            }
        )
        return state

    @classmethod
    def from_dict(cls: Type[TDigestType], data: Dict[str, Any]) -> TDigestType:
        """
        Deserialize a T-Digest from a dictionary representation.

        Args:
            data: Dictionary created by to_dict().

        Returns:
            A reconstructed TDigest instance.

        Raises:
            ValueError: If the dictionary is missing required keys or has invalid data.
        """
        # Validate the dictionary format
        if "type" not in data:
            raise ValueError("Invalid dictionary format for TDigest. Missing 'type'")

        if data.get("type") != cls.__name__:
            raise ValueError(
                f"Dictionary represents class '{data.get('type')}' but expected '{cls.__name__}'"
            )

        required_keys = {
            "compression",
            "total_weight",
            "centroids",
            "items_processed",
        }

        missing_keys = required_keys - data.keys()
        if missing_keys:
            raise ValueError(
                f"Invalid dictionary format for TDigest. Missing keys: {missing_keys}"
            )

        # Create instance with compression parameter
        instance = cls(compression=data["compression"])

        # Restore state
        instance._items_processed = data["items_processed"]
        instance.total_weight = data["total_weight"]
        instance._min_val = data.get("min_val")
        instance._max_val = data.get("max_val")

        # Restore centroids
        try:
            instance._centroids = [
                _Centroid.from_dict(c_data) for c_data in data["centroids"]
            ]
        except (ValueError, KeyError) as e:
            raise ValueError(f"Error deserializing centroids: {e}") from e

        # Set buffer size based on compression
        instance._buffer_size = max(
            10, cls.DEFAULT_BUFFER_FACTOR * instance.compression
        )
        instance._unmerged_buffer = []

        # Ensure centroids are sorted
        instance._centroids.sort()

        return instance

    def estimate_size(self) -> int:
        """
        Estimate the memory footprint of the T-Digest in bytes.

        Returns:
            Estimated size in bytes.
        """
        # Base object size
        size = super().estimate_size()

        # Size of scalar attributes
        size += sys.getsizeof(self.compression)
        size += sys.getsizeof(self.total_weight)
        size += sys.getsizeof(self._buffer_size)
        if self._min_val is not None:
            size += sys.getsizeof(self._min_val)
        if self._max_val is not None:
            size += sys.getsizeof(self._max_val)

        # Size of container data structures
        size += sys.getsizeof(self._centroids)
        if self._centroids:
            size += sum(sys.getsizeof(c) for c in self._centroids)
            size += sum(
                sys.getsizeof(c.mean) + sys.getsizeof(c.weight) for c in self._centroids
            )

        # Size of the unprocessed buffer
        size += sys.getsizeof(self._unmerged_buffer)
        if self._unmerged_buffer:
            size += sum(sys.getsizeof(item) for item in self._unmerged_buffer)

        return size

    def __len__(self) -> int:
        """Return the number of values processed by the sketch."""
        return self.items_processed

    @property
    def is_empty(self) -> bool:
        """Check if the sketch contains any data."""
        return self.items_processed == 0

    def get_centroids(self) -> List[Tuple[float, float]]:
        """
        Return the current centroids as (mean, weight) tuples.

        This is primarily for debugging and inspection purposes.

        Returns:
            List of centroids as (mean, weight) tuples, sorted by mean.
        """
        self._process_buffer()
        return [(c.mean, c.weight) for c in sorted(self._centroids)]
