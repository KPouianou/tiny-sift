# tiny_sift/algorithms/quantile_sketch.py

import math
import sys
from typing import List, Dict, Any, Tuple, Optional, Type, TypeVar, Union
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

        # Call parent class's update method to handle items_processed and timing
        super().update(item)

        item = float(item)

        self._unmerged_buffer.append(item)

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
        other._process_buffer()

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

    #
    # Enhanced benchmarking hooks
    #
    def get_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the current state of the T-Digest.

        This method provides insights into the sketch structure, accuracy characteristics,
        and memory usage to help with tuning and performance analysis.

        Returns:
            A dictionary containing various statistics about the digest state.
        """
        # Process buffer first to ensure complete state
        self._process_buffer()

        # Start with basic stats from parent class
        stats = super().get_stats()

        # Add T-Digest specific parameters
        stats.update(
            {
                "compression": self.compression,
                "buffer_size": self._buffer_size,
                "total_weight": self.total_weight,
            }
        )

        # Basic structure statistics
        stats.update(
            {
                "num_centroids": len(self._centroids),
                "buffer_items": len(self._unmerged_buffer),
                "compression_ratio": len(self._centroids) / max(1, self.compression),
            }
        )

        # Min/max tracking
        if self._min_val is not None:
            stats["min_value"] = self._min_val
        if self._max_val is not None:
            stats["max_value"] = self._max_val

        # If there are centroids, add centroid statistics
        if self._centroids:
            # Centroid weight statistics
            weights = [c.weight for c in self._centroids]
            stats.update(
                {
                    "min_weight": min(weights),
                    "max_weight": max(weights),
                    "avg_weight": sum(weights) / len(weights),
                    "total_centroid_weight": sum(weights),
                }
            )

            # Centroid distribution statistics
            means = [c.mean for c in self._centroids]
            if len(means) > 1:
                stats.update(
                    {
                        "centroid_span": max(means) - min(means),
                        "avg_centroid_spacing": (max(means) - min(means))
                        / (len(means) - 1),
                    }
                )

                # Analyze distribution of centroids (are they concentrated in the tails?)
                # Count centroids in different regions
                lower_tail = sum(
                    1 for m in means if m < min(means) + 0.1 * (max(means) - min(means))
                )
                upper_tail = sum(
                    1 for m in means if m > max(means) - 0.1 * (max(means) - min(means))
                )
                middle = len(means) - lower_tail - upper_tail

                stats.update(
                    {
                        "centroids_lower_10pct": lower_tail,
                        "centroids_middle_80pct": middle,
                        "centroids_upper_10pct": upper_tail,
                        "tail_concentration_ratio": (lower_tail + upper_tail)
                        / max(1, middle),
                    }
                )

            # Calculate a distribution measurement of mean spacing
            if len(means) > 2:
                sorted_means = sorted(means)
                spacings = [
                    sorted_means[i + 1] - sorted_means[i]
                    for i in range(len(sorted_means) - 1)
                ]
                stats.update(
                    {
                        "min_spacing": min(spacings),
                        "max_spacing": max(spacings),
                        "median_spacing": sorted(spacings)[len(spacings) // 2],
                    }
                )

        # Add error bounds information
        error_info = self.error_bounds()
        stats.update(error_info)

        # Memory metrics per item
        if self.items_processed > 0:
            stats["bytes_per_item"] = self.estimate_size() / self.items_processed

        return stats

    def error_bounds(self) -> Dict[str, Union[str, float]]:
        """
        Calculate the theoretical error bounds for this T-Digest.

        Unlike uniform-error sketches like HyperLogLog, T-Digest has non-uniform error
        that varies by quantile. This method provides error estimates across different
        quantile ranges.

        Returns:
            A dictionary with error characteristics at different quantiles.
        """
        # Process buffer first
        self._process_buffer()

        # Start with empty bounds
        bounds = {}

        # Return minimal information for empty digest
        if self.is_empty or not self._centroids:
            bounds["state"] = "empty"
            return bounds

        # T-Digest has non-uniform error: smaller at the tails, larger in the middle
        # Theoretical bounds based on compression parameter
        c = self.compression

        # Overall accuracy properties
        bounds["accuracy_model"] = "non-uniform (higher at tails)"
        bounds["theoretical_max_centroids"] = c

        # For different quantile regions, provide error estimates
        # Error should be roughly proportional to q(1-q)/c
        bounds["error_bounds"] = {}

        # Calculate error bounds for different quantiles
        quantiles = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
        for q in quantiles:
            # Error is roughly proportional to q(1-q)/c
            # Normalize to show relative error
            normalized_error = q * (1 - q) / c
            bounds["error_bounds"][f"q{q:.3f}"] = normalized_error

        # Add general notes on accuracy
        bounds["notes"] = [
            "T-Digest has non-uniform error, with higher accuracy at the tails",
            "Error at quantile q is roughly proportional to q(1-q)/c",
            "Where c is the compression parameter",
        ]

        # If we have centroids, estimate actual achieved error based on centroid spacing
        if len(self._centroids) > 1:
            # This is a rough heuristic based on centroid spacing
            sorted_means = sorted(c.mean for c in self._centroids)
            spacings = [
                sorted_means[i + 1] - sorted_means[i]
                for i in range(len(sorted_means) - 1)
            ]
            if spacings:
                avg_spacing = sum(spacings) / len(spacings)
                value_range = max(sorted_means) - min(sorted_means)
                if value_range > 0:
                    bounds["avg_spacing_ratio"] = avg_spacing / value_range

        # Add observations about compression effectiveness
        if self._centroids:
            bounds["actual_centroids"] = len(self._centroids)
            bounds["compression_efficiency"] = len(self._centroids) / max(1, c)

        return bounds

    def analyze_quantile_accuracy(
        self, reference_data: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze the accuracy of quantile estimates against reference data.

        This method is useful for benchmarking the accuracy of the T-Digest against
        actual data or a reference distribution.

        Args:
            reference_data: Optional list of values to compare against. If provided,
                           compares T-Digest quantile estimates with exact quantiles
                           from the data. If None, provides theoretical error bounds.

        Returns:
            A dictionary containing accuracy analysis information.
        """
        # Process buffer first
        self._process_buffer()

        analysis = {
            "algorithm": "T-Digest",
            "compression": self.compression,
            "num_centroids": len(self._centroids),
            "items_processed": self.items_processed,
        }

        # Define quantiles to analyze
        quantiles = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]

        if reference_data:
            # Sort reference data for exact quantile computation
            sorted_data = sorted(reference_data)
            data_len = len(sorted_data)

            if data_len == 0:
                return {"error": "Reference data is empty"}

            # Calculate exact quantiles from reference data
            exact_quantiles = {}
            for q in quantiles:
                index = min(int(q * data_len), data_len - 1)
                exact_quantiles[f"q{q:.3f}"] = sorted_data[index]

            # Calculate T-Digest estimates for the same quantiles
            tdigest_estimates = {}
            for q in quantiles:
                tdigest_estimates[f"q{q:.3f}"] = self.query(q)

            # Calculate absolute and relative errors
            abs_errors = {}
            rel_errors = {}

            for q in quantiles:
                q_key = f"q{q:.3f}"
                exact = exact_quantiles[q_key]
                estimate = tdigest_estimates[q_key]

                # Check for division by zero
                if math.isfinite(exact) and math.isfinite(estimate):
                    abs_errors[q_key] = abs(estimate - exact)
                    # Use abs value in denominator to handle negative values
                    if abs(exact) > 1e-10:
                        rel_errors[q_key] = abs_errors[q_key] / abs(exact)
                    else:
                        rel_errors[q_key] = abs_errors[q_key]

            analysis.update(
                {
                    "reference_data_size": data_len,
                    "exact_quantiles": exact_quantiles,
                    "tdigest_estimates": tdigest_estimates,
                    "absolute_errors": abs_errors,
                    "relative_errors": rel_errors,
                }
            )

            # Add summary statistics
            if rel_errors:
                analysis["max_relative_error"] = max(rel_errors.values())
                analysis["avg_relative_error"] = sum(rel_errors.values()) / len(
                    rel_errors
                )

                # Categorize errors by quantile region
                tail_errors = [
                    e
                    for q, e in rel_errors.items()
                    if "0.001" in q or "0.01" in q or "0.99" in q or "0.999" in q
                ]
                mid_errors = [
                    e
                    for q, e in rel_errors.items()
                    if "0.25" in q or "0.5" in q or "0.75" in q
                ]

                if tail_errors:
                    analysis["avg_tail_error"] = sum(tail_errors) / len(tail_errors)
                if mid_errors:
                    analysis["avg_mid_error"] = sum(mid_errors) / len(mid_errors)
        else:
            # No reference data - provide theoretical bounds
            theoretical_errors = {}
            for q in quantiles:
                # Error is roughly proportional to q(1-q)/c
                # This is a simplified model - actual error depends on data distribution
                theoretical_errors[f"q{q:.3f}"] = q * (1 - q) / self.compression

            analysis["theoretical_relative_errors"] = theoretical_errors

            # Simple summary of expected accuracy
            analysis["expected_median_error"] = 0.25 / self.compression
            analysis["expected_tail_error_q001"] = 0.001 * 0.999 / self.compression
            analysis["expected_tail_error_q999"] = 0.001 * 0.999 / self.compression

        return analysis

    def get_centroid_distribution(self) -> Dict[str, Any]:
        """
        Analyze the distribution of centroids across the value range.

        This method provides detailed information about how centroids are distributed,
        which is useful for understanding the T-Digest's approximation characteristics.

        Returns:
            A dictionary containing centroid distribution information.
        """
        # Process buffer first
        self._process_buffer()

        distribution = {
            "num_centroids": len(self._centroids),
            "total_weight": self.total_weight,
        }

        if not self._centroids:
            distribution["state"] = "empty"
            return distribution

        # Extract basic stats
        means = [c.mean for c in self._centroids]
        weights = [c.weight for c in self._centroids]

        # Sort centroids by mean for analysis
        sorted_centroids = sorted(self._centroids)
        sorted_means = [c.mean for c in sorted_centroids]
        sorted_weights = [c.weight for c in sorted_centroids]

        # Basic statistics
        distribution.update(
            {
                "value_range": [min(means), max(means)],
                "min_weight": min(weights),
                "max_weight": max(weights),
                "avg_weight": sum(weights) / len(weights),
            }
        )

        # Analyze centroid spacing
        if len(sorted_means) > 1:
            spacings = [
                sorted_means[i + 1] - sorted_means[i]
                for i in range(len(sorted_means) - 1)
            ]
            distribution.update(
                {
                    "min_spacing": min(spacings),
                    "max_spacing": max(spacings),
                    "mean_spacing": sum(spacings) / len(spacings),
                    "median_spacing": sorted(spacings)[len(spacings) // 2],
                }
            )

            # Analyze distribution uniformity
            # Coefficient of variation (CV) measures relative variability
            # High CV indicates non-uniform spacing (which is expected in T-Digest)
            if distribution["mean_spacing"] > 0:
                variance = sum(
                    (s - distribution["mean_spacing"]) ** 2 for s in spacings
                ) / len(spacings)
                std_dev = math.sqrt(variance)
                distribution["spacing_cv"] = std_dev / distribution["mean_spacing"]

            # Analyze weight distribution across value range
            # Divide the range into bins and count weight in each bin
            num_bins = min(10, len(self._centroids))
            if num_bins > 1:
                value_range = max(means) - min(means)
                if value_range > 0:
                    bins = []
                    bin_weights = [0.0] * num_bins

                    for i in range(num_bins + 1):
                        bins.append(min(means) + (i * value_range / num_bins))

                    # Assign centroids to bins
                    for c in self._centroids:
                        bin_index = min(
                            num_bins - 1,
                            int((c.mean - min(means)) * num_bins / value_range),
                        )
                        bin_weights[bin_index] += c.weight

                    # Normalize bin weights as percentages
                    total_weight = sum(bin_weights)
                    if total_weight > 0:
                        normalized_weights = [
                            w / total_weight * 100 for w in bin_weights
                        ]

                        distribution["bin_edges"] = bins
                        distribution["bin_weights_pct"] = normalized_weights

        # Analyze centroid concentration (T-Digest should have more centroids at the tails)
        if len(means) > 2:
            # Divide the range into three segments: lower tail (10%), middle (80%), upper tail (10%)
            low_threshold = min(means) + 0.1 * (max(means) - min(means))
            high_threshold = max(means) - 0.1 * (max(means) - min(means))

            low_centroids = [c for c in self._centroids if c.mean <= low_threshold]
            mid_centroids = [
                c for c in self._centroids if low_threshold < c.mean < high_threshold
            ]
            high_centroids = [c for c in self._centroids if c.mean >= high_threshold]

            distribution.update(
                {
                    "low_tail_centroids": len(low_centroids),
                    "mid_centroids": len(mid_centroids),
                    "high_tail_centroids": len(high_centroids),
                    "low_tail_weight": sum(c.weight for c in low_centroids),
                    "mid_weight": sum(c.weight for c in mid_centroids),
                    "high_tail_weight": sum(c.weight for c in high_centroids),
                }
            )

            # Expected t-digest centroid distribution should have a bias toward the tails
            total_centroids = len(self._centroids)
            if total_centroids > 0:
                distribution.update(
                    {
                        "low_tail_pct": len(low_centroids) / total_centroids * 100,
                        "mid_pct": len(mid_centroids) / total_centroids * 100,
                        "high_tail_pct": len(high_centroids) / total_centroids * 100,
                    }
                )

        return distribution

    def analyze_compression_efficiency(self) -> Dict[str, Any]:
        """
        Analyze how effectively the T-Digest is compressing the data.

        This method provides insights into the relationship between the compression
        parameter, the number of centroids, and memory usage.

        Returns:
            A dictionary containing compression efficiency metrics.
        """
        # Process buffer first
        self._process_buffer()

        analysis = {
            "compression_parameter": self.compression,
            "num_centroids": len(self._centroids),
            "items_processed": self.items_processed,
            "memory_usage_bytes": self.estimate_size(),
        }

        # Calculate efficiency metrics
        if self.compression > 0:
            analysis["centroid_utilization"] = len(self._centroids) / self.compression

        if self.items_processed > 0:
            analysis["compression_ratio"] = self.items_processed / max(
                1, len(self._centroids)
            )
            analysis["bytes_per_item"] = (
                analysis["memory_usage_bytes"] / self.items_processed
            )

        # Check if buffer is being used efficiently
        analysis["buffer_size"] = self._buffer_size
        analysis["buffer_utilization"] = len(self._unmerged_buffer) / max(
            1, self._buffer_size
        )

        # Estimate theoretical memory savings
        # If we stored all unique values instead of using T-Digest
        if self.items_processed > 0:
            # Rough estimate of storing all unique values (assuming 50% are unique)
            estimated_uniques = min(self.items_processed, self.items_processed * 0.5)
            naive_storage = estimated_uniques * (
                sys.getsizeof(0.0) + sys.getsizeof(1.0)
            )  # value + weight

            analysis["estimated_memory_savings"] = 1.0 - (
                analysis["memory_usage_bytes"] / max(1, naive_storage)
            )
            analysis["estimated_memory_savings_pct"] = (
                analysis["estimated_memory_savings"] * 100
            )

        return analysis

    def clear(self) -> None:
        """
        Reset the T-Digest to its initial empty state.

        This method clears all data while preserving configuration parameters.
        """
        # Clear all data structures
        self._centroids = []
        self._unmerged_buffer = []
        self.total_weight = 0.0
        self._min_val = None
        self._max_val = None
        self._items_processed = 0

    @classmethod
    def create_from_accuracy_target(
        cls, accuracy_target: float, tail_focus: bool = True
    ) -> "TDigest":
        """
        Create a T-Digest with a compression factor optimized for a target accuracy.

        Args:
            accuracy_target: Target relative error for quantile estimates (0.0-1.0).
                            Lower values create more accurate but larger sketches.
            tail_focus: If True, optimizes for accuracy at the tails (0.01, 0.99).
                       If False, optimizes for accuracy at the median (0.5).

        Returns:
            A new T-Digest configured for the target accuracy.

        Raises:
            ValueError: If accuracy_target is not between 0 and 1.
        """
        if not (0.0 < accuracy_target < 1.0):
            raise ValueError("Accuracy target must be between 0 and 1")

        # Calculate compression parameter based on target accuracy
        # For median (q=0.5), error is approximately 0.25/compression
        # For tails (q=0.01 or q=0.99), error is approximately 0.01*0.99/compression = 0.0099/compression

        if tail_focus:
            # Optimize for tail accuracy (q=0.01 or q=0.99)
            # Solve for compression: 0.0099/compression = accuracy_target
            compression = math.ceil(0.0099 / accuracy_target)
        else:
            # Optimize for median accuracy (q=0.5)
            # Solve for compression: 0.25/compression = accuracy_target
            compression = math.ceil(0.25 / accuracy_target)

        # Ensure compression is at least the minimum allowed
        compression = max(20, compression)

        return cls(compression=compression)

    def force_compress(self) -> None:
        """
        Force compression of the sketch, potentially reducing memory usage.

        This method processes any buffered values and aggressively compresses
        centroids to reduce memory footprint, potentially at the cost of accuracy.
        """
        # Process buffer first
        self._process_buffer()

        # Exit if no centroids
        if not self._centroids:
            return

        # Sort centroids
        self._centroids.sort()

        # Repeatedly merge adjacent centroids until we're below target size
        # Target a more aggressive compression than the standard algorithm
        target_size = max(20, self.compression // 2)

        while len(self._centroids) > target_size:
            # Find pair with minimum mean difference
            min_diff = float("inf")
            min_idx = -1

            for i in range(len(self._centroids) - 1):
                diff = self._centroids[i + 1].mean - self._centroids[i].mean
                if diff < min_diff:
                    min_diff = diff
                    min_idx = i

            if min_idx == -1:
                break  # No valid pair found

            # Merge the pair
            c1 = self._centroids[min_idx]
            c2 = self._centroids[min_idx + 1]
            total_weight = c1.weight + c2.weight

            if total_weight > 0:
                merged_mean = (c1.mean * c1.weight + c2.mean * c2.weight) / total_weight
                merged = _Centroid(mean=merged_mean, weight=total_weight)
                self._centroids[min_idx : min_idx + 2] = [merged]
            else:
                del self._centroids[min_idx : min_idx + 2]
