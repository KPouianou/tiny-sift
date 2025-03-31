# tiny_sift/algorithms/quantile_sketch.py

import math
import sys
from typing import List, Dict, Any, Tuple, Optional, Type, TypeVar
import bisect  # Used for efficient sorted list operations

# Assuming StreamSummary is in tiny_sift.core.base
# Adjust the import path if necessary.
try:
    from tiny_sift.core.base import StreamSummary
except ImportError:
    # Fallback for environments where the package structure isn't fully set up.
    print(
        "Warning: StreamSummary base class not found. Defining TDigest independently."
    )

# Type variable for the class itself (for from_dict)
TDigestType = TypeVar("TDigestType", bound="TDigest")


# Helper class for centroids (using __slots__ for memory efficiency)
class _Centroid:
    """Internal representation of a centroid for TDigest."""

    __slots__ = ["mean", "weight"]

    def __init__(self, mean: float, weight: float = 1.0):
        """Initializes a centroid. Weights must be non-negative."""
        if weight < 0:
            raise ValueError("Centroid weight cannot be negative")
        self.mean = float(mean)
        self.weight = float(weight)

    def __lt__(self, other: "_Centroid") -> bool:
        """Enables sorting centroids by mean."""
        return self.mean < other.mean

    def __repr__(self) -> str:
        """Provides a readable representation of the centroid."""
        return f"Centroid(mean={self.mean:.4g}, weight={self.weight:.4g})"

    def to_dict(self) -> Dict[str, float]:
        """Serializes the centroid to a dictionary."""
        return {"mean": self.mean, "weight": self.weight}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "_Centroid":
        """Deserializes a centroid from a dictionary."""
        if "mean" not in data or "weight" not in data:
            raise ValueError("Centroid dictionary missing 'mean' or 'weight'")
        # Basic validation during deserialization
        if data["weight"] < 0:
            raise ValueError(
                f"Invalid serialized data: Centroid weight cannot be negative ({data['weight']})"
            )
        return cls(mean=data["mean"], weight=data["weight"])


class TDigest(StreamSummary):
    """
    Estimates quantiles from a stream of floating-point numbers using the
    t-digest algorithm with bounded memory.

    The algorithm clusters data points into centroids. The maximum number of
    centroids is controlled by the `compression` parameter, providing a
    trade-off between accuracy and memory usage. Merging prioritizes combining
    centroids with the closest means.

    Reference:
    Dunning, Ted, and Otmar Ertl. "Computing extremely accurate quantiles using
    t-digests." arXiv preprint arXiv:1902.04023 (2019).
    (Note: This implementation uses common practical simplifications.)
    """

    DEFAULT_COMPRESSION: int = 100
    DEFAULT_BUFFER_FACTOR: int = (
        5  # Process buffer when it's this factor * compression size
    )

    def __init__(self, compression: int = DEFAULT_COMPRESSION):
        """
        Initializes the TDigest sketch.

        Args:
            compression (int): Controls the maximum number of centroids.
                Higher values generally increase accuracy and memory usage.
                Must be >= 20 (recommended minimum). Defaults to 100.
        """
        super().__init__()
        if not isinstance(compression, int) or compression < 20:
            raise ValueError("Compression factor must be an integer >= 20")

        self.compression: int = compression
        self._centroids: List[_Centroid] = []
        self._unmerged_buffer: List[float] = []
        # Determine buffer size based on compression factor, with a minimum size.
        self._buffer_size: int = max(10, self.DEFAULT_BUFFER_FACTOR * self.compression)

        self.total_weight: float = 0.0
        # Track min/max explicitly for accurate 0/1 quantiles and edge cases.
        self._min_val: Optional[float] = None
        self._max_val: Optional[float] = None

    def update(self, item: float) -> None:
        """
        Adds a floating-point number to the sketch.

        Args:
            item (float): The numerical value from the stream. Non-finite
                          values (NaN, inf) are ignored.
        """
        # Silently ignore non-finite numbers. Could raise ValueError instead.
        if not isinstance(item, (int, float)) or not math.isfinite(item):
            # print(f"Warning: Ignoring non-finite value: {item}") # Optional warning
            return

        item = float(item)  # Ensure float type

        self._unmerged_buffer.append(item)
        self._items_processed += 1

        # Update overall min/max values seen.
        if self._min_val is None or item < self._min_val:
            self._min_val = item
        if self._max_val is None or item > self._max_val:
            self._max_val = item

        # Process the buffer if it has reached its capacity.
        if len(self._unmerged_buffer) >= self._buffer_size:
            self._process_buffer()

    def _process_buffer(self) -> None:
        """Incorporates values from the buffer into centroids and compresses."""
        if not self._unmerged_buffer:
            return

        # Sort the buffer for efficient processing and potential locality benefits.
        self._unmerged_buffer.sort()

        # Create temporary centroids for each value in the buffer (weight=1).
        new_centroids = [
            _Centroid(mean=val, weight=1.0) for val in self._unmerged_buffer
        ]

        # Update the total weight tracked by the sketch.
        self.total_weight += len(self._unmerged_buffer)

        # Clear the buffer now that its contents are processed.
        processed_count = len(self._unmerged_buffer)  # Store count before clearing
        self._unmerged_buffer = []

        # Add the new centroids to the main list.
        self._centroids.extend(new_centroids)

        # If adding the new centroids significantly increased the count,
        # compress immediately. Otherwise, _compress might be deferred slightly.
        # We always compress after adding buffer contents to maintain bounds.
        self._compress()

    def _compress(self) -> None:
        """
        Reduces the number of centroids by merging adjacent pairs until the
        number of centroids is at most `self.compression`.

        This is the core memory bounding mechanism. It iteratively finds the
        adjacent pair of centroids with the smallest difference in their means
        and merges them. This strategy aims to preserve detail by merging
        clusters that are already close together. In case of ties in mean
        difference, the pair with the smaller combined weight is merged,
        prioritizing merging smaller clusters when means are equally close.

        Postconditions:
          - `len(self._centroids)` will be less than or equal to `self.compression`.
          - `self._centroids` remains sorted by `mean`.
        """
        if len(self._centroids) <= self.compression:
            return

        # Sort centroids by mean before compression. This is essential for
        # finding and merging *adjacent* centroids meaningfully.
        self._centroids.sort()

        # Iteratively merge centroids until the target count is reached.
        while len(self._centroids) > self.compression:
            # Find the adjacent pair to merge based on the chosen criteria.
            min_mean_diff = float("inf")
            min_idx = -1  # Index of the first centroid in the pair to merge

            for i in range(len(self._centroids) - 1):
                c1 = self._centroids[i]
                c2 = self._centroids[i + 1]

                mean_diff = c2.mean - c1.mean
                current_weight_sum = c1.weight + c2.weight

                # Determine if this pair is a better candidate for merging
                # than the best found so far.
                is_better_candidate = False
                if mean_diff < min_mean_diff:
                    # Primary criterion: Smallest difference in means.
                    is_better_candidate = True
                elif mean_diff == min_mean_diff:
                    # Tie-breaking criterion: Smaller combined weight.
                    if min_idx != -1:  # Check if a previous best exists
                        prev_best_weight = (
                            self._centroids[min_idx].weight
                            + self._centroids[min_idx + 1].weight
                        )
                        if current_weight_sum < prev_best_weight:
                            is_better_candidate = True
                    # If no previous best (min_idx == -1), the first pair qualifies implicitly.

                if is_better_candidate:
                    min_mean_diff = mean_diff
                    min_idx = i

            # If no suitable pair is found (should not happen if len > compression), break.
            if min_idx == -1:
                break  # Safety break

            # Perform the merge of the selected pair at min_idx and min_idx + 1
            c1 = self._centroids[min_idx]
            c2 = self._centroids[min_idx + 1]

            merged_weight = c1.weight + c2.weight

            # Create the merged centroid. merged_weight should be > 0 here
            # in standard operation as weights are non-negative.
            if merged_weight > 0:
                merged_mean = (
                    (c1.mean * c1.weight) + (c2.mean * c2.weight)
                ) / merged_weight
                merged_centroid = _Centroid(mean=merged_mean, weight=merged_weight)

                # Replace the two original centroids with the new merged one.
                # This replacement maintains the overall sorted order of the list,
                # as c1.mean <= merged_mean <= c2.mean.
                self._centroids[min_idx : min_idx + 2] = [merged_centroid]
            else:
                # Handle the unlikely case of merging two zero-weight centroids.
                # Remove both original centroids.
                del self._centroids[min_idx + 1]  # Delete second first (index shifts)
                del self._centroids[min_idx]  # Then delete the first

    def query(self, quantile: float) -> float:
        """
        Estimates the value at the given quantile.

        Args:
            quantile (float): The target quantile (must be between 0.0 and 1.0).

        Returns:
            float: The estimated value at the specified quantile. Returns NaN
                   if the sketch is empty or contains no valid data.

        Raises:
            ValueError: If the quantile is outside the range [0, 1].
        """
        if not (0.0 <= quantile <= 1.0):
            raise ValueError("Quantile must be between 0.0 and 1.0")

        # Ensure all buffered data is incorporated into centroids before querying.
        self._process_buffer()

        # Handle empty sketch case.
        if not self._centroids or self.total_weight == 0:
            return float("nan")

        # Handle edge quantiles precisely using tracked min/max.
        if quantile == 0.0:
            # Return minimum value seen, or NaN if none recorded (e.g., only NaNs added).
            return self._min_val if self._min_val is not None else float("nan")
        if quantile == 1.0:
            # Return maximum value seen, or NaN if none recorded.
            return self._max_val if self._max_val is not None else float("nan")

        # Calculate the target cumulative weight based on the quantile.
        target_weight = quantile * self.total_weight

        # Find the centroids bracketing the target cumulative weight.
        cumulative_weight = 0.0
        for i, c in enumerate(self._centroids):
            # Calculate the cumulative weight corresponding to the *center* (mean)
            # of the current centroid. Assumes weight is distributed uniformly
            # around the mean for interpolation purposes.
            centroid_mid_weight = cumulative_weight + c.weight / 2.0

            # Check if the target weight falls at or before the middle of this centroid.
            # Also ensure the centroid has weight, otherwise it cannot contain the target.
            if target_weight <= centroid_mid_weight and c.weight > 0:

                if i == 0:
                    # --- Special case: Target falls within the first centroid ---
                    # Interpolate linearly between the sketch's minimum value
                    # (`self._min_val`) and the mean of this first centroid (`c.mean`).
                    # Calculate the fraction based on the target weight's position
                    # within the first half of this centroid's weight.
                    fraction = target_weight / (c.weight / 2.0)
                    fraction = max(0.0, min(1.0, fraction))  # Clamp for robustness

                    # Use min_val if available, otherwise use the centroid's mean
                    # as the lower bound (handles case where min_val wasn't set).
                    lower_bound = self._min_val if self._min_val is not None else c.mean
                    return lower_bound + fraction * (c.mean - lower_bound)

                else:
                    # --- General case: Target falls within centroid i (i > 0) ---
                    # Interpolate linearly between the means of the previous
                    # centroid (c_{i-1}) and the current one (c_i).
                    prev_c = self._centroids[i - 1]

                    # Calculate cumulative weight at the midpoint of the *previous* centroid.
                    # Weight accumulated *before* prev_c starts:
                    prev_cumulative_start_weight = cumulative_weight - prev_c.weight
                    prev_centroid_mid_weight = (
                        prev_cumulative_start_weight + prev_c.weight / 2.0
                    )

                    # Total weight span between the midpoints of prev_c and c.
                    weight_diff = centroid_mid_weight - prev_centroid_mid_weight

                    # Avoid division by zero if midpoints coincide (e.g., zero-weight centroid).
                    if weight_diff <= 1e-9:  # Use tolerance for float comparison
                        return c.mean  # Return current mean if interval is negligible

                    # Calculate the fraction of the way the target_weight is
                    # between the previous midpoint and the current midpoint.
                    fraction = (target_weight - prev_centroid_mid_weight) / weight_diff
                    fraction = max(0.0, min(1.0, fraction))  # Clamp for robustness

                    # Perform linear interpolation between the means.
                    return prev_c.mean + fraction * (c.mean - prev_c.mean)

            # Accumulate the weight of the current centroid for the next iteration.
            # Only add positive weight.
            if c.weight > 0:
                cumulative_weight += c.weight

        # If the loop completes, target must be effectively quantile=1.0.
        # Return max_val as the best estimate. This fallback handles potential
        # floating point inaccuracies near q=1.0.
        return self._max_val if self._max_val is not None else float("nan")

    def merge(self: TDigestType, other: TDigestType) -> TDigestType:
        """
        Merges another TDigest sketch into this one.

        Creates a *new* TDigest instance representing the combined stream.
        The original sketches are not modified. Both sketches must have the
        same configuration (compression factor).

        Args:
            other (TDigest): The other TDigest sketch to merge.

        Returns:
            TDigest: A new TDigest sketch containing data from both inputs.

        Raises:
            TypeError: If 'other' is not a TDigest instance.
            ValueError: If the compression factors of the sketches do not match.
        """
        self._check_same_type(other)  # Checks if other is TDigest

        if self.compression != other.compression:
            raise ValueError(
                "Cannot merge TDigest sketches with different compression factors: "
                f"{self.compression} != {other.compression}"
            )

        # Create a new sketch instance for the merged result.
        merged_sketch = TDigest(compression=self.compression)

        # Ensure buffers of both sketches are processed before combining.
        # Process self (modifies self in place, but state is copied below).
        self._process_buffer()
        # Process other (must not modify other). We achieve this by adding its
        # buffer and centroids to the new sketch and letting its process/compress handle it.

        # Combine centroids and buffer from both sketches into the new sketch.
        # Note: Creates copies of the lists, originals are untouched.
        merged_sketch._centroids = self._centroids + other._centroids
        merged_sketch._unmerged_buffer = self._unmerged_buffer + other._unmerged_buffer

        # Combine scalar state.
        merged_sketch.total_weight = self.total_weight + other.total_weight
        # Use base class helper for combining items processed.
        merged_sketch._items_processed = self.items_processed + other.items_processed

        # Combine min/max values safely.
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

        # Process the combined buffer and compress the combined centroids in the new sketch.
        # This merges buffered points and compresses the potentially oversized centroid list.
        merged_sketch._process_buffer()

        # An extra compression might be needed if both buffers were empty but
        # combining centroids exceeded the limit. _process_buffer calls _compress,
        # covering most cases, but a final check ensures the constraint.
        # This is slightly redundant if buffer processing always compresses, but safe.
        if len(merged_sketch._centroids) > merged_sketch.compression:
            merged_sketch._compress()  # Ensure final compression if needed

        return merged_sketch

        # Inside the TDigest class...

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the TDigest sketch state to a dictionary.

        Processes the buffer before serialization to ensure consistent state.

        Returns:
            Dict[str, Any]: A dictionary containing the sketch's configuration
                            and internal state (centroids, weights, min/max).
        """
        # Ensure buffer is processed for a complete snapshot of centroids.
        self._process_buffer()

        state = self._base_dict()  # Get base class info (class name, items_processed)

        state.update(
            {
                "compression": self.compression,
                "total_weight": self.total_weight,
                "min_val": self._min_val,
                "max_val": self._max_val,
                "centroids": [c.to_dict() for c in self._centroids],
                # Buffer is processed, so it should be empty and not serialized.
            }
        )
        return state

    @classmethod
    def from_dict(cls: Type[TDigestType], data: Dict[str, Any]) -> TDigestType:
        """
        Deserializes a TDigest sketch from a dictionary representation.

        Args:
            data (Dict[str, Any]): The dictionary created by to_dict().

        Returns:
            TDigest: A new TDigest instance restored from the dictionary.

        Raises:
            ValueError: If the dictionary is missing required keys or has
                        incompatible/invalid data.
        """
        # Basic validation of required keys.
        required_keys = {"compression", "total_weight", "centroids", "items_processed"}
        if "type" not in data:
            raise ValueError(
                "Invalid dictionary format for TDigest. Missing required key: 'type'"
            )

        if data.get("type") != cls.__name__:
            raise ValueError(
                f"Dictionary represents class '{data.get('type')}' but expected '{cls.__name__}'"
            )

        required_tdigest_keys = {
            "compression",
            "total_weight",
            "centroids",
            "items_processed",
        }

        missing_keys = required_tdigest_keys - data.keys()
        if missing_keys:
            raise ValueError(
                f"Invalid dictionary format for TDigest. Missing keys: {missing_keys}"
            )

        instance = cls(compression=data["compression"])
        instance._items_processed = data["items_processed"]
        instance.total_weight = data["total_weight"]
        # Use .get() for optional min/max which might be None.
        instance._min_val = data.get("min_val")
        instance._max_val = data.get("max_val")

        # Deserialize centroids using the _Centroid helper class method.
        try:
            instance._centroids = [
                _Centroid.from_dict(c_data) for c_data in data["centroids"]
            ]
        except (ValueError, KeyError) as e:
            raise ValueError(f"Error deserializing centroids: {e}") from e

        # Ensure buffer size is consistent with loaded compression factor.
        instance._buffer_size = max(
            10, cls.DEFAULT_BUFFER_FACTOR * instance.compression
        )
        # Buffer itself is assumed empty after serialization, so initialize as empty.
        instance._unmerged_buffer = []

        # Ensure centroids are sorted after loading, as query/compress rely on order.
        instance._centroids.sort()

        # Add consistency check: total weight should roughly match sum of centroid weights
        calculated_weight = sum(c.weight for c in instance._centroids)
        if not math.isclose(instance.total_weight, calculated_weight, rel_tol=1e-6):
            print(
                f"Warning: Loaded total_weight ({instance.total_weight}) does not closely match sum of centroid weights ({calculated_weight}). Data might be inconsistent."
            )
            # Option: Correct total_weight based on centroids?
            # instance.total_weight = calculated_weight

        return instance

    def estimate_size(self) -> int:
        """
        Estimates the memory footprint of the TDigest sketch in bytes.

        Provides a rough estimate based on internal data structures.

        Returns:
            int: An estimated size in bytes.
        """
        # Base size from StreamSummary (object overhead, items_processed).
        size = super().estimate_size()

        # Size of configuration and state attributes.
        size += sys.getsizeof(self.compression)
        size += sys.getsizeof(self.total_weight)
        size += sys.getsizeof(self._buffer_size)
        if self._min_val is not None:
            size += sys.getsizeof(self._min_val)
        if self._max_val is not None:
            size += sys.getsizeof(self._max_val)

        # Size of the centroids list (list overhead + size of each centroid object).
        size += sys.getsizeof(self._centroids)  # List object overhead.
        # Estimate size per centroid (object overhead + 2 floats).
        # sys.getsizeof on __slots__ objects can be inaccurate, this is an approx.
        if self._centroids:
            # Size of the references in the list + size of each actual centroid object.
            # Manual sum might be slightly more accurate than multiplying by one size.
            size += sum(sys.getsizeof(c) for c in self._centroids)
            # Add size of the floats *within* each centroid (getsizeof includes object overhead per float).
            size += sum(
                sys.getsizeof(c.mean) + sys.getsizeof(c.weight) for c in self._centroids
            )

        # Size of the unmerged buffer (list overhead + size of floats).
        size += sys.getsizeof(self._unmerged_buffer)  # List object overhead.
        if self._unmerged_buffer:
            # Size of references + size of each float object.
            size += sum(sys.getsizeof(item) for item in self._unmerged_buffer)

        return size

    def __len__(self) -> int:
        """Returns the total number of items processed by the sketch."""
        return self.items_processed

    @property
    def is_empty(self) -> bool:
        """Checks if the sketch has processed any items."""
        return self.items_processed == 0

    # Optional helper for debugging or inspection.
    def get_centroids(self) -> List[Tuple[float, float]]:
        """Returns the current centroids as (mean, weight) tuples, sorted."""
        self._process_buffer()  # Ensure up-to-date representation
        return [(c.mean, c.weight) for c in sorted(self._centroids)]
