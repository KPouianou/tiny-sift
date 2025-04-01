# tiny_sift/algorithms/reservoir.py

"""
Reservoir Sampling implementation for TinySift.

This module provides implementations of Reservoir Sampling and its variants
for maintaining a random sample of a data stream with bounded memory.
Includes benchmarking hooks for performance and statistical analysis.
"""

import math
import random
import sys
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union
import bisect  # Used for efficient insertion in weighted sampling

from tiny_sift.core.base import StreamSummary, SampleMaintainer

T = TypeVar("T")  # Type for the items being processed


class ReservoirSampling(SampleMaintainer[T]):
    """
    Standard Reservoir Sampling algorithm (Algorithm R).

    Maintains a uniform random sample of fixed size from a stream of unknown length.
    Uses O(k) memory where k is the reservoir size. Includes benchmarking hooks.

    References:
        - Vitter, J. S. (1985). Random sampling with a reservoir.
          ACM Transactions on Mathematical Software, 11(1), 37-57.
    """

    def __init__(
        self,
        size: int,
        memory_limit_bytes: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize a new reservoir sampler.

        Args:
            size: The size of the reservoir (number of samples to maintain).
            memory_limit_bytes: Optional maximum memory usage in bytes.
            seed: Optional random seed for reproducibility.

        Raises:
            ValueError: If size is less than 1.
        """
        super().__init__(memory_limit_bytes)

        if size < 1:
            raise ValueError("Reservoir size must be at least 1")

        self._size = size
        self._reservoir: List[T] = []
        self._random = random.Random(seed)

    def update(self, item: T) -> None:
        """
        Update the reservoir with a new item from the stream.

        Uses Algorithm R: If the reservoir isn't full yet, add the item.
        Otherwise, replace a random item with probability size/items_processed.

        Args:
            item: The new item to process.
        """
        super().update(item)  # Increment items_processed and handles timing

        if len(self._reservoir) < self._size:
            # Reservoir not full yet, add item
            self._reservoir.append(item)
        else:
            # Randomly decide whether to replace an item
            # Algorithm R: choose index j uniformly from [0, items_processed - 1]
            j = self._random.randrange(self._items_processed)
            if j < self._size:
                self._reservoir[j] = item

    def query(self, *args: Any, **kwargs: Any) -> List[T]:
        """
        Query the current state of the reservoir.

        Returns:
            The current reservoir sample.
        """
        return self.get_sample()

    def get_sample(self) -> List[T]:
        """
        Get the current reservoir sample.

        Returns:
            A list containing the current sample.
        """
        return self._reservoir.copy()

    def merge(self, other: "ReservoirSampling[T]") -> "ReservoirSampling[T]":
        """
        Merge this reservoir with another reservoir.

        Note: This implementation uses a resampling heuristic that combines both
        reservoirs and then takes a uniform random sample if the combined size
        exceeds the target size. This is a simple approach that works well for many
        applications but does not perfectly preserve the statistical properties
        of reservoir sampling across the combined stream history. For statistically
        accurate merging, more advanced algorithms are required.

        Args:
            other: Another ReservoirSampling object.

        Returns:
            A new merged ReservoirSampling object.

        Raises:
            TypeError: If other is not a ReservoirSampling object.
            ValueError: If reservoir sizes differ.
        """
        self._check_same_type(other)
        if self._size != other._size:
            raise ValueError(
                f"Cannot merge reservoirs with different sizes: "
                f"{self._size} and {other._size}"
            )

        # Create a new reservoir with the same size and a new random state
        result = ReservoirSampling[T](
            self._size,
            self._memory_limit_bytes,
            seed=self._random.randint(0, 2**32 - 1),
        )

        # Combine the two reservoirs
        combined = self._reservoir + other._reservoir

        # Update the items processed count *before* potentially reducing the sample
        result._items_processed = self._combine_items_processed(other)

        # If the combined size is less than or equal to the target size,
        # just use all of them
        if len(combined) <= self._size:
            result._reservoir = combined
        else:
            # Resample uniformly from the combined pool to maintain the correct size.
            # Note: This approach loses history weighting.
            result._reservoir = result._random.sample(combined, self._size)

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the reservoir to a dictionary for serialization.

        Returns:
            A dictionary representation of the reservoir.
        """
        data = self._base_dict()
        data.update(
            {
                "size": self._size,
                "reservoir": self._reservoir,
                # Note: Random state is not serialized to JSON.
                # Deserialization will create a new random state.
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReservoirSampling[T]":
        """
        Create a reservoir from a dictionary representation.

        Args:
            data: The dictionary containing the reservoir state.

        Returns:
            A new ReservoirSampling object.
        """
        # Create a new reservoir. Seed cannot be restored perfectly from JSON.
        reservoir = cls(
            size=data["size"], memory_limit_bytes=data.get("memory_limit_bytes")
        )

        # Restore the state
        reservoir._reservoir = data["reservoir"]
        reservoir._items_processed = data["items_processed"]

        return reservoir

    def estimate_size(self) -> int:
        """
        Estimate the current memory usage of this reservoir in bytes.
        Provides a more accurate estimate than the base class.

        Returns:
            Estimated memory usage in bytes.
        """
        # Start with base object size (includes attributes, dict, etc.)
        size = super().estimate_size()

        # Add size of the reservoir list object itself
        size += sys.getsizeof(self._reservoir)

        # Add approximate size of all items currently in the reservoir
        # This is an estimate; actual memory can vary based on object sharing etc.
        for item in self._reservoir:
            size += sys.getsizeof(item)

        # Add size of the random state object
        if hasattr(self, "_random"):
            size += sys.getsizeof(self._random)

        return size

    def get_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the current state of the reservoir,
        including sample size and fullness.

        Returns:
            A dictionary containing various statistics about the reservoir state.
        """
        stats = (
            super().get_stats()
        )  # Includes type, items_processed, memory, performance

        current_size = len(self._reservoir)
        max_size = self._size

        stats.update(
            {
                "sample_size": current_size,
                "max_sample_size": max_size,
                "sample_fullness_pct": (
                    (current_size / max_size) * 100 if max_size > 0 else 0
                ),
            }
        )

        # Example: Add stats about the sample data itself if numeric
        # if current_size > 0 and all(isinstance(x, (int, float)) for x in self._reservoir):
        #     numeric_sample = [x for x in self._reservoir if isinstance(x, (int, float))]
        #     stats['sample_min'] = min(numeric_sample)
        #     stats['sample_max'] = max(numeric_sample)
        #     stats['sample_mean'] = sum(numeric_sample) / current_size
        #     if current_size > 1:
        #         variance = sum((x - stats['sample_mean']) ** 2 for x in numeric_sample) / (current_size - 1)
        #         stats['sample_stdev'] = math.sqrt(variance)

        return stats

    def error_bounds(self) -> Dict[str, Union[str, float]]:
        """
        Describe the theoretical properties of the sampling process.

        For standard reservoir sampling, the key property is uniformity.

        Returns:
            A dictionary describing the sampling characteristics.
        """
        # Calculate the current inclusion probability for an item processed so far
        if self._items_processed == 0:
            prob_str = "N/A (empty stream)"
        elif self._items_processed <= self._size:
            prob_str = "1.0 (reservoir not full)"
        else:
            # Each item seen has prob size/items_processed of being in the final sample
            prob = self._size / self._items_processed
            prob_str = f"{self._size}/{self._items_processed} (≈{prob:.4f})"

        return {
            "sampling_type": "Uniform Reservoir (Algorithm R)",
            "property": "Each stream item processed so far has an equal probability of being in the final sample.",
            "inclusion_probability": prob_str,
        }

    def assess_representativeness(self) -> Dict[str, Any]:
        """
        Provides basic statistics about the sample itself.

        Assessing true representativeness requires comparing the sample's
        distribution to the full stream's distribution, which is not possible
        in a pure streaming context without storing the entire stream or using
        more advanced techniques.

        This method returns statistics *about the current sample* which can
        offer some insights, but should not be solely relied upon to determine
        if the sample accurately reflects the stream characteristics.

        Returns:
            A dictionary containing statistics about the sample distribution.
        """
        sample = self._reservoir
        current_size = len(sample)

        stats = {
            "assessment_type": "Sample Internal Statistics",
            "limitation": "Does not compare sample to full stream distribution.",
            "sample_size": current_size,
            "max_sample_size": self._size,
        }

        if current_size == 0:
            stats["message"] = "Sample is empty."
            return stats

        # Check if items are numeric for detailed stats
        try:
            # Attempt numeric conversion cautiously
            numeric_sample = [float(x) for x in sample]
            is_numeric = True
        except (ValueError, TypeError):
            is_numeric = False

        if is_numeric:
            stats["sample_min"] = min(numeric_sample)
            stats["sample_max"] = max(numeric_sample)
            sample_mean = sum(numeric_sample) / current_size
            stats["sample_mean"] = sample_mean
            if current_size > 1:
                variance = sum((x - sample_mean) ** 2 for x in numeric_sample) / (
                    current_size - 1
                )
                stats["sample_stdev"] = math.sqrt(variance)
            else:
                stats["sample_stdev"] = 0.0
            stats["data_type_assessed"] = "numeric"
        else:
            # For non-numeric data, check value diversity if hashable
            try:
                unique_items_in_sample = len(set(sample))
                stats["unique_items_in_sample"] = unique_items_in_sample
                stats["sample_diversity_ratio"] = (
                    unique_items_in_sample / current_size if current_size > 0 else 0.0
                )
                stats["data_type_assessed"] = "hashable"
            except TypeError:
                stats["message"] = (
                    "Sample contains non-hashable, non-numeric items. Limited stats available."
                )
                stats["data_type_assessed"] = "mixed/unhashable"

        return stats


class WeightedReservoirSampling(ReservoirSampling[T]):
    """
    Weighted Reservoir Sampling using exponential jumps (Algorithm A-Exp-Res).

    Maintains a random sample with probability proportional to item weights.
    Uses an efficient key-based approach where the key for each item is
    random^(1/weight), and keeps items with the highest keys.
    Includes benchmarking hooks.

    References:
        - Efraimidis, P. S., & Spirakis, P. G. (2006). Weighted random sampling with a reservoir.
          Information Processing Letters, 97(5), 181-185.
    """

    def __init__(
        self,
        size: int,
        memory_limit_bytes: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize a new weighted reservoir sampler.

        Args:
            size: The size of the reservoir (number of samples to maintain).
            memory_limit_bytes: Optional maximum memory usage in bytes.
            seed: Optional random seed for reproducibility.
        """
        # Call StreamSummary init directly, as this class manages its own reservoir structure
        StreamSummary.__init__(self, memory_limit_bytes)

        if size < 1:
            raise ValueError("Reservoir size must be at least 1")

        self._size = size
        self._random = random.Random(seed)

        # Store tuples of (item, weight, key)
        self._weighted_reservoir: List[Tuple[T, float, float]] = []
        # Total weight of all items *seen* in the stream
        self._total_stream_weight = 0.0
        # Smallest key currently in the reservoir (if full)
        self._min_key_in_reservoir: Optional[float] = None

    def update(self, item: T, weight: float = 1.0) -> None:
        """
        Update the weighted reservoir with a new item from the stream.

        Uses Algorithm A-Exp-Res: Assign a key to each item based on a random value
        and its weight, then keep the items with the largest keys.

        Args:
            item: The new item to process.
            weight: The weight of the item (default is 1.0).

        Raises:
            ValueError: If weight is not positive.
        """
        if weight <= 0:
            raise ValueError("Weight must be positive")

        # Increment items_processed and handle timing via StreamSummary explicitly
        StreamSummary.update(self, item)

        # Update total weight seen in the stream
        self._total_stream_weight += weight

        # Generate a random key: key = random^(1/weight)
        u = self._random.random()  # random float in [0.0, 1.0)
        if u == 0.0:  # Avoid math domain error for u=0
            key = float("inf")  # Effectively highest possible key
        else:
            key = u ** (1.0 / weight)

        # Case 1: Reservoir is not full
        if len(self._weighted_reservoir) < self._size:
            self._weighted_reservoir.append((item, weight, key))
            # If it just became full, sort by key (descending) and find min key
            if len(self._weighted_reservoir) == self._size:
                self._weighted_reservoir.sort(key=lambda x: x[2], reverse=True)
                self._min_key_in_reservoir = self._weighted_reservoir[-1][2]

        # Case 2: Reservoir is full, check if new item's key is large enough
        elif (
            self._min_key_in_reservoir is not None and key > self._min_key_in_reservoir
        ):
            # Replace the item with the smallest key (at the end of sorted list)
            self._weighted_reservoir.pop()
            # Insert the new item while maintaining descending sorted order by key
            # Use bisect for efficiency O(log k + k) vs O(k log k) for sort
            entry_to_insert = (item, weight, key)
            # Find insertion point using key
            index_to_insert = bisect.bisect_left(
                [
                    -tpl[2] for tpl in self._weighted_reservoir
                ],  # Search on negative keys for descending sort bisect_left
                -key,  # Search for negative new key
            )
            self._weighted_reservoir.insert(index_to_insert, entry_to_insert)
            # Update the minimum key tracker
            self._min_key_in_reservoir = self._weighted_reservoir[-1][2]

    def get_sample(self) -> List[T]:
        """
        Get the current weighted reservoir sample (items only).

        Returns:
            A list containing the current sample items.
        """
        return [item for item, _, _ in self._weighted_reservoir]

    def merge(
        self, other: "WeightedReservoirSampling[T]"
    ) -> "WeightedReservoirSampling[T]":
        """
        Merge this weighted reservoir with another weighted reservoir.

        Combines items based on their calculated keys, ensuring the result
        maintains the weighted sampling property.

        Args:
            other: Another WeightedReservoirSampling object.

        Returns:
            A new merged WeightedReservoirSampling object.

        Raises:
            TypeError: If other is not a WeightedReservoirSampling object.
            ValueError: If reservoir sizes differ.
        """
        self._check_same_type(other)
        if self._size != other._size:
            raise ValueError(
                f"Cannot merge weighted reservoirs with different sizes: "
                f"{self._size} and {other._size}"
            )

        # Create a new reservoir with the same size and a new random state
        result = WeightedReservoirSampling[T](
            self._size,
            self._memory_limit_bytes,
            seed=self._random.randint(0, 2**32 - 1),
        )

        # Combine the weighted reservoirs' internal tuples
        combined = self._weighted_reservoir + other._weighted_reservoir

        # Sort the combined list by key (descending)
        combined.sort(key=lambda x: x[2], reverse=True)

        # Keep only the top 'size' elements based on keys
        result._weighted_reservoir = combined[: self._size]
        if result._weighted_reservoir:
            result._min_key_in_reservoir = result._weighted_reservoir[-1][2]

        # Update total stream weight and items processed count
        result._total_stream_weight = (
            self._total_stream_weight + other._total_stream_weight
        )
        result._items_processed = self._combine_items_processed(other)

        return result

    def get_weighted_sample(self) -> List[Tuple[T, float]]:
        """
        Get the current weighted reservoir sample with their weights.

        Returns:
            A list of (item, weight) tuples.
        """
        return [(item, weight) for item, weight, _ in self._weighted_reservoir]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the weighted reservoir to a dictionary for serialization.

        Returns:
            A dictionary representation of the weighted reservoir.
        """
        data = self._base_dict()
        data["type"] = "WeightedReservoirSampling"  # Explicitly set type
        data.update(
            {
                "size": self._size,
                "weighted_reservoir": self._weighted_reservoir,  # Includes item, weight, key
                "total_stream_weight": self._total_stream_weight,
                "min_key_in_reservoir": self._min_key_in_reservoir,
                # Note: Random state is not serialized.
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeightedReservoirSampling[T]":
        """
        Create a weighted reservoir from a dictionary representation.

        Args:
            data: The dictionary containing the weighted reservoir state.

        Returns:
            A new WeightedReservoirSampling object.
        """
        # Creates new random state
        reservoir = cls(
            size=data["size"], memory_limit_bytes=data.get("memory_limit_bytes")
        )

        # Restore the state
        reservoir._weighted_reservoir = [
            tuple(entry) for entry in data["weighted_reservoir"]
        ]
        reservoir._total_stream_weight = data["total_stream_weight"]
        reservoir._items_processed = data["items_processed"]
        reservoir._min_key_in_reservoir = data.get("min_key_in_reservoir")

        return reservoir

    def estimate_size(self) -> int:
        """
        Estimate the current memory usage of this weighted reservoir in bytes.
        Provides a more accurate estimate for the weighted structure.

        Returns:
            Estimated memory usage in bytes.
        """
        # Start with base object size
        size = StreamSummary.estimate_size(self)

        # Add size of the weighted reservoir list object itself
        size += sys.getsizeof(self._weighted_reservoir)
        # Add size of tracked attributes
        size += sys.getsizeof(self._total_stream_weight)
        if self._min_key_in_reservoir is not None:
            size += sys.getsizeof(self._min_key_in_reservoir)

        # Add approximate size of all tuples in the weighted reservoir
        tuple_overhead_estimate = 64  # Approximate overhead per tuple
        for item, weight, key in self._weighted_reservoir:
            size += sys.getsizeof(item) + sys.getsizeof(weight) + sys.getsizeof(key)
            size += tuple_overhead_estimate

        # Add size of the random state object
        if hasattr(self, "_random"):
            size += sys.getsizeof(self._random)

        return size

    def get_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the current state of the weighted reservoir,
        including sample size, fullness, and weight distribution information.

        Returns:
            A dictionary containing various statistics about the reservoir state.
        """
        # Start with base StreamSummary stats (processing count, memory, perf)
        stats = StreamSummary.get_stats(self)

        current_size = len(self._weighted_reservoir)
        max_size = self._size

        stats.update(
            {
                "sample_size": current_size,
                "max_sample_size": max_size,
                "sample_fullness_pct": (
                    (current_size / max_size) * 100 if max_size > 0 else 0
                ),
                "total_stream_weight": self._total_stream_weight,
            }
        )

        # Add weight-specific stats if sample is not empty
        if current_size > 0:
            weights_in_sample = [w for _, w, _ in self._weighted_reservoir]
            total_weight_in_sample = sum(weights_in_sample)
            stats["total_weight_in_sample"] = total_weight_in_sample
            stats["min_weight_in_sample"] = min(weights_in_sample)
            stats["max_weight_in_sample"] = max(weights_in_sample)
            stats["avg_weight_in_sample"] = total_weight_in_sample / current_size

            # Add key statistics (useful for diagnosing sampling issues)
            keys_in_sample = [k for _, _, k in self._weighted_reservoir]
            stats["min_key_in_sample"] = min(keys_in_sample)
            stats["max_key_in_sample"] = max(keys_in_sample)
            if self._min_key_in_reservoir is not None:
                stats["tracked_min_key"] = self._min_key_in_reservoir

        return stats

    def error_bounds(self) -> Dict[str, Union[str, float]]:
        """
        Describe the theoretical properties of the weighted sampling process.

        Returns:
            A dictionary describing the sampling characteristics.
        """
        return {
            "sampling_type": "Weighted Reservoir (A-Exp-Res)",
            "property": "Inclusion probability is proportional to item weight.",
            # Note: Quantifying the exact probability requires tracking complex history.
        }

    def assess_representativeness(self) -> Dict[str, Any]:
        """
        Provides basic statistics about the weighted sample itself.

        Assessing true representativeness requires comparing the sample's
        properties (considering weights) to the full stream's properties.
        This method provides statistics *about the current sample* and its weights.

        Returns:
            A dictionary containing statistics about the sample and weight distribution.
        """
        sample_tuples = self._weighted_reservoir
        current_size = len(sample_tuples)

        stats = {
            "assessment_type": "Weighted Sample Internal Statistics",
            "limitation": "Does not compare sample to full stream distribution.",
            "sample_size": current_size,
            "max_sample_size": self._size,
        }

        if current_size == 0:
            stats["message"] = "Sample is empty."
            return stats

        # Weight distribution in sample
        weights_in_sample = [w for _, w, _ in sample_tuples]
        total_weight_in_sample = sum(weights_in_sample)
        stats["total_weight_in_sample"] = total_weight_in_sample
        stats["min_weight_in_sample"] = min(weights_in_sample)
        stats["max_weight_in_sample"] = max(weights_in_sample)
        stats["avg_weight_in_sample"] = total_weight_in_sample / current_size
        if current_size > 1:
            avg_w = stats["avg_weight_in_sample"]
            weight_variance = sum((w - avg_w) ** 2 for w in weights_in_sample) / (
                current_size - 1
            )
            stats["stdev_weight_in_sample"] = math.sqrt(weight_variance)
        else:
            stats["stdev_weight_in_sample"] = 0.0

        # Comparison of weight ratios (crude check for bias)
        if self._total_stream_weight > 0:
            stats["sample_weight_fraction"] = (
                total_weight_in_sample / self._total_stream_weight
            )
        if self._items_processed > 0:
            stats["sample_count_fraction"] = current_size / self._items_processed

        # Check if items are numeric for value stats
        try:
            numeric_sample_values = [float(item) for item, _, _ in sample_tuples]
            is_numeric = True
        except (ValueError, TypeError):
            is_numeric = False

        if is_numeric:
            stats["sample_min_value"] = min(numeric_sample_values)
            stats["sample_max_value"] = max(numeric_sample_values)
            sample_value_mean = sum(numeric_sample_values) / current_size
            stats["sample_mean_value"] = sample_value_mean  # Unweighted mean
            # Weighted mean calculation: sum(value * weight) / sum(weight)
            weighted_sum = sum(float(item) * w for item, w, _ in sample_tuples)
            stats["sample_weighted_mean_value"] = (
                weighted_sum / total_weight_in_sample
                if total_weight_in_sample > 0
                else 0.0
            )
            stats["data_type_assessed"] = "numeric"
        else:
            # Check diversity if items are hashable
            try:
                unique_items_in_sample = len(set(item for item, _, _ in sample_tuples))
                stats["unique_items_in_sample"] = unique_items_in_sample
                stats["sample_diversity_ratio"] = (
                    unique_items_in_sample / current_size if current_size > 0 else 0.0
                )
                stats["data_type_assessed"] = "hashable"
            except TypeError:
                stats["message"] = "Sample contains non-hashable, non-numeric items."
                stats["data_type_assessed"] = "mixed/unhashable"

        return stats
