"""
Space-Saving algorithm implementation for TinySift.

This module provides an implementation of the Space-Saving algorithm,
a streaming algorithm for finding the top-k most frequent elements in a data stream
with guaranteed error bounds and bounded memory usage.

The Space-Saving algorithm efficiently identifies frequent items with the following guarantees:
1. Space Complexity: O(k) where k is the number of counters
2. Update Time: O(1) amortized time per update
3. Error Bound: Maximum error in frequency estimation is at most N/k,
                where N is the total count and k is the number of counters

References:
    - Metwally, A., Agrawal, D., & El Abbadi, A. (2005).
      Efficient computation of frequent and top-k elements in data streams.
      In International Conference on Database Theory (pp. 398-412).
"""

import heapq
import math
import sys
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, TypeVar, cast

from tiny_sift.core.base import FrequencyEstimator

T = TypeVar("T")  # Type for the items being processed


class CounterEntry:
    """
    An entry in the Space-Saving counter table.

    Each entry tracks an item, its estimated count, and a maximum error.
    The error represents the maximum possible overestimation of the item's frequency.
    """

    __slots__ = ["item", "count", "error"]

    def __init__(self, item: T, count: int = 1, error: int = 0):
        """
        Initialize a new counter entry.

        Args:
            item: The item being counted.
            count: The estimated count of the item (default: 1).
            error: The maximum error in the count (default: 0).
        """
        self.item = item
        self.count = count
        self.error = error

    def __lt__(self, other: "CounterEntry") -> bool:
        """
        Compare entries based on count for use in a min-heap.

        Args:
            other: Another CounterEntry to compare with.

        Returns:
            True if this entry's count is less than other's count.
        """
        return self.count < other.count

    def __eq__(self, other: object) -> bool:
        """
        Check if two entries are equal based on item.

        Args:
            other: Another object to compare with.

        Returns:
            True if other is a CounterEntry and has the same item.
        """
        if not isinstance(other, CounterEntry):
            return False
        return self.item == other.item

    def __hash__(self) -> int:
        """
        Get hash value based on the item.

        Returns:
            Hash value of the item.
        """
        return hash(self.item)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entry to a dictionary for serialization.

        Returns:
            A dictionary representation of the entry.
        """
        return {"item": self.item, "count": self.count, "error": self.error}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CounterEntry":
        """
        Create an entry from a dictionary representation.

        Args:
            data: Dictionary containing the entry data.

        Returns:
            A new CounterEntry initialized from the dictionary.
        """
        return cls(item=data["item"], count=data["count"], error=data["error"])

    def __repr__(self) -> str:
        """
        Get string representation of the entry.

        Returns:
            String representation of the entry.
        """
        return f"CounterEntry(item={self.item}, count={self.count}, error={self.error})"


class SpaceSaving(FrequencyEstimator[T]):
    """
    Space-Saving algorithm for finding frequent items in a stream.

    The Space-Saving algorithm efficiently tracks the approximate frequencies of items
    in a data stream using bounded memory. It provides guaranteed error bounds and
    is particularly useful for finding heavy hitters and top-k frequent items.

    The algorithm works by maintaining a fixed-size set of counters. When a new item
    is seen that doesn't have a counter, it replaces the item with the smallest count,
    and the error for the new item is set to the count of the replaced item.

    The maximum count error for any item is limited by N/m, where N is the total count
    of all items and m is the number of counters.

    References:
        - Metwally, A., Agrawal, D., & El Abbadi, A. (2005).
          Efficient computation of frequent and top-k elements in data streams.
    """

    def __init__(
        self,
        capacity: int = 100,
        memory_limit_bytes: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize a new Space-Saving counter.

        Args:
            capacity: The maximum number of items to track (default: 100).
            memory_limit_bytes: Optional maximum memory usage in bytes.
            seed: Optional random seed (not used, included for API consistency).

        Raises:
            ValueError: If capacity is less than 1.
        """
        super().__init__(memory_limit_bytes)

        if capacity < 1:
            raise ValueError("Capacity (number of counters) must be at least 1")

        self._capacity = capacity
        self._counters: List[CounterEntry] = []  # Min-heap of counters
        self._item_map: Dict[T, CounterEntry] = {}  # Map from items to their entries
        self._total_count = 0  # Total count of all items seen

    def update(self, item: T, count: int = 1) -> None:
        """
        Update the sketch with a new item from the stream.

        This implements the core Space-Saving algorithm logic:
        1. If the item is already being monitored, increment its counter
        2. If not and there's room, add a new counter for the item
        3. If not and there's no room, replace the item with minimum frequency,
           setting the new item's error to the old item's count

        Args:
            item: The new item to process.
            count: The count to add for this item (default: 1).

        Raises:
            ValueError: If count is negative.
        """
        if count < 0:
            raise ValueError("Count must be non-negative")

        if count == 0:
            return

        # Call parent class's update method to increment items_processed
        super().update(item)

        # Update the total count
        self._total_count += count

        # Case 1: Item already has a counter
        if item in self._item_map:
            entry = self._item_map[item]

            # To maintain heap property, we need to remove and re-insert the item
            # Note: This is O(log k) operation. A more efficient implementation might
            # use a different data structure for the heap to support direct updates.
            self._counters.remove(entry)
            entry.count += count
            heapq.heappush(self._counters, entry)

        # Case 2: Item doesn't have a counter
        else:
            # Case 2a: There's room for a new counter
            if len(self._counters) < self._capacity:
                entry = CounterEntry(item, count)
                self._item_map[item] = entry
                heapq.heappush(self._counters, entry)

            # Case 2b: No room, replace the minimum counter
            else:
                # Get the item with minimum frequency
                min_entry = self._counters[0]

                # Remove the minimum entry from tracking
                min_count = min_entry.count
                min_item = min_entry.item
                heapq.heappop(self._counters)
                del self._item_map[min_item]

                # Create a new entry for the current item
                # The error is set to the count of the replaced item
                entry = CounterEntry(item, min_count + count, min_count)
                self._item_map[item] = entry
                heapq.heappush(self._counters, entry)

    def query(self, *args: Any, **kwargs: Any) -> float:
        """
        Query the current state of the sketch.

        This is a convenience method that calls estimate_frequency.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The estimated frequency.
        """
        if len(args) > 0:
            return self.estimate_frequency(args[0])
        if "item" in kwargs:
            return self.estimate_frequency(kwargs["item"])
        raise ValueError("Missing required argument 'item'")

    def estimate_frequency(self, item: T) -> float:
        """
        Estimate the frequency of an item in the stream.

        Args:
            item: The item to estimate the frequency for.

        Returns:
            The estimated frequency of the item. Returns 0 if the item is not being tracked.
        """
        if item in self._item_map:
            return self._item_map[item].count
        return 0.0

    def estimate_frequency_error(self, item: T) -> Tuple[float, float]:
        """
        Estimate the frequency of an item along with its maximum error.

        Args:
            item: The item to estimate the frequency for.

        Returns:
            A tuple of (estimated_frequency, maximum_error).
            For items not being tracked, returns (0, N/k) where N is the total count
            and k is the capacity.
        """
        if item in self._item_map:
            entry = self._item_map[item]
            return (entry.count, entry.error)
        else:
            # For non-tracked items, the max error is the frequency of the min element
            # or the theoretical bound N/k, whichever is smaller
            if self._counters:
                min_count = self._counters[0].count
                return (0, min_count)
            return (0, 0)

    def get_heavy_hitters(self, threshold: float) -> Dict[T, float]:
        """
        Get items that appear more frequently than the threshold.

        Args:
            threshold: The minimum frequency ratio (0.0 to 1.0) to include.

        Returns:
            A dictionary mapping items to their estimated frequencies.

        Raises:
            ValueError: If threshold is not between 0 and 1.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        result = {}

        # Calculate the minimum count based on threshold
        min_count = threshold * self._total_count

        # Include all items with frequency above min_count
        for entry in self._counters:
            if entry.count >= min_count:
                result[entry.item] = entry.count

        return result

    def get_top_k(self, k: Optional[int] = None) -> List[Tuple[T, float]]:
        """
        Get the top-k most frequent items.

        Args:
            k: The number of top items to return. If None, returns all tracked items.

        Returns:
            A list of (item, frequency) tuples, sorted by frequency in descending order.
        """
        if k is None:
            k = len(self._counters)

        # Sort by count in descending order and take top k
        top_k = sorted(self._counters, key=lambda entry: entry.count, reverse=True)[:k]

        # Convert to (item, frequency) tuples
        return [(entry.item, entry.count) for entry in top_k]

    def merge(self, other: "SpaceSaving[T]") -> "SpaceSaving[T]":
        """
        Merge this sketch with another Space-Saving sketch.

        Args:
            other: Another Space-Saving sketch.

        Returns:
            A new merged Space-Saving sketch.

        Raises:
            TypeError: If other is not a SpaceSaving object.
            ValueError: If sketches have different capacities.
        """
        # Check that other is a SpaceSaving object
        self._check_same_type(other)

        # Check that capacities match
        if self._capacity != other._capacity:
            raise ValueError(
                f"Cannot merge sketches with different capacities: "
                f"{self._capacity} and {other._capacity}"
            )

        # Create a new sketch with the same capacity
        result = SpaceSaving[T](
            capacity=self._capacity, memory_limit_bytes=self._memory_limit_bytes
        )

        # Add all items from both sketches
        # This is a simple but not necessarily optimal merging strategy
        # First, add all items from this sketch
        for entry in self._counters:
            result.update(entry.item, entry.count)

        # Then add all items from the other sketch
        for entry in other._counters:
            result.update(entry.item, entry.count)

        # Update the total count and items processed
        result._total_count = self._total_count + other._total_count
        result._items_processed = self._combine_items_processed(other)

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
                "capacity": self._capacity,
                "total_count": self._total_count,
                "counters": [entry.to_dict() for entry in self._counters],
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpaceSaving[T]":
        """
        Create a sketch from a dictionary representation.

        Args:
            data: The dictionary containing the sketch state.

        Returns:
            A new SpaceSaving object.
        """
        # Create a new sketch
        sketch = cls(
            capacity=data["capacity"], memory_limit_bytes=data.get("memory_limit_bytes")
        )

        # Restore state
        sketch._total_count = data["total_count"]
        sketch._items_processed = data["items_processed"]

        # Restore counters
        sketch._counters = []
        sketch._item_map = {}

        for entry_data in data["counters"]:
            entry = CounterEntry.from_dict(entry_data)
            sketch._counters.append(entry)
            sketch._item_map[entry.item] = entry

        # Ensure the heap property is maintained
        heapq.heapify(sketch._counters)

        return sketch

    def estimate_size(self) -> int:
        """
        Estimate the current memory usage of this sketch in bytes.

        Returns:
            Estimated memory usage in bytes.
        """
        # Base size of the object
        size = sys.getsizeof(self)

        # Size of the counters list
        size += sys.getsizeof(self._counters)

        # Size of the item map
        size += sys.getsizeof(self._item_map)

        # Size of each counter entry
        for entry in self._counters:
            size += sys.getsizeof(entry)
            # Add size of item if it's a complex object
            if hasattr(entry.item, "__sizeof__"):
                size += entry.item.__sizeof__()
            else:
                size += sys.getsizeof(entry.item)

        return size

    def clear(self) -> None:
        """
        Reset the sketch to its initial empty state.

        This method clears all counters but keeps the same capacity.
        """
        self._counters = []
        self._item_map = {}
        self._total_count = 0
        self._items_processed = 0

    def __len__(self) -> int:
        """
        Get the number of unique items being tracked.

        Returns:
            Number of unique items being tracked.
        """
        return len(self._counters)

    @classmethod
    def create_from_error_rate(
        cls,
        threshold: float,
        error_rate: float,
        memory_limit_bytes: Optional[int] = None,
    ) -> "SpaceSaving[T]":
        """
        Create a Space-Saving sketch configured for a given error rate.

        Args:
            threshold: The frequency threshold for heavy hitters (between 0 and 1).
            error_rate: The maximum relative error in frequency estimation.
            memory_limit_bytes: Optional maximum memory usage in bytes.

        Returns:
            A new SpaceSaving object configured for the given error rate.

        Raises:
            ValueError: If threshold or error_rate are not between 0 and 1.
        """
        if not 0 < threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        if not 0 < error_rate < 1:
            raise ValueError("Error rate must be between 0 and 1")

        # Calculate the required capacity
        # For a threshold φ and error rate ε, we need capacity ≥ 1/(φ·ε)
        capacity = math.ceil(1.0 / (threshold * error_rate))

        return cls(capacity=capacity, memory_limit_bytes=memory_limit_bytes)
