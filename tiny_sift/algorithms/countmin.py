"""
Count-Min Sketch implementation for TinySift.

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
import math
import sys
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from tiny_sift.core.base import FrequencyEstimator
from tiny_sift.core.hash import murmurhash3_32

T = TypeVar("T")  # Type for the items being processed


class CountMinSketch(FrequencyEstimator[T]):
    """
    Count-Min Sketch for frequency estimation in data streams.

    This implementation uses a 2D array of counters and multiple hash functions
    to estimate item frequencies with a bounded memory footprint. It provides
    an estimate that is always greater than or equal to the true frequency.

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

            # Increment the counter by count
            self._counters[i][index] += count

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

        return sketch

    def estimate_size(self) -> int:
        """
        Estimate the current memory usage of this sketch in bytes.

        Returns:
            Estimated memory usage in bytes.
        """
        # Base size of the object
        size = sys.getsizeof(self)

        # Size of the counter arrays
        for row in self._counters:
            size += sys.getsizeof(row)

        return size

    def error_bounds(self) -> Dict[str, float]:
        """
        Calculate the theoretical error bounds for this sketch.

        Returns:
            A dictionary with the error bounds:
            - epsilon: The error factor (errors are less than epsilon * total_frequency)
            - delta: The probability of exceeding the error bound
            - error_bounds: The absolute error bound (epsilon * total_frequency)
        """
        return {
            "epsilon": self._epsilon,
            "delta": self._delta,
            "error_bounds": self._epsilon * self._total_frequency,
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
