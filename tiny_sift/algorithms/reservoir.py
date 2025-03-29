"""
Reservoir Sampling implementation for TinySift.

This module provides implementations of Reservoir Sampling and its variants
for maintaining a random sample of a data stream with bounded memory.
"""

import random
import sys
from typing import Any, Dict, Generic, List, Optional, TypeVar

from tiny_sift.core.base import SampleMaintainer

T = TypeVar('T')  # Type for the items being processed


class ReservoirSampling(SampleMaintainer[T]):
    """
    Standard Reservoir Sampling algorithm (Algorithm R).
    
    Maintains a uniform random sample of fixed size from a stream of unknown length.
    Uses O(k) memory where k is the reservoir size.
    
    References:
        - Vitter, J. S. (1985). Random sampling with a reservoir.
          ACM Transactions on Mathematical Software, 11(1), 37-57.
    """
    
    def __init__(self, size: int, memory_limit_bytes: Optional[int] = None, seed: Optional[int] = None):
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
        super().update(item)  # Increment items_processed
        
        if len(self._reservoir) < self._size:
            # Reservoir not full yet, add item
            self._reservoir.append(item)
        else:
            # Randomly decide whether to replace an item
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
    
    def merge(self, other: 'ReservoirSampling[T]') -> 'ReservoirSampling[T]':
        """
        Merge this reservoir with another reservoir.
        
        This merges two reservoirs by creating a new one and randomly selecting
        items from both reservoirs based on their relative sizes.
        
        Args:
            other: Another ReservoirSampling object.
            
        Returns:
            A new merged ReservoirSampling object.
            
        Raises:
            TypeError: If other is not a ReservoirSampling object.
        """
        self._check_same_type(other)
        
        # Create a new reservoir with the same size
        result = ReservoirSampling[T](self._size, self._memory_limit_bytes)
        
        # Combine the two reservoirs
        combined = self._reservoir + other._reservoir
        
        # If the combined size is less than or equal to the target size,
        # just use all of them
        if len(combined) <= self._size:
            result._reservoir = combined
        else:
            # Otherwise, randomly select items
            result._reservoir = self._random.sample(combined, self._size)
        
        # Update the items processed count
        result._items_processed = self._combine_items_processed(other)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the reservoir to a dictionary for serialization.
        
        Returns:
            A dictionary representation of the reservoir.
        """
        data = self._base_dict()
        data.update({
            "size": self._size,
            "reservoir": self._reservoir,
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReservoirSampling[T]':
        """
        Create a reservoir from a dictionary representation.
        
        Args:
            data: The dictionary containing the reservoir state.
            
        Returns:
            A new ReservoirSampling object.
        """
        # Create a new reservoir
        reservoir = cls(
            size=data["size"],
            memory_limit_bytes=data.get("memory_limit_bytes")
        )
        
        # Restore the state
        reservoir._reservoir = data["reservoir"]
        reservoir._items_processed = data["items_processed"]
        
        return reservoir
    
    def estimate_size(self) -> int:
        """
        Estimate the current memory usage of this reservoir in bytes.
        
        Returns:
            Estimated memory usage in bytes.
        """
        # Base size of the object
        size = sys.getsizeof(self)
        
        # Size of the reservoir list itself
        size += sys.getsizeof(self._reservoir)
        
        # Approximate size of all items in the reservoir
        for item in self._reservoir:
            size += sys.getsizeof(item)
        
        return size


class WeightedReservoirSampling(ReservoirSampling[T]):
    """
    Weighted Reservoir Sampling using exponential jumps (Algorithm A-Exp-Res).
    
    Maintains a random sample with probability proportional to item weights.
    Uses an efficient key-based approach where the key for each item is 
    random^(1/weight), and keeps items with the highest keys.
    
    References:
        - Efraimidis, P. S., & Spirakis, P. G. (2006). Weighted random sampling with a reservoir.
          Information Processing Letters, 97(5), 181-185.
    """
    
    def __init__(self, size: int, memory_limit_bytes: Optional[int] = None, seed: Optional[int] = None):
        """
        Initialize a new weighted reservoir sampler.
        
        Args:
            size: The size of the reservoir (number of samples to maintain).
            memory_limit_bytes: Optional maximum memory usage in bytes.
            seed: Optional random seed for reproducibility.
        """
        super().__init__(size, memory_limit_bytes, seed)
        
        # Store items along with their weights and keys
        self._weighted_reservoir: List[tuple[T, float, float]] = []
        self._total_weight = 0.0
    
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
        
        # Increment the items processed counter
        self._items_processed += 1
        
        # Update total weight
        self._total_weight += weight
        
        # Generate a random key for the item
        # Key = random ^ (1/weight) for proper weighting
        u = self._random.random()
        key = u ** (1.0 / weight)
        
        # If reservoir not full yet, add item with its weight and key
        if len(self._weighted_reservoir) < self._size:
            self._weighted_reservoir.append((item, weight, key))
            # Sort by key (descending) when reservoir becomes full
            if len(self._weighted_reservoir) == self._size:
                self._weighted_reservoir.sort(key=lambda x: x[2], reverse=True)
        elif key < self._weighted_reservoir[-1][2]:
            # If key is smaller than the smallest in the reservoir, skip
            return
        else:
            # Replace the item with the smallest key
            # First, find insertion point (descending order)
            index = 0
            while index < len(self._weighted_reservoir) and self._weighted_reservoir[index][2] > key:
                index += 1
            
            # Insert the new item and remove the smallest key
            self._weighted_reservoir.insert(index, (item, weight, key))
            self._weighted_reservoir.pop(-1)
    
    def get_sample(self) -> List[T]:
        """
        Get the current weighted reservoir sample.
        
        Returns:
            A list containing the current sample (without weights).
        """
        return [item for item, _, _ in self._weighted_reservoir]
    
    def get_weighted_sample(self) -> List[tuple[T, float]]:
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
        data.update({
            "size": self._size,
            "weighted_reservoir": self._weighted_reservoir,
            "total_weight": self._total_weight,
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WeightedReservoirSampling[T]':
        """
        Create a weighted reservoir from a dictionary representation.
        
        Args:
            data: The dictionary containing the weighted reservoir state.
            
        Returns:
            A new WeightedReservoirSampling object.
        """
        # Create a new weighted reservoir
        reservoir = cls(
            size=data["size"],
            memory_limit_bytes=data.get("memory_limit_bytes")
        )
        
        # Restore the state
        reservoir._weighted_reservoir = data["weighted_reservoir"]
        reservoir._total_weight = data["total_weight"]
        reservoir._items_processed = data["items_processed"]
        
        return reservoir
    
    def estimate_size(self) -> int:
        """
        Estimate the current memory usage of this weighted reservoir in bytes.
        
        Returns:
            Estimated memory usage in bytes.
        """
        # Base size of the object
        size = sys.getsizeof(self)
        
        # Size of the weighted reservoir list itself
        size += sys.getsizeof(self._weighted_reservoir)
        
        # Approximate size of all items in the weighted reservoir
        for item, weight, key in self._weighted_reservoir:
            size += sys.getsizeof(item) + sys.getsizeof(weight) + sys.getsizeof(key)
        
        return size