"""
Base classes and interfaces for TinySift streaming algorithms.

This module defines the abstract base classes that all streaming algorithms
must implement to provide a consistent interface across the library.
"""

import abc
import json
import sys
from typing import Any, Dict, Generic, Optional, Protocol, TypeVar, Union

T = TypeVar("T")  # Type for the items being processed
R = TypeVar("R")  # Type for the result of queries


class Serializable(Protocol):
    """Protocol defining methods for serialization and deserialization."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the object to a dictionary representation."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Serializable":
        """Create an object from its dictionary representation."""
        ...


class StreamSummary(Generic[T, R], abc.ABC):
    """
    Abstract base class for all streaming data structures.

    This class defines the common interface that all streaming algorithms must
    implement, including methods for updating with new items, querying results,
    merging with other summaries, and serialization.
    """

    def __init__(self, memory_limit_bytes: Optional[int] = None):
        """
        Initialize a new stream summary.

        Args:
            memory_limit_bytes: Optional maximum memory usage in bytes.
                                None means no explicit limit.
        """
        self._memory_limit_bytes = memory_limit_bytes
        self._items_processed = 0

    @abc.abstractmethod
    def update(self, item: T) -> None:
        """
        Update the summary with a new item from the stream.

        Args:
            item: The new item to process.
        """
        self._items_processed += 1

    @abc.abstractmethod
    def query(self, *args: Any, **kwargs: Any) -> R:
        """
        Query the current state of the summary.

        The parameters and return value depend on the specific algorithm.

        Returns:
            The result of the query, which depends on the specific algorithm.
        """
        pass

    @abc.abstractmethod
    def merge(self, other: "StreamSummary[T, R]") -> "StreamSummary[T, R]":
        """
        Merge this summary with another of the same type.

        Args:
            other: Another stream summary of the same type.

        Returns:
            A new merged stream summary.

        Raises:
            TypeError: If other is not of the same type.
        """
        pass

    def _check_same_type(self, other: "StreamSummary[T, R]") -> None:
        """
        Helper method to check if another summary is of the same type.

        Args:
            other: Another stream summary to check.

        Raises:
            TypeError: If other is not of the same type.
        """
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot merge with {other.__class__.__name__}")

    def _combine_items_processed(self, other: "StreamSummary[T, R]") -> int:
        """
        Helper method to combine items processed counts during merging.

        Args:
            other: Another stream summary.

        Returns:
            The combined count of processed items.
        """
        return self._items_processed + other._items_processed

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the summary to a dictionary for serialization.

        Returns:
            A dictionary representation of the summary.
        """
        pass

    def _base_dict(self) -> Dict[str, Any]:
        """
        Create a dictionary with base attributes common to all summaries.

        Returns:
            A dictionary with base attributes.
        """
        return {
            "type": self.__class__.__name__,
            "items_processed": self._items_processed,
            "memory_limit_bytes": self._memory_limit_bytes,
        }

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamSummary[T, R]":
        """
        Create a summary from a dictionary representation.

        Args:
            data: The dictionary containing the summary state.

        Returns:
            A new stream summary initialized with the given state.
        """
        pass

    def serialize(self, format: str = "json") -> Union[str, bytes]:
        """
        Serialize the summary to a string or bytes.

        Args:
            format: The serialization format ('json' or 'binary').

        Returns:
            The serialized representation of the summary.

        Raises:
            ValueError: If the format is not supported.
        """
        if format == "json":
            return json.dumps(self.to_dict())
        elif format == "binary":
            # This is a placeholder for binary serialization
            # In a real implementation, we might use pickle, msgpack, or a custom binary format
            # For now, just use JSON
            return json.dumps(self.to_dict()).encode("utf-8")
        else:
            raise ValueError(f"Unsupported serialization format: {format}")

    @classmethod
    def deserialize(
        cls, data: Union[str, bytes], format: str = "json"
    ) -> "StreamSummary[T, R]":
        """
        Deserialize a summary from a string or bytes.

        Args:
            data: The serialized summary.
            format: The serialization format ('json' or 'binary').

        Returns:
            A new stream summary.

        Raises:
            ValueError: If the format is not supported.
        """
        if format == "json":
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            return cls.from_dict(json.loads(data))
        elif format == "binary":
            # This is a placeholder for binary deserialization
            if isinstance(data, str):
                data = data.encode("utf-8")
            return cls.from_dict(json.loads(data.decode("utf-8")))
        else:
            raise ValueError(f"Unsupported serialization format: {format}")

    def estimate_size(self) -> int:
        """
        Estimate the current memory usage of this summary in bytes.

        This is a rough estimation and may not be exact due to Python's
        memory management.

        Returns:
            Estimated memory usage in bytes.
        """
        # This is a very basic estimation
        # In actual implementations, each algorithm should override this
        # to provide more accurate estimates
        return sys.getsizeof(self)

    def check_memory_limit(self) -> bool:
        """
        Check if the current memory usage exceeds the limit.

        Returns:
            True if the memory usage is within limits, False otherwise.
        """
        if self._memory_limit_bytes is None:
            return True

        return self.estimate_size() <= self._memory_limit_bytes

    @property
    def items_processed(self) -> int:
        """Get the total number of items processed by this summary."""
        return self._items_processed


class SampleMaintainer(StreamSummary[T, list[T]], abc.ABC):
    """
    Abstract base class for algorithms that maintain a sample of the stream.

    Examples include reservoir sampling and its variants.
    """

    @abc.abstractmethod
    def get_sample(self) -> list[T]:
        """
        Get the current sample maintained by the algorithm.

        Returns:
            A list containing the current sample.
        """
        pass


class FrequencyEstimator(StreamSummary[T, float], abc.ABC):
    """
    Abstract base class for frequency estimation algorithms.

    Examples include Count-Min Sketch and variants.
    """

    @abc.abstractmethod
    def estimate_frequency(self, item: T) -> float:
        """
        Estimate the frequency of an item in the stream.

        Args:
            item: The item to estimate the frequency for.

        Returns:
            The estimated frequency of the item.
        """
        pass

    @abc.abstractmethod
    def get_heavy_hitters(self, threshold: float) -> Dict[T, float]:
        """
        Get items that appear more frequently than the threshold.

        Args:
            threshold: The minimum frequency ratio (0.0 to 1.0) to include.

        Returns:
            A dictionary mapping items to their estimated frequencies.
        """
        pass


class CardinalityEstimator(StreamSummary[T, int], abc.ABC):
    """
    Abstract base class for cardinality estimation algorithms.

    Examples include HyperLogLog and variants.
    """

    @abc.abstractmethod
    def estimate_cardinality(self) -> int:
        """
        Estimate the number of unique items in the stream.

        Returns:
            The estimated cardinality.
        """
        pass


class WindowAggregator(StreamSummary[T, Dict[str, float]], abc.ABC):
    """
    Abstract base class for sliding window aggregation algorithms.

    Examples include Exponential Histograms.
    """

    @abc.abstractmethod
    def get_window_stats(self) -> Dict[str, float]:
        """
        Get statistics for the current window.

        Returns:
            A dictionary with statistics like mean, min, max, etc.
        """
        pass
