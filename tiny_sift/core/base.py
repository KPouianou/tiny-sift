"""
Base classes and interfaces for TinySift streaming algorithms.

This module defines the abstract base classes that all streaming algorithms
must implement to provide a consistent interface across the library.
It includes benchmarking hooks for measuring and comparing performance
characteristics.
"""

import abc
import json
import sys
import time
from collections import deque
from typing import Any, Dict, Generic, List, Optional, Protocol, Tuple, TypeVar, Union

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
    merging with other summaries, and serialization. It also provides benchmarking
    hooks for measuring performance characteristics.
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

        # Performance tracking attributes
        self._last_update_time: float = 0.0
        self._total_update_time: float = 0.0
        self._update_count: int = 0

        # Optional performance tracking buffer for recent updates
        self._track_recent_updates: bool = False
        self._recent_update_times: Optional[deque] = None
        self._max_update_history: int = 100

    @abc.abstractmethod
    def update(self, item: T) -> None:
        """
        Update the summary with a new item from the stream.

        Args:
            item: The new item to process.
        """
        self._items_processed += 1

        # Performance tracking - only measure time if explicitly requested
        if self._track_recent_updates:
            # Initialize deque for tracking if not already done
            if self._recent_update_times is None:
                self._recent_update_times = deque(maxlen=self._max_update_history)

            # Record update time
            start_time = time.time()
            # The actual update logic is implemented in derived classes
            # after calling super().update(item)
            self._last_update_time = time.time() - start_time
            self._total_update_time += self._last_update_time
            self._update_count += 1

            # Store recent update time
            if self._recent_update_times is not None:
                self._recent_update_times.append(self._last_update_time)

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

        This method provides a rough estimation of the memory footprint of the
        data structure. It accounts for the base object size and key data
        structures, but may not capture all memory usage due to Python's
        memory management.

        Derived classes should override this method to provide more accurate
        estimates specific to their internal data structures.

        Returns:
            Estimated memory usage in bytes.
        """
        # Start with the base size of the object
        size = sys.getsizeof(self)

        # Add size of instance dictionary
        if hasattr(self, "__dict__"):
            size += sys.getsizeof(self.__dict__)

        # Add performance tracking structures if present
        if self._recent_update_times is not None:
            size += sys.getsizeof(self._recent_update_times)
            # Add approximate size of contents (float values)
            size += len(self._recent_update_times) * sys.getsizeof(0.0)

        return size

    def check_memory_limit(self) -> bool:
        """
        Check if the current memory usage exceeds the limit.

        Returns:
            True if the memory usage is within limits, False otherwise.
        """
        if self._memory_limit_bytes is None:
            return True

        return self.estimate_size() <= self._memory_limit_bytes

    def clear(self) -> None:
        """
        Reset the summary to its initial empty state.

        This method resets the base counters and tracking metrics. Derived
        classes must override this method to properly clear their specific
        data structures while calling super().clear() to ensure base metrics
        are reset correctly.
        """
        self._items_processed = 0
        self._total_update_time = 0.0
        self._update_count = 0
        self._last_update_time = 0.0

        if self._recent_update_times is not None:
            self._recent_update_times.clear()

    def enable_performance_tracking(
        self, track_recent_updates: bool = True, max_history: int = 100
    ) -> None:
        """
        Enable detailed performance tracking for benchmarking.

        Performance tracking adds some overhead, so it should only be
        enabled when benchmarking or debugging performance issues.

        Args:
            track_recent_updates: Whether to track timing of recent updates.
            max_history: Maximum number of recent updates to track.
        """
        self._track_recent_updates = track_recent_updates
        self._max_update_history = max(1, max_history)

        if track_recent_updates and self._recent_update_times is None:
            self._recent_update_times = deque(maxlen=self._max_update_history)

    def disable_performance_tracking(self) -> None:
        """Disable performance tracking to reduce overhead."""
        self._track_recent_updates = False
        self._recent_update_times = None

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for this summary.

        Returns:
            A dictionary containing performance metrics such as:
            - Total items processed
            - Average update time
            - Recent update times (if tracking is enabled)
            - Memory usage
        """
        stats = {
            "items_processed": self._items_processed,
            "memory_bytes": self.estimate_size(),
        }

        # Add timing statistics if available
        if self._update_count > 0:
            stats["avg_update_time_ns"] = (
                self._total_update_time / self._update_count
            ) * 1e9
            stats["last_update_time_ns"] = self._last_update_time * 1e9

        # Add recent update times if tracking is enabled
        if self._recent_update_times and len(self._recent_update_times) > 0:
            # Convert to nanoseconds for better readability of small values
            recent_times_ns = [t * 1e9 for t in self._recent_update_times]
            stats["recent_update_times_ns"] = recent_times_ns
            stats["min_update_time_ns"] = min(recent_times_ns)
            stats["max_update_time_ns"] = max(recent_times_ns)

        return stats

    def get_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the current state of the summary.

        This method returns algorithm-specific statistics and can be used
        for monitoring, debugging, or benchmarking. It combines performance
        statistics with algorithm-specific information.

        Derived classes should override this method to include their specific
        statistics while calling super().get_stats() to include base metrics.

        Returns:
            A dictionary containing various statistics about the summary state.
        """
        # Start with basic attributes and performance stats
        stats = {
            "type": self.__class__.__name__,
            "items_processed": self._items_processed,
            "memory_bytes": self.estimate_size(),
        }

        # Add memory limit if specified
        if self._memory_limit_bytes is not None:
            stats["memory_limit_bytes"] = self._memory_limit_bytes
            stats["memory_usage_pct"] = (
                self.estimate_size() / self._memory_limit_bytes
            ) * 100

        # Add performance tracking stats if available
        if self._update_count > 0:
            stats["avg_update_time_ns"] = (
                self._total_update_time / self._update_count
            ) * 1e9

        # Add error bounds information
        error_bounds = self.error_bounds()
        if error_bounds:
            stats.update(error_bounds)

        return stats

    def error_bounds(self) -> Dict[str, float]:
        """
        Get the theoretical error bounds for this summary.

        This method returns algorithm-specific error bounds and guarantees,
        which can help users understand the accuracy-memory tradeoff for
        their chosen algorithm and parameters.

        Derived classes should override this method to provide their specific
        error characteristics. The base implementation returns an empty dictionary.

        Returns:
            A dictionary containing error bound information specific to the algorithm.
        """
        return {}

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

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current state of the sample.

        Returns:
            A dictionary with sample-specific statistics.
        """
        stats = super().get_stats()

        # Add sample-specific information
        sample = self.get_sample()
        stats["sample_size"] = len(sample)

        return stats


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

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current state of the frequency estimator.

        Returns:
            A dictionary with frequency estimation specific statistics.
        """
        stats = super().get_stats()

        # Try to add the number of heavy hitters at a few thresholds
        try:
            stats["heavy_hitters_1pct"] = len(self.get_heavy_hitters(0.01))
            stats["heavy_hitters_5pct"] = len(self.get_heavy_hitters(0.05))
            stats["heavy_hitters_10pct"] = len(self.get_heavy_hitters(0.10))
        except Exception:
            # Skip if the implementation can't efficiently compute this
            pass

        return stats


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

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current state of the cardinality estimator.

        Returns:
            A dictionary with cardinality estimation specific statistics.
        """
        stats = super().get_stats()

        # Add cardinality information
        stats["estimated_cardinality"] = self.estimate_cardinality()

        return stats


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

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current state of the window aggregator.

        Returns:
            A dictionary with window-specific statistics.
        """
        stats = super().get_stats()

        # Add window statistics
        try:
            window_stats = self.get_window_stats()
            stats.update({f"window_{k}": v for k, v in window_stats.items()})
        except Exception:
            # Skip if window stats can't be computed yet
            pass

        return stats
