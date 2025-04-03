"""
Bloom Filter implementation for TinySift with enhanced benchmarking hooks.

This module provides an implementation of the Bloom Filter, a space-efficient
probabilistic data structure used for testing set membership with tunable
false positive rates and no false negatives.

The implementation includes comprehensive benchmarking hooks to help users
understand the performance characteristics, memory usage, and accuracy
of their Bloom Filter configuration.

References:
    - Bloom, B. H. (1970). Space/time trade-offs in hash coding with allowable errors.
      Communications of the ACM, 13(7), 422-426.
"""

import array
import math
import sys
from collections import Counter
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union, cast

from tiny_sift.core.base import StreamSummary
from tiny_sift.core.hash import fnv1a_32, murmurhash3_32

T = TypeVar("T")  # Type for the items being processed


class BloomFilter(StreamSummary[T, bool]):
    """
    Bloom Filter for efficient set membership testing with benchmarking hooks.

    A Bloom filter is a space-efficient probabilistic data structure used to test
    whether an element is a member of a set. False positives are possible, but
    false negatives are not – in other words, a query returns either "possibly in set"
    or "definitely not in set".

    This implementation includes comprehensive benchmarking hooks to help users
    understand the performance characteristics, memory usage, and accuracy
    of their Bloom Filter configuration.

    Example:
        # Create a filter with 1% false positive rate for 1000 items
        bloom = BloomFilter(expected_items=1000, false_positive_rate=0.01)

        # Add some items
        bloom.update("apple")
        bloom.update("banana")

        # Check for membership
        contains_apple = bloom.contains("apple")  # Returns True
        contains_orange = bloom.contains("orange")  # Returns False

        # Get performance statistics
        stats = bloom.get_stats()
        performance = bloom.analyze_performance()

    References:
        - Bloom, B. H. (1970). Space/time trade-offs in hash coding with allowable errors.
          Communications of the ACM, 13(7), 422-426.
    """

    def __init__(
        self,
        expected_items: int = 10000,
        false_positive_rate: float = 0.01,
        memory_limit_bytes: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize a new Bloom filter.

        Args:
            expected_items: Expected number of unique items to be added to the filter.
            false_positive_rate: Target false positive rate (between 0 and 1).
            memory_limit_bytes: Optional maximum memory usage in bytes.
                                Note: This is primarily used by create_from_memory_limit.
                                If provided here, it doesn't override calculations
                                based on expected_items and false_positive_rate.
            seed: Optional random seed for hash functions.

        Raises:
            ValueError: If expected_items is less than 1.
                       If false_positive_rate is not between 0 and 1.
        """
        super().__init__(memory_limit_bytes)

        if expected_items < 1:
            raise ValueError("Expected number of items must be at least 1")
        if not (0 < false_positive_rate < 1):
            raise ValueError("False positive rate must be between 0 and 1")

        self._expected_items = expected_items
        self._false_positive_rate = false_positive_rate
        self._seed = seed if seed is not None else 0

        # Calculate optimal bit array size (m) and number of hash functions (k)
        # These calculations are based on the desired properties, not the memory limit directly.
        bit_size = self._calculate_bit_size(expected_items, false_positive_rate)
        hash_count = self._calculate_hash_count(bit_size, expected_items)

        self._bit_size = bit_size
        self._hash_count = hash_count

        # Initialize the bit array using array.array for memory efficiency
        # 'B' typecode gives unsigned char (8 bits, 0 to 255)
        # We need to convert bit_size to bytes (rounded up)
        num_bytes = (bit_size + 7) // 8
        self._bytes = array.array("B", [0] * num_bytes)

        # Track the number of unique items (approximate)
        self._approximate_count = 0

        # Storage for memory breakdown (used by estimate_size)
        self._memory_breakdown = None

    def _calculate_bit_size(self, n: int, p: float) -> int:
        """
        Calculate the optimal bit array size for the given parameters.

        Args:
            n: Expected number of items.
            p: Target false positive rate.

        Returns:
            Optimal bit array size.
        """
        # Optimal bit size formula: m = -(n * ln(p)) / (ln(2)^2)
        # Ensure p is slightly > 0 for log
        p_safe = max(p, 1e-12)
        m = -(n * math.log(p_safe)) / (math.log(2) ** 2)
        return max(8, math.ceil(m))  # Ensure at least one byte

    def _calculate_hash_count(self, m: int, n: int) -> int:
        """
        Calculate the optimal number of hash functions.

        Args:
            m: Bit array size.
            n: Expected number of items.

        Returns:
            Optimal number of hash functions.
        """
        # Optimal hash count formula: k = (m/n) * ln(2)
        # Ensure n > 0
        n_safe = max(n, 1)
        k = (m / n_safe) * math.log(2)
        return max(1, math.ceil(k))  # Ensure at least one hash function

    def _get_bit_positions(self, item: T) -> List[int]:
        """
        Generate the bit positions for an item using multiple hash functions.

        This method uses a technique to generate multiple hash values from two
        independent hash functions, which is more efficient than computing
        k independent hash values.

        Args:
            item: The item to hash.

        Returns:
            List of bit positions to set or check.
        """
        positions = []

        # Use two different hash functions as the basis
        # Ensure consistent hashing for different types
        item_bytes = str(item).encode("utf-8")
        h1 = murmurhash3_32(item_bytes, seed=self._seed)
        h2 = fnv1a_32(item_bytes, seed=self._seed)

        # Generate k hash values using the formula: (h1 + i*h2) % bit_size
        # This is a common technique to avoid computing k independent hash functions
        for i in range(self._hash_count):
            position = (h1 + i * h2) % self._bit_size
            positions.append(position)

        return positions

    def _set_bit(self, position: int) -> None:
        """
        Set a bit at the given position.

        Args:
            position: Bit position to set.
        """
        byte_index = position // 8
        bit_index = position % 8
        # Check bounds to prevent IndexError if position is somehow invalid
        if 0 <= byte_index < len(self._bytes):
            self._bytes[byte_index] |= 1 << bit_index

    def _test_bit(self, position: int) -> bool:
        """
        Test if a bit is set at the given position.

        Args:
            position: Bit position to test.

        Returns:
            True if the bit is set, False otherwise.
        """
        byte_index = position // 8
        bit_index = position % 8
        # Check bounds
        if 0 <= byte_index < len(self._bytes):
            return bool(self._bytes[byte_index] & (1 << bit_index))
        return False  # Position out of bounds

    def update(self, item: T) -> None:
        """
        Add an item to the Bloom filter.

        This method sets the bits at positions determined by the hash functions
        for the given item.

        Args:
            item: The item to add to the filter.
        """
        # Call parent class's update method to increment items_processed
        super().update(item)

        # Keep track if this is potentially a new item
        all_bits_set = True

        # Get bit positions for this item
        positions = self._get_bit_positions(item)

        # Check if all bits are already set *before* setting them
        for position in positions:
            if not self._test_bit(position):
                all_bits_set = False
                # Set the bit immediately once we know it's not set
                self._set_bit(position)
            # If the bit was already set, we don't need to test others
            # to know if it's a new item, but we still need to ensure
            # all bits are eventually set for this item.

        # If we didn't find an unset bit above, set any remaining bits
        # (This loop is redundant if the first loop always sets the bit)
        # Keep the original structure for clarity: first check, then set all.
        if not all_bits_set:
            # If it wasn't all set, re-iterate to ensure all are set now
            # (some might have been set in the first loop)
            for position in positions:
                self._set_bit(position)  # Safe to call again, idempotent

        # If not all bits were already set, this might be a new item
        if not all_bits_set:
            self._approximate_count += 1

    def contains(self, item: T) -> bool:
        """
        Test if an item might be in the set.

        Args:
            item: The item to test.

        Returns:
            True if the item might be in the set, False if definitely not in the set.
        """
        # Get bit positions for this item
        positions = self._get_bit_positions(item)

        # Check if all bits are set
        for position in positions:
            if not self._test_bit(position):
                return False

        # All bits are set, item might be in the set
        return True

    def query(self, item: T, *args: Any, **kwargs: Any) -> bool:
        """
        Query the Bloom filter for an item.

        This is a convenience method that calls contains().

        Args:
            item: The item to test.

        Returns:
            True if the item might be in the set, False if definitely not in the set.
        """
        return self.contains(item)

    def merge(self, other: "BloomFilter[T]") -> "BloomFilter[T]":
        """
        Merge this Bloom filter with another one.

        This operation allows combining two Bloom filters that have the same
        parameters (bit size and hash count). The resulting filter will contain
        all items from both filters.

        Args:
            other: Another BloomFilter with the same parameters.

        Returns:
            A new merged BloomFilter.

        Raises:
            TypeError: If other is not a BloomFilter.
            ValueError: If filters have incompatible parameters.
        """
        # Check that other is a BloomFilter
        self._check_same_type(other)

        # Check that parameters match
        if self._bit_size != other._bit_size or self._hash_count != other._hash_count:
            raise ValueError(
                f"Cannot merge Bloom filters with different parameters: "
                f"({self._bit_size}, {self._hash_count}) and "
                f"({other._bit_size}, {other._hash_count})"
            )

        # Create a new filter with the same parameters
        result = BloomFilter[T](
            expected_items=self._expected_items,  # Or maybe max(self._expected, other._expected)?
            false_positive_rate=self._false_positive_rate,  # Must be the same
            memory_limit_bytes=self._memory_limit_bytes,  # Or combine?
            seed=self._seed,  # Must be the same if hashing relies on it implicitly
        )
        # Ensure the created filter *actually* has the same runtime parameters
        if result._bit_size != self._bit_size or result._hash_count != self._hash_count:
            raise RuntimeError("Internal inconsistency during merge initialization.")

        # Combine bit arrays using bitwise OR
        for i in range(len(self._bytes)):
            result._bytes[i] = self._bytes[i] | other._bytes[i]

        # Approximate the count (this is just an estimate)
        # Simple sum capped by expected_items might be sufficient
        result._approximate_count = min(
            self._approximate_count + other._approximate_count,
            result._expected_items,  # Cap at the capacity of the *new* filter
        )
        # A potentially better estimate after merging comes from estimate_cardinality()

        # Update the items processed count
        result._items_processed = self._combine_items_processed(other)

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Bloom filter to a dictionary for serialization.

        Returns:
            A dictionary representation of the filter.
        """
        data = self._base_dict()
        data.update(
            {
                "expected_items": self._expected_items,
                "false_positive_rate": self._false_positive_rate,
                "bit_size": self._bit_size,
                "hash_count": self._hash_count,
                "bytes": list(self._bytes),  # Convert array to list for JSON
                "approximate_count": self._approximate_count,
                "seed": self._seed,
            }
        )
        data["type"] = self.__class__.__name__  # Add type explicitly
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BloomFilter[T]":
        """
        Create a Bloom filter from a dictionary representation.

        Args:
            data: The dictionary containing the filter state.

        Returns:
            A new BloomFilter initialized with the given state.
        """
        # Create a new filter using stored parameters
        filter_instance = cls(
            expected_items=data["expected_items"],
            false_positive_rate=data["false_positive_rate"],
            memory_limit_bytes=data.get("memory_limit_bytes"),  # Restore if present
            seed=data.get("seed", 0),  # Restore seed
        )

        # Restore the exact state, overriding calculated values if necessary
        # Check if stored size/hash match calculated; warn or error if not?
        # For now, assume dict holds the definitive state.
        if (
            filter_instance._bit_size != data["bit_size"]
            or filter_instance._hash_count != data["hash_count"]
        ):
            # This might happen if formulas change or floats differ slightly.
            # Decide whether to trust dict or recalculate. Trusting dict is safer.
            # print("Warning: Loaded state differs from calculated parameters. Using loaded state.")
            pass  # Allow override

        filter_instance._bit_size = data["bit_size"]
        filter_instance._hash_count = data["hash_count"]

        # Ensure byte array size matches bit_size
        num_bytes = (filter_instance._bit_size + 7) // 8
        filter_instance._bytes = array.array(
            "B", data["bytes"][:num_bytes]
        )  # Use list from dict
        if len(filter_instance._bytes) != num_bytes:
            # Pad with zeros or truncate if necessary, or raise error
            raise ValueError(
                "Mismatch between bit_size and length of loaded byte array"
            )

        filter_instance._approximate_count = data["approximate_count"]
        filter_instance._items_processed = data["items_processed"]

        return filter_instance

    def estimate_size(self) -> int:
        """
        Estimate the current memory usage of this filter in bytes.

        Provides a detailed breakdown of memory used by different components.

        Returns:
            Estimated memory usage in bytes.
        """
        # Start with the base object size
        size = sys.getsizeof(self)

        # Add size of instance dictionary
        if hasattr(self, "__dict__"):
            size += sys.getsizeof(self.__dict__)

        # Add size of the byte array (includes object overhead + buffer)
        size += sys.getsizeof(self._bytes)

        # Calculate detailed memory breakdown
        memory_breakdown = {
            "base_object": sys.getsizeof(self),
            "byte_array": sys.getsizeof(self._bytes),
            "byte_array_buffer": len(self._bytes),  # Actual storage bytes
            "parameters": sum(
                sys.getsizeof(getattr(self, attr))
                for attr in [
                    "_bit_size",
                    "_hash_count",
                    "_expected_items",
                    "_false_positive_rate",
                    "_items_processed",
                ]
            ),
            "estimated_total": size,
        }

        # Store this breakdown for access in get_stats
        self._memory_breakdown = memory_breakdown

        return size

    def estimate_cardinality(self) -> int:
        """
        Estimate the number of unique items in the filter.

        This is an approximation based on the fill ratio of the bit array.
        The estimate becomes less accurate as the filter becomes more saturated.

        Returns:
            Estimated number of unique items.
        """
        # Count the number of bits set to 1
        set_bits = sum(bin(byte).count("1") for byte in self._bytes)

        # Use the formula: n ≈ -m * ln(1 - X/m) / k
        # where X is the number of bits set to 1, m is the bit size, and k is the hash count
        # This estimates the number of unique items based on the fill ratio

        # Avoid division by zero or taking log of zero/negative
        if self._hash_count == 0 or self._bit_size == 0:
            return 0  # Not a valid filter state
        if set_bits == 0:
            return 0
        if set_bits >= self._bit_size:
            # Filter is saturated or over-saturated. Estimate is unreliable.
            # Return expected_items or items_processed as upper bound?
            # Using the formula can yield infinity or NaN here.
            # A common approach is to return a large number or cap.
            # Let's cap at items_processed as a pragmatic upper bound.
            return self._items_processed

        fraction_bits_set = set_bits / self._bit_size
        # Ensure argument to log is > 0
        log_arg = 1.0 - fraction_bits_set
        if log_arg <= 0:
            # Should be covered by set_bits >= self._bit_size, but for safety:
            return self._items_processed

        try:
            estimate = -self._bit_size * math.log(log_arg) / self._hash_count
        except (ValueError, OverflowError):
            # Handle potential math errors if estimate becomes huge
            return self._items_processed

        # Return integer estimate, bounded by total items processed
        # (cannot have more unique items than total items processed)
        return min(max(0, int(round(estimate))), self._items_processed)

    def false_positive_probability(self) -> float:
        """
        Calculate the current false positive probability based on the fill ratio.

        The false positive probability increases as more items are added to the filter.
        This provides an *estimate* based on the current state, not the initial target rate.

        Returns:
            Current estimated false positive probability.
        """
        # Count the number of bits set to 1
        set_bits = sum(bin(byte).count("1") for byte in self._bytes)

        # Calculate the fraction of bits set to 1
        if self._bit_size == 0:
            return 1.0  # Invalid state, assume worst case

        fraction_bits_set = set_bits / self._bit_size

        # Estimate the false positive probability using the fill ratio:
        # FPP ≈ (fraction_bits_set)^k
        # This is a common approximation.
        try:
            fpp = fraction_bits_set**self._hash_count
        except OverflowError:
            # Handle potential overflow if fraction is large and k is large
            fpp = 1.0

        # Alternative formula using estimated cardinality 'n_est':
        # FPP ≈ (1 - exp(-k * n_est / m))^k
        # n_est = self.estimate_cardinality()
        # if n_est == 0: return 0.0
        # exponent = -self._hash_count * n_est / self._bit_size
        # try:
        #     fpp = (1.0 - math.exp(exponent)) ** self._hash_count
        # except OverflowError:
        #     fpp = 1.0 # If exp becomes too small or large

        return max(0.0, min(fpp, 1.0))  # Clamp between 0 and 1

    def is_empty(self) -> bool:
        """
        Check if the filter is empty (no items added or all bits are zero).

        Returns:
            True if the filter is empty, False otherwise.
        """
        # Check if all bytes in the array are zero
        return all(byte == 0 for byte in self._bytes)

    def clear(self) -> None:
        """
        Reset the filter to its initial empty state.

        This method clears all bits but keeps the same parameters (size, hash count).
        """
        # More efficient way to clear the array than iterating
        # Create a new zeroed array of the same size
        num_bytes = len(self._bytes)
        self._bytes = array.array("B", [0] * num_bytes)

        self._approximate_count = 0
        self._items_processed = 0

    @classmethod
    def create_from_memory_limit(
        cls,
        memory_bytes: int,
        false_positive_rate: float = 0.01,
        seed: Optional[int] = None,
    ) -> "BloomFilter[T]":
        """
        Create a Bloom filter optimized for a given memory limit.

        This factory method calculates the maximum number of items that can be
        stored within the specified memory limit while maintaining the desired
        false positive rate. The memory limit calculation accounts for the
        base object size and the overhead of the array.array itself.

        Args:
            memory_bytes: Maximum desired memory usage in bytes.
            false_positive_rate: Target false positive rate.
            seed: Optional random seed for hash functions.

        Returns:
            A new BloomFilter optimized for the memory constraint.

        Raises:
            ValueError: If memory_bytes is too small for a useful filter
                        (cannot hold minimal object structure + 1 byte array).
        """
        if memory_bytes <= 0:
            raise ValueError("Memory limit must be positive")
        if not (0 < false_positive_rate < 1):
            raise ValueError("False positive rate must be between 0 and 1")

        # --- Estimation of Overheads ---
        # 1. Base object size (estimate using a minimal instance)
        try:
            # Need to handle potential args required by __init__ in derived classes
            temp_instance_for_size = cls(expected_items=1, false_positive_rate=0.5)
            base_size = sys.getsizeof(temp_instance_for_size)
            if hasattr(temp_instance_for_size, "__dict__"):
                base_size += sys.getsizeof(temp_instance_for_size.__dict__)
            del temp_instance_for_size  # clean up
        except Exception:
            # Fallback if init is complex - less accurate
            base_size = sys.getsizeof(object()) + 200  # Rough estimate

        # 2. Overhead of the array.array object itself (excluding buffer)
        try:
            empty_array_overhead = sys.getsizeof(array.array("B"))
        except NameError:  # Should be imported, but safety check
            import array

            empty_array_overhead = sys.getsizeof(array.array("B"))

        # --- Calculate Available Space for Array Buffer ---
        total_overhead = base_size + empty_array_overhead
        available_bytes_for_buffer = memory_bytes - total_overhead

        if available_bytes_for_buffer < 1:
            raise ValueError(
                f"Memory limit {memory_bytes} bytes is too small. "
                f"Estimated overhead is {total_overhead} bytes. "
                f"Need at least {total_overhead + 1} bytes."
            )

        # --- Determine Filter Parameters based on Buffer Size ---
        # Calculate maximum bit size based *only* on buffer space
        max_bit_size = available_bytes_for_buffer * 8
        # Ensure bit size is at least 8 (1 byte array minimum useful size)
        max_bit_size = max(8, max_bit_size)

        # Calculate maximum number of items 'n' for this bit size 'm' and FPP 'p'
        # Using the inverse of the bit size formula: n ≈ -m * ln(p) / (ln(2)^2)
        ln2_squared = math.log(2) ** 2
        # Ensure p is slightly > 0 for log
        p_safe = max(false_positive_rate, 1e-12)
        log_p = math.log(p_safe)

        if log_p >= 0:  # Avoid division by zero or positive log(p)
            # Cannot achieve target FPP with this size, default to 1 item capacity
            max_items = 1
        else:
            max_items = int(-(max_bit_size * ln2_squared) / log_p)

        # Ensure at least one item capacity
        max_items = max(1, max_items)

        # --- Create the Actual Filter ---
        # Use the calculated max_items and the target FPP to initialize.
        # The internal __init__ will then recalculate bit_size and hash_count.
        # These recalculated values might require slightly *less* memory than
        # our buffer calculation allowed, which is fine. They should not require *more*.
        instance = cls(
            expected_items=max_items,
            false_positive_rate=false_positive_rate,
            memory_limit_bytes=memory_bytes,  # Store the original constraint if needed
            seed=seed,
        )

        # --- Final Sanity Check (Optional) ---
        # Verify the instance created fits within the original limit.
        # Due to calculation nuances (ceil, float precision), it might slightly exceed.
        final_estimated_size = instance.estimate_size()
        if final_estimated_size > memory_bytes:
            # This *can* happen if the overhead estimation wasn't perfect or if
            # the calculated bit_size maps to an array allocation slightly larger
            # than available_bytes_for_buffer.
            # For practical purposes, this small overflow might be acceptable,
            # or one could iteratively reduce max_items slightly and recreate
            # the instance until it fits, but that adds complexity.
            print(
                f"Warning: Created filter estimated size ({final_estimated_size} bytes) "
                f"slightly exceeds memory limit ({memory_bytes} bytes). "
                f"This can happen due to overhead estimation or allocation granularity."
            )
            # If strict adherence is critical, consider raising an error or implementing
            # the iterative reduction mentioned above.

        return instance

    def get_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the current state of the Bloom filter.

        This method extends the base class implementation to include Bloom filter-specific
        statistics such as bit distribution, fill ratio, and estimated false positive rate.

        Returns:
            A dictionary containing various statistics about the filter state.
        """
        # Start with base class statistics
        stats = super().get_stats()

        # Calculate total number of bits in the filter
        bit_size = self._bit_size

        # Count set bits (1s) in the filter
        set_bits = sum(bin(byte).count("1") for byte in self._bytes)

        # Calculate fill ratio
        fill_ratio = set_bits / bit_size if bit_size > 0 else 0.0

        # Add Bloom filter specific information
        stats.update(
            {
                "expected_items": self._expected_items,
                "false_positive_rate": self._false_positive_rate,
                "bit_size": bit_size,
                "hash_count": self._hash_count,
                "items_processed": self._items_processed,
                "set_bits": set_bits,
                "fill_ratio": fill_ratio,
                "estimated_unique_items": self.estimate_cardinality(),
                "current_fpp": self.false_positive_probability(),
            }
        )

        # Analyze bit distribution
        # Group bytes by their population count (number of 1 bits)
        byte_distribution = Counter([bin(byte).count("1") for byte in self._bytes])
        byte_stats = {
            "zero_bytes": byte_distribution.get(0, 0),
            "full_bytes": byte_distribution.get(8, 0),
            "distribution": {
                str(bits): count for bits, count in sorted(byte_distribution.items())
            },
        }
        stats["byte_stats"] = byte_stats

        # Calculate theoretical vs observed fill ratio
        # Theoretical: 1 - (1 - 1/m)^(k*n) or approximately 1 - e^(-k*n/m)
        # where k is hash count, n is number of items, m is bit size
        if bit_size > 0 and self._items_processed > 0:
            theoretical_fill = 1.0 - math.exp(
                -(self._hash_count * self._items_processed) / bit_size
            )
            stats["theoretical_fill_ratio"] = theoretical_fill
            stats["observed_vs_theoretical_ratio"] = (
                fill_ratio / theoretical_fill if theoretical_fill > 0 else 0
            )

        # Add error bounds information
        error_bounds = self.error_bounds()
        if error_bounds:
            stats.update(error_bounds)

        # Memory metrics per item
        if self._items_processed > 0:
            stats["bits_per_item"] = bit_size / self._items_processed
            stats["bytes_per_item"] = self.estimate_size() / self._items_processed

        return stats

    def error_bounds(self) -> Dict[str, float]:
        """
        Calculate the theoretical error bounds for this Bloom filter.

        This method provides detailed error bound information specific to Bloom filters,
        including the theoretical false positive probability and its relationship to fill ratio.

        Returns:
            A dictionary with error bound information specific to the filter.
        """
        # Start with any values from the base class
        bounds = super().error_bounds()

        # Calculate theoretical false positive probability based on current state
        # Formula: (1 - e^(-k*n/m))^k ≈ (1-(1-1/m)^(k*n))^k
        # where k is hash count, n is number of items, m is bit size
        bit_size = self._bit_size
        hash_count = self._hash_count
        items = self._items_processed

        if bit_size > 0:
            # Theoretical best FPP for the current parameters
            optimal_fpp = (
                (0.6185) ** (bit_size / self._expected_items)
                if self._expected_items > 0
                else 0
            )

            # Current estimated FPP based on items_processed
            if items > 0:
                # Calculate using the precise formula
                current_fpp = (
                    1 - math.exp(-(hash_count * items) / bit_size)
                ) ** hash_count

                # For nearly full filters, ensure value is capped
                current_fpp = min(current_fpp, 1.0)

                bounds["theoretical_optimal_fpp"] = optimal_fpp
                bounds["current_theoretical_fpp"] = current_fpp
                bounds["fpp_ratio"] = (
                    current_fpp / optimal_fpp if optimal_fpp > 0 else float("inf")
                )

                # Add error margin based on fill ratio
                # As the filter gets fuller, error tends to increase
                fill_ratio = 1 - math.exp(-(hash_count * items) / bit_size)
                if fill_ratio < 0.5:
                    bounds["error_margin"] = "low"
                elif fill_ratio < 0.8:
                    bounds["error_margin"] = "moderate"
                else:
                    bounds["error_margin"] = "high"

                bounds["fill_ratio"] = fill_ratio

        return bounds

    def analyze_hash_quality(self) -> Dict[str, Any]:
        """
        Analyze the quality of hash functions based on bit distribution.

        This method examines the distribution of set bits in the filter
        to assess how well the hash functions are distributing items.

        Returns:
            A dictionary with hash quality metrics.
        """
        if not self._bytes or self._bit_size == 0:
            return {"status": "empty_filter"}

        # Calculate bit distribution
        bit_counts = []
        for byte in self._bytes:
            for bit_pos in range(8):
                if (byte >> bit_pos) & 1:
                    bit_counts.append(1)
                else:
                    bit_counts.append(0)

        # Trim to actual bit size (last byte might have unused bits)
        bit_counts = bit_counts[: self._bit_size]

        # Calculate statistical measures
        set_bits = sum(bit_counts)
        total_bits = len(bit_counts)

        # Expected ratio if hash functions are uniform
        expected_ratio = 1.0 - math.exp(
            -(self._hash_count * self._items_processed) / self._bit_size
        )

        # Analyze bit distribution uniformity
        # Divide bit array into regions and compare fill ratios
        num_regions = min(10, self._bit_size // 100)
        if num_regions >= 2:
            region_size = self._bit_size // num_regions
            region_fill_ratios = []

            for i in range(num_regions):
                start_idx = i * region_size
                end_idx = min(start_idx + region_size, self._bit_size)
                region_bits = bit_counts[start_idx:end_idx]
                region_fill = sum(region_bits) / len(region_bits) if region_bits else 0
                region_fill_ratios.append(region_fill)

            # Calculate coefficient of variation as a measure of uniformity
            # Lower CV indicates more uniform distribution
            avg_fill = sum(region_fill_ratios) / len(region_fill_ratios)
            variance = sum((r - avg_fill) ** 2 for r in region_fill_ratios) / len(
                region_fill_ratios
            )
            std_dev = math.sqrt(variance)
            cv = std_dev / avg_fill if avg_fill > 0 else 0

            # Determine quality based on CV
            # Ideal: CV < 0.05 (very uniform)
            # Good: CV < 0.1 (acceptable uniformity)
            # Fair: CV < 0.2 (some non-uniformity)
            # Poor: CV >= 0.2 (significant non-uniformity)
            if cv < 0.05:
                quality = "ideal"
            elif cv < 0.1:
                quality = "good"
            elif cv < 0.2:
                quality = "fair"
            else:
                quality = "poor"

            hash_quality = {
                "bit_sample_size": self._bit_size,
                "filled_ratio": set_bits / total_bits,
                "expected_ratio": expected_ratio,
                "ratio_difference": abs(set_bits / total_bits - expected_ratio),
                "region_std_dev": std_dev,
                "region_cv": cv,
                "uniformity_quality": quality,
                "region_fill_ratios": region_fill_ratios,
            }
        else:
            # Not enough bits for regional analysis
            hash_quality = {
                "bit_sample_size": self._bit_size,
                "filled_ratio": set_bits / total_bits,
                "expected_ratio": expected_ratio,
                "ratio_difference": abs(set_bits / total_bits - expected_ratio),
                "note": "Filter too small for detailed uniformity analysis",
            }

        return hash_quality

    def get_optimal_parameters(
        self, error_rate: Optional[float] = None, items: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculate optimal filter parameters based on desired error rate and item count.

        This method can help users determine the ideal configuration for their needs.

        Args:
            error_rate: Desired false positive rate (default: use current filter's rate)
            items: Expected number of items (default: use current filter's value)

        Returns:
            A dictionary with recommended parameters.
        """
        error_rate = error_rate if error_rate is not None else self._false_positive_rate
        items = items if items is not None else self._expected_items

        if not (0 < error_rate < 1) or items < 1:
            return {
                "error": "Invalid parameters. Ensure 0 < error_rate < 1 and items >= 1"
            }

        # Optimal bit size: m = -n*ln(p) / (ln(2)^2)
        optimal_bit_size = int(
            math.ceil(-items * math.log(error_rate) / (math.log(2) ** 2))
        )

        # Optimal hash count: k = (m/n) * ln(2)
        optimal_hash_count = int(math.ceil((optimal_bit_size / items) * math.log(2)))

        # Calculate memory requirements
        bytes_required = (optimal_bit_size + 7) // 8

        # Compare with current parameters
        current_params = {
            "bit_size": self._bit_size,
            "hash_count": self._hash_count,
            "bytes": len(self._bytes),
        }

        recommendations = {
            "target_error_rate": error_rate,
            "target_items": items,
            "optimal_bit_size": optimal_bit_size,
            "optimal_hash_count": optimal_hash_count,
            "optimal_bytes": bytes_required,
            "current_params": current_params,
            "size_difference": bytes_required - len(self._bytes),
            "expected_fpp_at_capacity": (
                1 - math.exp(-optimal_hash_count * items / optimal_bit_size)
            )
            ** optimal_hash_count,
        }

        return recommendations

    def analyze_performance(self) -> Dict[str, Any]:
        """
        Perform comprehensive performance analysis of the Bloom filter.

        This method combines various metrics to provide an overall assessment
        of the filter's performance, efficiency, and accuracy.

        Returns:
            A dictionary containing performance metrics and recommendations.
        """
        # Get basic stats
        stats = self.get_stats()
        error_info = self.error_bounds()
        hash_quality = self.analyze_hash_quality()

        analysis = {
            "algorithm": "Bloom Filter",
            "memory_efficiency": {
                "bits_per_item": stats.get("bits_per_item", 0),
                "total_bytes": self.estimate_size(),
                "bit_array_bytes": len(self._bytes),
                "optimal_bits_per_item": (
                    -math.log(self._false_positive_rate) / (math.log(2) ** 2)
                    if self._false_positive_rate > 0
                    else 0
                ),
            },
            "accuracy": {
                "target_fpp": self._false_positive_rate,
                "current_fpp": stats.get("current_fpp", 0),
                "bit_size": self._bit_size,
                "hash_count": self._hash_count,
            },
            "saturation": {
                "fill_ratio": stats.get("fill_ratio", 0),
                "items_ratio": (
                    self._items_processed / self._expected_items
                    if self._expected_items > 0
                    else 0
                ),
            },
            "hash_quality": {
                "uniformity": hash_quality.get("uniformity_quality", "unknown")
            },
        }

        # Add recommendations based on analysis
        recommendations = []

        # Check if filter is approaching saturation
        if analysis["saturation"]["fill_ratio"] > 0.8:
            recommendations.append(
                "Filter is nearing saturation (>80% full). False positive rate will be higher than configured."
            )

        # Check if items processed significantly exceeds expected items
        if analysis["saturation"]["items_ratio"] > 1.2:
            recommendations.append(
                f"Items processed ({self._items_processed}) exceeds expected items ({self._expected_items}) by >20%. "
                f"Consider creating a larger filter for better accuracy."
            )

        # Check if hash uniformity is poor
        if hash_quality.get("uniformity_quality") == "poor":
            recommendations.append(
                "Hash function uniformity is poor. Consider using different hash functions or seeds."
            )

        # Check memory efficiency
        optimal_bits = analysis["memory_efficiency"]["optimal_bits_per_item"]
        actual_bits = analysis["memory_efficiency"]["bits_per_item"]

        if actual_bits > 0 and optimal_bits > 0 and (actual_bits / optimal_bits > 1.5):
            recommendations.append(
                f"Filter uses {actual_bits:.1f} bits per item, but theoretical optimum is {optimal_bits:.1f} bits per item. "
                f"Consider reducing bit size or increasing expected items for better memory efficiency."
            )

        analysis["recommendations"] = recommendations

        return analysis
