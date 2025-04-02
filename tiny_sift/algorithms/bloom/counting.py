"""
Counting Bloom Filter implementation for TinySift.

This module provides an implementation of the Counting Bloom Filter, an extension
of the Bloom Filter that supports deletion operations by replacing single bits
with small counters.

References:
    - Fan, L., Cao, P., Almeida, J., & Broder, A. Z. (2000).
      Summary cache: a scalable wide-area web cache sharing protocol.
      IEEE/ACM Transactions on Networking, 8(3), 281-293.
"""

import array
import math
import sys
from typing import Any, Dict, Optional, TypeVar, Union

# Assuming bloom_base is in the same directory or accessible via package structure
from tiny_sift.algorithms.bloom.base import BloomFilter
from tiny_sift.core.base import StreamSummary

T = TypeVar("T")  # Type for the items being processed


class CountingBloomFilter(BloomFilter[T]):
    """
    Counting Bloom Filter for set membership testing with deletion support.

    A Counting Bloom Filter extends the standard Bloom Filter by replacing single bits
    with small counters (typically 2 to 8 bits), allowing for item deletion.
    This comes at the cost of increased memory usage compared to a standard Bloom Filter
    with the same capacity and false positive rate target.

    Each logical position in the filter corresponds to a counter. When an item
    is added, the counters at its hash positions are incremented. When an item is
    deleted, the counters are decremented. A check (`contains`) returns true only
    if all relevant counters are greater than zero.

    The CountingBloomFilter inherits the false positive characteristics of the standard
    BloomFilter, but adds the ability to delete items. It still guarantees no false negatives,
    provided counters do not overflow or underflow inappropriately.

    References:
        - Fan, L., Cao, P., Almeida, J., & Broder, A. Z. (2000).
          Summary cache: a scalable wide-area web cache sharing protocol.
          IEEE/ACM Transactions on Networking, 8(3), 281-293.
    """

    # Define supported counter bit sizes at the class level
    SUPPORTED_COUNTER_BITS = list(range(2, 9))  # 2 to 8 bits

    def __init__(
        self,
        expected_items: int = 10000,
        false_positive_rate: float = 0.01,
        counter_bits: int = 4,
        memory_limit_bytes: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize a new Counting Bloom filter.

        Args:
            expected_items: Expected number of unique items to be added to the filter.
                            This influences the underlying size calculation.
            false_positive_rate: Target false positive rate (between 0 and 1).
            counter_bits: Number of bits per counter (default 4). Must be between 2 and 8.
                          Affects memory usage and the maximum count per position.
            memory_limit_bytes: Optional maximum memory usage in bytes. If provided,
                                it might influence `expected_items` via the base class
                                or `create_from_memory_limit`.
            seed: Optional random seed for hash functions.

        Raises:
            ValueError: If expected_items is less than 1.
                       If false_positive_rate is not between 0 and 1.
                       If counter_bits is not supported (typically 2-8).
            TypeError: If counter_bits is not an integer.
        """
        # --- Validate counter_bits *before* using it in calculations ---
        if not isinstance(counter_bits, int):
            # Raise TypeError early if it's not even an int
            raise TypeError(
                f"Counter bits must be an integer, got {type(counter_bits)}"
            )
        if counter_bits not in CountingBloomFilter.SUPPORTED_COUNTER_BITS:
            # Use class attribute for check
            raise ValueError(
                f"Counter bits must be one of {CountingBloomFilter.SUPPORTED_COUNTER_BITS}"
            )

        # --- Initialize parent class ---
        # This calculates initial self._bit_size (number of counters) and self._hash_count
        super().__init__(
            expected_items=expected_items,
            false_positive_rate=false_positive_rate,
            memory_limit_bytes=memory_limit_bytes,
            seed=seed,
        )

        # --- Set CBF specific attributes ---
        self._counter_bits = counter_bits
        # Now it's safe to calculate _counter_max
        self._counter_max = (1 << counter_bits) - 1  # Maximum value a counter can hold

        # --- Override the byte array initialization ---
        # The `self._bit_size` from the parent now represents the number of *counters*.
        # Calculate the total number of *bits* needed for all counters.
        total_bits_needed = self._bit_size * self._counter_bits

        # Calculate the number of *bytes* needed in the array.
        num_bytes = (total_bits_needed + 7) // 8

        # Initialize the byte array. Parent's _bytes is overwritten.
        self._bytes = array.array("B", [0] * num_bytes)
        # Parent's estimate_size will now use this larger byte array.

    # --- Optimized Counter Access Methods ---

    def _get_counter(self, position: int) -> int:
        """
        Get the counter value at a specific logical position using bit manipulation.

        Args:
            position: Logical counter position (0 <= position < self._bit_size).

        Returns:
            The value of the counter at the given position.
        """
        # Bounds check (optional but good practice)
        if not (0 <= position < self._bit_size):
            # Or handle differently depending on expected usage
            raise IndexError(
                f"Counter position {position} out of range (0 to {self._bit_size - 1})"
            )

        bit_start = position * self._counter_bits
        byte_start = bit_start // 8
        bit_offset = bit_start % 8  # Offset of counter start within the starting byte

        # Calculate the end bit (exclusive) and the last byte index (inclusive)
        bit_end = bit_start + self._counter_bits
        byte_end = (bit_end - 1) // 8

        # Read the necessary bytes (usually 1 or 2, max 3 for 8 bits near boundary)
        combined_bytes = 0
        bytes_to_read = byte_end - byte_start + 1
        array_len = len(self._bytes)

        for i in range(bytes_to_read):
            byte_index = byte_start + i
            # Read byte and shift it into the correct place in combined_bytes
            # Bytes are read in order (index increases), so shift later bytes left
            # Ensure index is within bounds before reading
            if byte_index < array_len:
                combined_bytes |= self._bytes[byte_index] << (i * 8)
            # else: implicitly treat out-of-bounds bytes as 0 if byte_index >= array_len

        # Create a mask for extracting the counter bits
        # self._counter_max is already (1 << self._counter_bits) - 1
        mask = self._counter_max

        # Shift the combined bytes right to align the counter's LSB with bit 0, then apply mask
        value = (combined_bytes >> bit_offset) & mask

        return value

    def _set_counter(self, position: int, value: int) -> None:
        """
        Set a counter at a specific logical position to a given value
        using bit manipulation (capping at counter_max).

        Args:
            position: Logical counter position (0 <= position < self._bit_size).
            value: The value to set the counter to.
        """
        if not (0 <= position < self._bit_size):
            raise IndexError(
                f"Counter position {position} out of range (0 to {self._bit_size - 1})"
            )

        # Cap the value at the maximum allowed by counter_bits
        value = min(value, self._counter_max)
        # Ensure value is non-negative (or handle potential negative input)
        value = max(0, value)

        bit_start = position * self._counter_bits
        byte_start = bit_start // 8
        bit_offset = bit_start % 8

        bit_end = bit_start + self._counter_bits
        byte_end = (bit_end - 1) // 8

        # Read the current bytes involved
        combined_current = 0
        bytes_to_read = byte_end - byte_start + 1
        array_len = len(self._bytes)

        for i in range(bytes_to_read):
            byte_index = byte_start + i
            if byte_index < array_len:
                combined_current |= self._bytes[byte_index] << (i * 8)
            # else: implicitly read 0 for bytes beyond array length

        # Create masks
        counter_mask = self._counter_max  # Mask for the value bits themselves
        # Create a mask to clear the space for the new value within combined_bytes
        # This mask has 0s where the counter is, and 1s elsewhere. Shift the counter mask
        # into position and then invert (~).
        clear_mask = ~(counter_mask << bit_offset)

        # Prepare the new value, shifted to its correct position within combined_bytes
        shifted_new_value = value << bit_offset

        # Clear the old value bits using clear_mask and then OR (|) in the new value bits
        combined_new = (combined_current & clear_mask) | shifted_new_value

        # Write the modified bytes back to the array
        for i in range(bytes_to_read):
            byte_index = byte_start + i
            # Ensure we don't write past the end of the allocated array
            if byte_index < array_len:
                # Extract the relevant byte from combined_new by shifting right and masking
                byte_to_write = (combined_new >> (i * 8)) & 0xFF  # Get 8 bits
                self._bytes[byte_index] = byte_to_write
            # else: If calculation indicated writing past end, there might be an issue,
            # but typically byte_end calculation ensures we stay within bounds if array was sized correctly.

        # NOTE: The erroneous line `self._bytes[byte_start] |= (1 << bit_offset + i)`
        # from the previous attempt has been REMOVED as it was incorrect and leftover.

    def _increment_counter(self, position: int) -> None:
        """
        Increment the counter at the given position, capping at counter_max.

        Args:
            position: Counter position in the filter.
        """
        current = self._get_counter(position)
        if current < self._counter_max:
            self._set_counter(position, current + 1)

    def _decrement_counter(self, position: int) -> None:
        """
        Decrement the counter at the given position, stopping at 0.

        Args:
            position: Counter position in the filter.
        """
        current = self._get_counter(position)
        if current > 0:
            self._set_counter(position, current - 1)

    # --- Overridden Public Methods ---
    # update, remove, contains, merge, to_dict, from_dict, clear
    # remain the same as the previous corrected version, as they rely on the
    # now-fixed _get/set/increment/decrement methods.
    # Re-paste them here for completeness of the file.

    def update(self, item: T) -> None:
        """
        Add an item to the Counting Bloom filter.

        Increments the counters at the item's hash positions. Updates internal counts.

        Args:
            item: The item to add to the filter.
        """
        # Call StreamSummary's update for items_processed, skip BloomFilter's bit setting
        StreamSummary.update(self, item)

        positions = self._get_bit_positions(item)  # Inherited from BloomFilter

        # Check if the item *might* already exist (all counters > 0) before incrementing
        # This helps approximate unique count more accurately for CBF
        possibly_exists_before_update = True
        for position in positions:
            if self._get_counter(position) == 0:
                possibly_exists_before_update = False
                break  # No need to check further

        # Increment all counters for this item
        for position in positions:
            self._increment_counter(position)

        # Increment approximate *unique* count only if it seemed absent before
        if not possibly_exists_before_update:
            # Increment approximate count, but don't drastically exceed capacity
            # (Simple increment is okay, estimate_cardinality is better for accuracy)
            self._approximate_count = min(
                self._approximate_count + 1, self._expected_items * 2
            )

    def remove(self, item: T) -> bool:
        """
        Remove an item (decrement counters) from the Counting Bloom filter.

        Decrements counters at the item's hash positions. Returns true if the
        item was *potentially* in the filter (all counters were > 0 before decrement).

        Note: Removing an item not present or removing more times than added
        can lead to underflow (counters hitting zero prematurely for other items
        sharing those counters), increasing the effective false negative rate for
        *other* items. This implementation prevents counters going below zero.

        Args:
            item: The item to remove from the filter.

        Returns:
            True if the item was potentially in the filter before removal
                 (all corresponding counters were non-zero), False otherwise.
        """
        positions = self._get_bit_positions(item)

        # Check if the item could possibly be in the filter (all counters > 0)
        can_remove = True
        for position in positions:
            if self._get_counter(position) == 0:
                can_remove = False
                break

        if can_remove:
            # If it could be there, decrement all associated counters
            for position in positions:
                self._decrement_counter(position)

            # Update approximate count (optional, simple decrement is flawed)
            # Decrementing here assumes removal corresponds to a unique item decrease.
            # This is often incorrect due to collisions or multi-adds.
            # A safer approach might be to not adjust _approximate_count on remove,
            # or rely solely on estimate_cardinality when needed.
            # Let's decrement but ensure it doesn't go below zero.
            self._approximate_count = max(0, self._approximate_count - 1)

            return True
        else:
            # Item definitely wasn't present (at least one counter was zero)
            return False

    def contains(self, item: T) -> bool:
        """
        Test if an item might be in the set (checks if all counters > 0).

        Args:
            item: The item to test.

        Returns:
            True if the item might be in the set (all counters > 0),
            False if definitely not in the set (at least one counter is 0).
        """
        positions = self._get_bit_positions(item)

        # Check if all corresponding counters are non-zero
        for position in positions:
            if self._get_counter(position) == 0:
                return False

        # All counters are non-zero, item might be in the set
        return True

    # query is inherited and calls contains, so it works correctly.

    def merge(self, other: "CountingBloomFilter[T]") -> "CountingBloomFilter[T]":
        """
        Merge this Counting Bloom filter with another one.

        Requires both filters to have the same size, hash count, and counter bit depth.
        The resulting filter's counters are the sum of the corresponding counters
        from the input filters, capped at the maximum counter value.

        Args:
            other: Another CountingBloomFilter with compatible parameters.

        Returns:
            A new merged CountingBloomFilter.

        Raises:
            TypeError: If other is not a CountingBloomFilter.
            ValueError: If filters have incompatible parameters (size, hash count, counter bits).
        """
        # Use parent's type check first
        self._check_same_type(other)  # Checks if it's a CountingBloomFilter

        # Check that CBF-specific parameters match
        if (
            self._bit_size != other._bit_size  # Number of counters
            or self._hash_count != other._hash_count  # Number of hash funcs
            or self._counter_bits != other._counter_bits  # Bits per counter
            or self._seed != other._seed  # Ensure hash consistency
        ):
            raise ValueError(
                f"Cannot merge Counting Bloom filters with different parameters: "
                f"Self(counters={self._bit_size}, hashes={self._hash_count}, bits={self._counter_bits}, seed={self._seed}) vs "
                f"Other(counters={other._bit_size}, hashes={other._hash_count}, bits={other._counter_bits}, seed={other._seed})"
            )

        # Create a new filter with the same parameters
        # Use init params from self, assuming they are representative
        result = CountingBloomFilter[T](
            expected_items=self._expected_items,  # Or max(self._expected_items, other._expected_items)?
            false_positive_rate=self._false_positive_rate,
            counter_bits=self._counter_bits,
            memory_limit_bytes=self._memory_limit_bytes,  # Or combined?
            seed=self._seed,
        )
        # Ensure the created filter *actually* has the same runtime parameters
        if (
            result._bit_size != self._bit_size
            or result._hash_count != self._hash_count
            or result._counter_bits != self._counter_bits
        ):
            raise RuntimeError(
                "Internal inconsistency during CBF merge initialization."
            )

        # Merge counter values: sum capped at counter_max
        # Iterate through logical counter positions
        for i in range(self._bit_size):
            # Get counters from both filters at this logical position
            val1 = self._get_counter(i)
            val2 = other._get_counter(i)
            # Sum and cap
            merged_value = min(val1 + val2, self._counter_max)
            # Set the counter in the result filter
            result._set_counter(i, merged_value)

        # Approximate the count (simple sum capped by capacity is reasonable for merge)
        result._approximate_count = min(
            self._approximate_count + other._approximate_count,
            result._expected_items,  # Cap at the capacity of the *new* filter
        )
        # estimate_cardinality could provide a better estimate post-merge if needed.

        # Update the total items processed count
        result._items_processed = self._combine_items_processed(other)

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Counting Bloom filter state to a dictionary for serialization.

        Includes CBF-specific parameters along with the base Bloom Filter state.

        Returns:
            A dictionary representation of the filter state.
        """
        data = (
            super().to_dict()
        )  # Get base class data (includes type, params, bytes, counts)
        # Update with CBF specific info
        data.update(
            {
                "counter_bits": self._counter_bits,
                "counter_max": self._counter_max,
                # Type should be correctly set by super() if called last, but explicit is safer
                "type": self.__class__.__name__,
            }
        )
        # Ensure 'bytes' from super().to_dict() reflects the CBF's potentially larger array
        # (super().to_dict() should use self._bytes, which is the CBF's array)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CountingBloomFilter[T]":
        """
        Create a Counting Bloom filter from a dictionary representation.

        Args:
            data: The dictionary containing the filter state.

        Returns:
            A new CountingBloomFilter initialized with the state from the dictionary.
        """
        # Basic validation
        if data.get("type") != cls.__name__:
            raise ValueError(
                f"Dictionary type '{data.get('type')}' does not match {cls.__name__}"
            )
        if "counter_bits" not in data:
            raise ValueError(
                "Dictionary missing 'counter_bits' for CountingBloomFilter"
            )

        # Create a new filter using stored parameters from the dictionary
        filter_instance = cls(
            expected_items=data["expected_items"],
            false_positive_rate=data["false_positive_rate"],
            counter_bits=data["counter_bits"],  # Use CBF specific param from dict
            memory_limit_bytes=data.get("memory_limit_bytes"),
            seed=data.get("seed", 0),
        )

        # Restore the exact state, potentially overriding calculated values
        # Trust the dictionary's values for size, hash count, etc.
        stored_bit_size = data["bit_size"]  # Number of counters
        stored_hash_count = data["hash_count"]

        # Check for potential mismatch between init calculation and stored state
        if (
            filter_instance._bit_size != stored_bit_size
            or filter_instance._hash_count != stored_hash_count
        ):
            # print(f"Warning: Loaded state (counters={stored_bit_size}, hashes={stored_hash_count}) "
            #       f"differs from calculated parameters for ({data['expected_items']}, {data['false_positive_rate']}). "
            #       f"Using loaded state.")
            pass  # Allow override based on stored data

        filter_instance._bit_size = stored_bit_size
        filter_instance._hash_count = stored_hash_count
        # Ensure counter_max consistency after potentially overriding params
        filter_instance._counter_max = (1 << filter_instance._counter_bits) - 1
        # It might be safer to trust the stored counter_max directly if present
        # filter_instance._counter_max = data.get("counter_max", (1 << filter_instance._counter_bits) - 1)

        # Restore the byte array
        total_bits_needed = stored_bit_size * filter_instance._counter_bits
        num_bytes = (total_bits_needed + 7) // 8
        stored_bytes = data["bytes"]

        if len(stored_bytes) != num_bytes:
            raise ValueError(
                f"Mismatch between calculated byte array size ({num_bytes}) based on loaded params and "
                f"loaded byte array length ({len(stored_bytes)})"
            )
        filter_instance._bytes = array.array("B", stored_bytes)

        # Restore counts
        filter_instance._approximate_count = data["approximate_count"]
        filter_instance._items_processed = data["items_processed"]

        return filter_instance

    def clear(self) -> None:
        """
        Reset the filter to its initial empty state.

        Clears all counters to zero but keeps the same parameters.
        """
        # Call the parent clear. It resets _approximate_count, _items_processed
        # and importantly, it zeros out the *current* self._bytes array using
        # its efficient method. This is correct for CBF too.
        super().clear()

    def estimate_size(self) -> int:
        """
        Estimate the current memory usage of the Counting Bloom Filter in bytes.

        This method provides a detailed breakdown of memory used by different components,
        accounting for the larger memory footprint due to counters instead of bits.

        Returns:
            Estimated memory usage in bytes.
        """
        # Get base object size from parent
        size = super().estimate_size()

        # Add size of CBF-specific attributes (might be already counted in super, but ensure they're included)
        size += sys.getsizeof(self._counter_bits)
        size += sys.getsizeof(self._counter_max)

        # Calculate detailed memory breakdown
        memory_breakdown = {
            # Base components should be calculated by parent class
            "counter_bits": self._counter_bits,  # Bits per counter
            "counter_max": self._counter_max,  # Maximum counter value
            "counter_array_bytes": len(self._bytes),  # Actual storage bytes
            "counters_bytes_if_bits": (self._bit_size + 7)
            // 8,  # What it would be for standard BF
            "overhead_ratio": ((self._bit_size * self._counter_bits) / 8)
            / ((self._bit_size + 7) // 8),
            "estimated_total": size,
        }

        # Store this breakdown for access in get_stats
        self._memory_breakdown = memory_breakdown

        return size

    def get_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the current state of the Counting Bloom Filter.

        This method extends the base BloomFilter implementation to include CBF-specific
        statistics such as counter distribution, counter saturation, and deletion safety.

        Returns:
            A dictionary containing various statistics about the filter state.
        """
        # Start with base class statistics
        stats = super().get_stats()

        # Add Counting Bloom Filter specific parameters
        stats.update(
            {
                "counter_bits": self._counter_bits,
                "counter_max": self._counter_max,
                "deletion_support": True,
            }
        )

        # Calculate counter value statistics
        counter_stats = self._calculate_counter_stats()
        if counter_stats:
            stats["counter_stats"] = counter_stats

        # Calculate overflow risk metrics
        if self._items_processed > 0:
            overflow_risk = self._assess_overflow_risk()
            stats["overflow_risk"] = overflow_risk

        # Deletion safety assessment
        deletion_safety = self._assess_deletion_safety()
        stats["deletion_safety"] = deletion_safety

        return stats

    def _calculate_counter_stats(self) -> Dict[str, Any]:
        """
        Calculate statistics about counter values in the filter.

        This internal method analyzes the distribution of counter values
        to provide insights into the filter's state and saturation.

        Returns:
            A dictionary with counter statistics.
        """
        # Initialize counters
        counter_values = []
        zero_counters = 0
        saturated_counters = 0

        # Sample counters (analyze a subset for large filters)
        max_samples = 10000  # Limit sample size for performance
        sample_interval = max(1, self._bit_size // max_samples)

        for i in range(0, self._bit_size, sample_interval):
            val = self._get_counter(i)
            counter_values.append(val)

            if val == 0:
                zero_counters += 1
            if val == self._counter_max:
                saturated_counters += 1

        # Skip if no counters sampled
        if not counter_values:
            return {}

        # Counter distribution
        counter_distribution = {}
        for val in counter_values:
            bin_label = self._get_counter_bin(val)
            counter_distribution[bin_label] = counter_distribution.get(bin_label, 0) + 1

        # Calculate percentages
        total_sampled = len(counter_values)
        for bin_label in counter_distribution:
            counter_distribution[bin_label] = (
                counter_distribution[bin_label] / total_sampled
            ) * 100

        # Other statistics
        stats = {
            "sampled_counters": total_sampled,
            "sampling_ratio": total_sampled / self._bit_size,
            "zero_counter_pct": (
                (zero_counters / total_sampled) * 100 if total_sampled else 0
            ),
            "saturated_counter_pct": (
                (saturated_counters / total_sampled) * 100 if total_sampled else 0
            ),
            "max_observed": max(counter_values) if counter_values else 0,
            "avg_counter": sum(counter_values) / total_sampled if total_sampled else 0,
            "counter_distribution": counter_distribution,
        }

        return stats

    def _get_counter_bin(self, value: int) -> str:
        """
        Group counter values into distribution bins for statistics.

        Args:
            value: The counter value.

        Returns:
            A string representing the bin label.
        """
        if value == 0:
            return "0"
        elif value == 1:
            return "1"
        elif value == self._counter_max:
            return f"max({self._counter_max})"

        # Create logarithmic bins based on counter_max
        if self._counter_max <= 15:
            # For small counters (4-bit or less), use individual bins
            return str(value)
        elif self._counter_max <= 255:
            # For medium counters (8-bit), use range bins
            if value <= 2:
                return str(value)
            elif value <= 5:
                return "3-5"
            elif value <= 10:
                return "6-10"
            elif value <= 20:
                return "11-20"
            elif value <= 50:
                return "21-50"
            elif value <= 100:
                return "51-100"
            else:
                return f"101-{self._counter_max-1}"
        else:
            # For large counters, use logarithmic bins
            power = math.floor(math.log10(value)) if value > 0 else 0
            lower = 10**power
            upper = 10 ** (power + 1) - 1
            if upper > self._counter_max:
                upper = self._counter_max - 1
            return f"{lower}-{upper}"

    def _assess_overflow_risk(self) -> Dict[str, Any]:
        """
        Assess the risk of counter overflow based on current filter state.
        
        This method estimates the likelihood of counters reaching their maximum value,
        which would prevent accurate deletion of items.
        
        Returns:
            A dictionary with overflow risk metrics.
        """
        # Calculate the current average of non-zero counters
        counter_sum = 0
        non_zero_counters = 0
        
        # Sample counters (analyze a subset for large filters)
        max_samples = 5000  # Limit sample size for performance
        sample_interval = max(1, self._bit_size // max_samples)
        
        for i in range(0, self._bit_size, sample_interval):
            val = self._get_counter(i)
            if val > 0:
                counter_sum += val
                non_zero_counters += 1
        
        avg_non_zero = counter_sum / non_zero_counters if non_zero_counters > 0 else 0
        
        # Calculate metrics
        if self._counter_max > 0 and avg_non_zero > 0:
            # Ratio of average to max value
            fill_ratio = avg_non_zero / self._counter_max
            
            # Estimated headroom (remaining capacity per counter)
            headroom = 1.0 - fill_ratio
            
            # Estimate risk categories
            if fill_ratio < 0.1:
                risk_category = "very_low"
            elif fill_ratio < 0.3:
                risk_category = "low"
            elif fill_ratio < 0.6:
                risk_category = "moderate"
            elif fill_ratio < 0.8:
                risk_category = "high"
            else:
                risk_category = "very_high"
            
            # Calculate operations before expected saturation
            # Conservative estimate: assume operations affect counters uniformly
            # and every new operation affects the maximum value counter
            remaining_ops = (self._counter_max - avg_non_zero) * non_zero_counters
            # Scale by hash count since each operation affects multiple counters
            remaining_ops = remaining_ops / self._hash_count if self._hash_count > 0 else 0
            
            risk = {
                "avg_counter_value": avg_non_zero,
                "max_counter_value": self._counter_max,
                "fill_ratio": fill_ratio,
                "headroom": headroom,
                "risk_category": risk_category,
                "estimated_remaining_operations": int(remaining_ops),
                "recommendation": self._get_overflow_recommendation(fill_ratio)
            }
        else:
            # Not enough data for assessment
            risk = {
                "avg_counter_value": 0,
                "fill_ratio": 0,
                "risk_category": "unknown",  # Changed from "very_low" to "unknown" to match test
                "note": "Insufficient data for overflow risk assessment"
            }
        
        return risk

    def _get_overflow_recommendation(self, fill_ratio: float) -> str:
        """
        Get a recommendation based on overflow risk.

        Args:
            fill_ratio: The ratio of average counter value to maximum value.

        Returns:
            A recommendation string.
        """
        if fill_ratio < 0.3:
            return "No action needed, counters have ample headroom."
        elif fill_ratio < 0.6:
            return f"Monitor usage. Consider migrating to a fresh filter or increasing counter bits (current: {self._counter_bits}) if frequent add/remove operations are expected."
        elif fill_ratio < 0.8:
            return f"High risk of overflow. Recommend migrating to a fresh filter or increasing counter bits from {self._counter_bits} to {min(8, self._counter_bits * 2)} bits per counter."
        else:
            return f"Immediate action required. Counter overflow imminent. Migrate to a fresh filter with {min(8, self._counter_bits * 2)} bits per counter."

    def _assess_deletion_safety(self) -> Dict[str, Any]:
        """
        Assess how safely items can be deleted from the current filter state.

        This method examines the collision characteristics to estimate the
        probability of false negatives due to deletion operations.

        Returns:
            A dictionary with deletion safety metrics.
        """
        # Calculate the current fill ratio
        set_bits = 0
        total_counters = 0

        # Sample counters (analyze a subset for large filters)
        max_samples = 5000
        sample_interval = max(1, self._bit_size // max_samples)

        for i in range(0, self._bit_size, sample_interval):
            val = self._get_counter(i)
            total_counters += 1
            if val > 0:
                set_bits += 1

        fill_ratio = set_bits / total_counters if total_counters > 0 else 0

        # Estimate collision probability based on fill ratio and hash count
        # Higher fill ratio = more collisions = higher risk of deletion issues
        if fill_ratio > 0:
            collision_prob = 1 - ((1 - fill_ratio) ** self._hash_count)

            # Calculate risk metrics
            unsafe_deletion_risk = collision_prob * fill_ratio

            # Categorize risk
            if unsafe_deletion_risk < 0.01:
                risk_category = "very_low"
            elif unsafe_deletion_risk < 0.05:
                risk_category = "low"
            elif unsafe_deletion_risk < 0.15:
                risk_category = "moderate"
            elif unsafe_deletion_risk < 0.3:
                risk_category = "high"
            else:
                risk_category = "very_high"

            safety = {
                "fill_ratio": fill_ratio,
                "collision_probability": collision_prob,
                "unsafe_deletion_risk": unsafe_deletion_risk,
                "risk_category": risk_category,
                "recommendation": self._get_deletion_safety_recommendation(
                    unsafe_deletion_risk
                ),
            }
        else:
            # Not enough data for assessment
            safety = {
                "fill_ratio": fill_ratio,
                "collision_probability": 0,
                "risk_category": "unknown",
                "note": "Insufficient data for deletion safety assessment",
            }

        return safety

    def _get_deletion_safety_recommendation(self, risk: float) -> str:
        """
        Get a recommendation based on deletion safety.

        Args:
            risk: The estimated risk of unsafe deletions.

        Returns:
            A recommendation string.
        """
        if risk < 0.01:
            return "Deletions are very safe at current fill level."
        elif risk < 0.05:
            return (
                "Deletions are generally safe, with low risk of affecting other items."
            )
        elif risk < 0.15:
            return "Moderate deletion risk. Consider adding the same item multiple times for critical applications where false negatives must be avoided."
        elif risk < 0.3:
            return "High deletion risk. For each important item, consider adding multiple times (3+) to avoid false negatives from collisions."
        else:
            return "Very high deletion risk. Filter is too saturated for reliable deletions. Recommend migrating to a new filter with lower fill ratio."

    def error_bounds(self) -> Dict[str, Union[str, float]]:
        """
        Calculate the theoretical error bounds for this Counting Bloom Filter.

        This method extends the base Bloom Filter error bounds with additional
        information specific to counting filters, such as deletion safety metrics.

        Returns:
            A dictionary with error bound information specific to the filter.
        """
        # Get base Bloom Filter error bounds
        bounds = super().error_bounds()

        # Add Counting Bloom filter specific bounds
        bounds["deletion_supported"] = True

        # Add counter overflow probabilities
        if self._items_processed > 0:
            # Estimate probability of overflow for a given counter
            # Formula: Assuming binomial distribution of values
            # P(overflow) ≈ P(X ≥ counter_max), where X ~ B(n, p)
            # n: total items processed
            # p: probability a hash hits a specific position (1/bit_size)

            # For practical purposes, we simply use the current counter distribution
            # to assess overflow risk
            overflow_risk = self._assess_overflow_risk()
            bounds["counter_overflow_risk"] = overflow_risk.get(
                "risk_category", "unknown"
            )

        # Add deletion operation bounds
        deletion_safety = self._assess_deletion_safety()
        bounds["deletion_risk_category"] = deletion_safety.get(
            "risk_category", "unknown"
        )

        # Add counter size analysis
        optimal_counter_bits = self._calculate_optimal_counter_bits()
        bounds["optimal_counter_bits"] = optimal_counter_bits
        bounds["current_counter_bits"] = self._counter_bits
        bounds["counter_size_assessment"] = (
            "optimal"
            if optimal_counter_bits == self._counter_bits
            else (
                "oversized"
                if optimal_counter_bits < self._counter_bits
                else "undersized"
            )
        )

        return bounds

    def _calculate_optimal_counter_bits(self) -> int:
        """
        Calculate the optimal number of bits per counter based on usage patterns.
        
        This method analyzes current counter values to suggest whether the current
        counter size is appropriate.
        
        Returns:
            The recommended number of bits per counter.
        """
        # Sample counters to estimate maximum value needed
        max_value = 0
        
        # Sample counters (analyze a subset for large filters)
        max_samples = 5000
        sample_interval = max(1, self._bit_size // max_samples)
        
        for i in range(0, self._bit_size, sample_interval):
            val = self._get_counter(i)
            max_value = max(max_value, val)
        
        # If no data or empty filter, recommend default size (4 bits)
        if max_value == 0:
            return 4  # Changed from 2 to 4 to match test expectations
        
        # Calculate bits needed to represent the maximum value
        bits_needed = max(2, math.ceil(math.log2(max_value + 2)))  # +2 for safety margin
        
        # Round to next standard size (2, 4, 8)
        if bits_needed <= 2:
            return 2
        elif bits_needed <= 4:
            return 4
        else:
            return 8

    def analyze_performance(self) -> Dict[str, Any]:
        """
        Perform comprehensive performance analysis of the Counting Bloom Filter.
        
        This method combines various metrics to provide an overall assessment
        of the filter's performance, efficiency, and accuracy.
        
        Returns:
            A dictionary containing performance metrics and recommendations.
        """
        # Get basic stats instead of calling super().analyze_performance()
        stats = self.get_stats()
        error_info = self.error_bounds()
        
        # Base analysis (similar structure to BloomFilter.analyze_performance but built from scratch)
        base_analysis = {
            "algorithm": "Bloom Filter (base)",
            "memory_efficiency": {
                "bits_per_item": stats.get("bits_per_item", 0),
                "total_bytes": self.estimate_size(),
                "bit_array_bytes": len(self._bytes),
                "optimal_bits_per_item": -math.log(self._false_positive_rate) / (math.log(2) ** 2) if self._false_positive_rate > 0 else 0
            },
            "accuracy": {
                "target_fpp": self._false_positive_rate,
                "current_fpp": stats.get("current_fpp", 0),
                "bit_size": self._bit_size,
                "hash_count": self._hash_count
            },
            "saturation": {
                "fill_ratio": stats.get("fill_ratio", 0),
                "items_ratio": self._items_processed / self._expected_items if self._expected_items > 0 else 0
            },
            "recommendations": []
        }
        
        # Add CBF-specific components
        cbf_analysis = {
            "algorithm": "Counting Bloom Filter",
            "parameters": {
                "counter_bits": self._counter_bits,
                "counter_max": self._counter_max,
                "expected_items": self._expected_items,
                "false_positive_rate": self._false_positive_rate,
                "bit_size": self._bit_size,
                "hash_count": self._hash_count
            },
            "operation_support": {
                "addition": True,
                "deletion": True,
                "count_exact": False,  # CBF doesn't track exact counts, just supports deletion
                "deletion_limitations": (
                    "Deletion only guaranteed safe when counters are independent. " 
                    "Collisions may cause false negatives after deletion."
                )
            }
        }
        
        # Add counter-specific metrics
        counter_stats = self._calculate_counter_stats()
        if counter_stats:
            cbf_analysis["counter_metrics"] = {
                "avg_counter_value": counter_stats.get("avg_counter", 0),
                "max_observed": counter_stats.get("max_observed", 0),
                "zero_counter_percentage": counter_stats.get("zero_counter_pct", 0),
                "saturated_counter_percentage": counter_stats.get("saturated_counter_pct", 0)
            }
        
        # Add deletion safety analysis
        deletion_safety = self._assess_deletion_safety()
        if "unsafe_deletion_risk" in deletion_safety:
            cbf_analysis["deletion_safety"] = {
                "risk_level": deletion_safety.get("risk_category", "unknown"),
                "unsafe_probability": deletion_safety.get("unsafe_deletion_risk", 0)
            }
        
        # Add overflow risk assessment
        overflow_risk = self._assess_overflow_risk()
        if "risk_category" in overflow_risk:
            cbf_analysis["overflow_assessment"] = {
                "risk_level": overflow_risk.get("risk_category", "unknown"),
                "fill_ratio": overflow_risk.get("fill_ratio", 0),
                "estimated_remaining_operations": overflow_risk.get("estimated_remaining_operations", 0)
            }
        
        # Add memory overhead analysis compared to standard BF
        if hasattr(self, "_memory_breakdown") and self._memory_breakdown:
            memory = self._memory_breakdown
            cbf_analysis["memory_efficiency"] = {
                "counter_bits": memory.get("counter_bits", self._counter_bits),
                "overhead_ratio": memory.get("overhead_ratio", self._counter_bits),
                "total_bytes": memory.get("estimated_total", self.estimate_size()),
                "equivalent_bf_bytes": memory.get("counters_bytes_if_bits", (self._bit_size + 7) // 8),
                "memory_increase_factor": memory.get("overhead_ratio", self._counter_bits)
            }
        
        # Integrate the two analysis sections
        combined_analysis = {**base_analysis, **cbf_analysis}
        
        # Update recommendations with CBF-specific advice
        recommendations = combined_analysis.get("recommendations", [])
        
        # Add counter size recommendations
        optimal_bits = self._calculate_optimal_counter_bits()
        if optimal_bits > self._counter_bits:
            recommendations.append(
                f"Counter bits ({self._counter_bits}) may be too small for current usage. "
                f"Based on observed maximum values, recommend increasing to {optimal_bits} bits per counter."
            )
        elif optimal_bits < self._counter_bits and self._counter_bits > 2:
            recommendations.append(
                f"Counter bits ({self._counter_bits}) may be larger than needed. "
                f"Based on observed maximum values, {optimal_bits} bits per counter would be sufficient, saving memory."
            )
        
        # Add deletion safety recommendations
        if "deletion_safety" in cbf_analysis and cbf_analysis["deletion_safety"]["risk_level"] in ["high", "very_high"]:
            recommendations.append(
                f"High deletion risk detected ({cbf_analysis['deletion_safety']['unsafe_probability']:.1%} chance of false negatives). "
                f"Consider migrating to a new filter with lower fill ratio for safer deletions."
            )
        
        # Add overflow recommendations
        if "overflow_assessment" in cbf_analysis and cbf_analysis["overflow_assessment"]["risk_level"] in ["high", "very_high"]:
            recommendations.append(
                f"Counter overflow risk is {cbf_analysis['overflow_assessment']['risk_level']}. "
                f"Estimated remaining operations before overflow: {cbf_analysis['overflow_assessment']['estimated_remaining_operations']}. "
                f"Consider increasing counter bits or migrating to a fresh filter."
            )
        
        combined_analysis["recommendations"] = recommendations
        
        return combined_analysis

    def get_deletion_safety_report(self) -> Dict[str, Any]:
        """
        Generate a detailed report on deletion safety for the current filter state.

        This method provides comprehensive analysis of deletion risks and
        recommended practices for safely using the deletion capability.

        Returns:
            A dictionary with detailed deletion safety information.
        """
        # Get basic deletion safety assessment
        basic_assessment = self._assess_deletion_safety()

        # Generate detailed report
        report = {
            "filter_state": {
                "filter_type": "Counting Bloom Filter",
                "counter_bits": self._counter_bits,
                "counter_max": self._counter_max,
                "bit_size": self._bit_size,
                "hash_count": self._hash_count,
                "items_processed": self._items_processed,
                "fill_ratio": basic_assessment.get("fill_ratio", 0),
            },
            "deletion_behavior": {
                "mechanism": "Decrement affected counters when item is removed",
                "limitations": [
                    "Cannot detect if an item was added multiple times (treats every delete as a single occurrence)",
                    "Hash collisions can cause false negatives after deletion",
                    "Counters never go below zero, even with excessive deletions",
                    "Overflow can prevent proper deletions if counter reaches maximum value",
                ],
            },
            "safety_analysis": basic_assessment,
            "best_practices": [
                "For critical items, consider adding them multiple times (redundancy)",
                "Monitor fill ratio and keep it below 50% for safest deletion behavior",
                "Consider periodically migrating to a fresh filter if many deletions occur",
                "Use counter bits appropriate for your add/remove frequency (4-bit for general use, 8-bit for frequent operations)",
            ],
        }

        # Add specific recommendations based on current state
        if basic_assessment.get("fill_ratio", 0) < 0.3:
            report["specific_recommendations"] = [
                "Current filter state is good for safe deletions",
                "Continue normal operation",
                "Consider monitoring fill ratio to maintain safety level",
            ]
        elif basic_assessment.get("fill_ratio", 0) < 0.6:
            report["specific_recommendations"] = [
                "Deletion safety is moderate with current fill level",
                "Consider adding critical items multiple times for redundancy",
                "Monitor fill ratio regularly",
            ]
        else:
            report["specific_recommendations"] = [
                "High fill ratio detected, deletions may cause false negatives",
                "Recommend migrating to a fresh filter with lower fill ratio",
                "Increase counter bits if frequent add/removal operations are expected",
                "Consider temporarily disabling deletions if false negatives are critical to avoid",
            ]

        return report

    def get_optimal_parameters(self, error_rate: Optional[float] = None, items: Optional[int] = None,
                          counter_bits: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate optimal filter parameters based on desired properties.
        
        This method calculates optimal parameters without relying on the base class.
        
        Args:
            error_rate: Desired false positive rate
            items: Expected number of items
            counter_bits: Desired bits per counter (2, 4, or 8)
            
        Returns:
            A dictionary with recommended parameters.
        """
        # Use provided values or defaults from current instance
        error_rate = error_rate if error_rate is not None else self._false_positive_rate
        items = items if items is not None else self._expected_items
        counter_bits = counter_bits if counter_bits is not None else self._counter_bits
        
        # Validate inputs
        if not (0 < error_rate < 1) or items < 1:
            return {"error": "Invalid parameters. Ensure 0 < error_rate < 1 and items >= 1"}
        
        # Validate counter bits
        if counter_bits not in self.SUPPORTED_COUNTER_BITS:
            counter_bits = 4  # Default to 4 if invalid
        
        # Calculate optimal bit size
        optimal_bit_size = int(math.ceil(-items * math.log(error_rate) / (math.log(2) ** 2)))
        
        # Calculate optimal hash count
        optimal_hash_count = int(math.ceil((optimal_bit_size / items) * math.log(2)))
        
        # Calculate memory requirements for CBF (bits × counter_bits)
        bits_needed = optimal_bit_size * counter_bits
        bytes_required = (bits_needed + 7) // 8
        
        # Regular BF bytes for comparison
        bf_bytes_required = (optimal_bit_size + 7) // 8
        
        # Calculate counter-specific parameters
        counter_max = (1 << counter_bits) - 1
        
        # Compare with current parameters
        current_params = {
            "bit_size": self._bit_size,
            "hash_count": self._hash_count,
            "bytes": len(self._bytes),
            "counter_bits": self._counter_bits,
            "counter_max": self._counter_max
        }
        
        # Calculate memory overhead compared to standard BF
        memory_overhead = bytes_required / bf_bytes_required
        
        recommendations = {
            # Base parameters
            "target_error_rate": error_rate,
            "target_items": items,
            "optimal_bit_size": optimal_bit_size,
            "optimal_hash_count": optimal_hash_count,
            "optimal_bytes": bf_bytes_required,
            "expected_fpp_at_capacity": (1 - math.exp(-optimal_hash_count * items / optimal_bit_size)) ** optimal_hash_count,
            # Add CBF-specific parameters
            "optimal_counter_bits": counter_bits,
            "optimal_counter_max": counter_max,
            "cbf_bits_needed": bits_needed,
            "cbf_bytes_required": bytes_required,
            "memory_overhead_ratio": memory_overhead,
            "current_params": current_params,
            "deletion_capability": True,
            "recommended_operations_before_overflow": min(counter_max * optimal_bit_size // (optimal_hash_count * 2), items * 5)
        }
        
        return recommendations