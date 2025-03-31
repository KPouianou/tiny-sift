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
from typing import Any, Dict, Optional, TypeVar

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
