"""
HyperLogLog implementation for TinySift.

This module provides an implementation of the HyperLogLog algorithm, a probabilistic
data structure used for estimating the cardinality (number of unique elements) of
large data streams with bounded memory usage.

HyperLogLog provides the following guarantees:
1. Space Complexity: O(2^p), where p is the precision parameter
2. Update Time: O(1)
3. Query Time: O(2^p)
4. Relative Error: Approximately 1.04/sqrt(2^p)

The algorithm works by using hash functions to map elements to registers and
recording the maximum number of leading zeros in the binary representation of
each register's hash value. These observations are then combined to estimate
the total cardinality.

References:
    - Flajolet, P., Fusy, É., Gandouet, O., & Meunier, F. (2007).
      HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm.
      In AofA: Analysis of Algorithms (pp. 137-156).
    - Heule, S., Nunkesser, M., & Hall, A. (2013, June).
      HyperLogLog in practice: algorithmic engineering of a state of the art
      cardinality estimation algorithm. In EDBT 2013: Proceedings of the 16th
      International Conference on Extending Database Technology (pp. 683-692).
"""

import array
import math
import sys
from typing import Any, Dict, List, Optional, TypeVar, Union, cast

from tiny_sift.core.base import CardinalityEstimator
from tiny_sift.core.hash import murmurhash3_32

T = TypeVar("T")  # Type for the items being processed


class HyperLogLog(CardinalityEstimator[T]):
    """
    HyperLogLog for cardinality estimation in data streams.

    HyperLogLog is an algorithm for estimating the number of unique elements in a
    large data stream using a minimal amount of memory. It's based on the observation
    that the cardinality of a set can be estimated by keeping track of the maximum
    number of leading zeros in the binary representation of hash values.

    The precision parameter (p) determines both the accuracy and the memory usage:
    - Memory usage is approximately 2^p registers (each storing a small counter)
    - The standard error is roughly 1.04/sqrt(2^p)

    For common use cases:
    - p=10: ~1KB memory, ~3.25% error
    - p=12: ~4KB memory, ~1.62% error (default)
    - p=14: ~16KB memory, ~0.81% error
    - p=16: ~64KB memory, ~0.40% error

    References:
        - Flajolet, P., Fusy, É., Gandouet, O., & Meunier, F. (2007).
          HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm.
    """

    # Alpha values for the HyperLogLog algorithm
    # These are constants used in the cardinality estimation formula
    # They are derived analytically and depend on the number of registers (m)
    # The index corresponds to the precision parameter
    _ALPHA = [0, 0, 0.351, 0.532, 0.625, 0.673, 0.697, 0.709]
    _ALPHA.extend([0.7213 / (1.0 + 1.079 / m) for m in [2**p for p in range(8, 64)]])

    # Constants for various bias correction thresholds
    _THRESHOLD_SMALL = 5  # For small cardinality correction
    _THRESHOLD_LINEAR_COUNTING = 2.5  # For linear counting switch

    def __init__(
        self,
        precision: int = 12,
        memory_limit_bytes: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize a new HyperLogLog estimator.

        Args:
            precision: The precision parameter (p), controlling accuracy and memory usage.
                      Valid values are 4 to 16. Higher values increase accuracy but use more memory.
                      Default is 12, providing ~1.62% standard error.
            memory_limit_bytes: Optional maximum memory usage in bytes.
            seed: Optional random seed for hash functions.

        Raises:
            ValueError: If precision is outside the valid range of 4 to 16.
        """
        super().__init__(memory_limit_bytes)

        # Validate precision parameter
        if not 4 <= precision <= 16:
            raise ValueError("Precision must be between 4 and 16 (inclusive)")

        self._precision = precision
        # Number of registers (m = 2^precision)
        self._m = 1 << precision
        # Calculate alpha constant for given precision
        self._alpha = self._ALPHA[precision]

        # Bit mask for register index extraction (the first 'p' bits of the hash)
        self._index_mask = self._m - 1

        # Initialize the registers (using array.array for memory efficiency)
        # Each register stores the maximum number of leading zeros observed
        # 'B' typecode gives unsigned char (8 bits, 0 to 255)
        # Since register values represent run lengths, they won't exceed 32
        # for a 32-bit hash function (and we only need 5 bits theoretically)
        self._registers = array.array("B", [0] * self._m)

        # Track if we've seen any items yet
        self._is_empty = True

        # Store the seed for hash function
        self._seed = seed if seed is not None else 0

    def update(self, item: T) -> None:
        """
        Update the HyperLogLog estimator with a new item from the stream.

        This method:
        1. Hashes the item to get a 32-bit value
        2. Uses the first 'p' bits to determine which register to update
        3. Counts the number of leading zeros in the remaining bits
        4. Updates the register with the maximum value observed

        Args:
            item: The item to add to the estimator.
        """
        # Increment the items processed counter
        super().update(item)

        # Mark that we've seen at least one item
        self._is_empty = False

        # Hash the item to a 32-bit value
        hash_value = murmurhash3_32(item, seed=self._seed)

        # Extract the register index from the first 'p' bits of the hash
        # We can use bitwise AND with the precalculated mask for this
        register_index = hash_value & self._index_mask

        # Calculate the pattern (number of leading zeros + 1) in the remaining bits
        # First, we shift right by 'p' bits to remove the bits used for the index
        remaining_bits = hash_value >> self._precision

        # Count the number of leading zeros + 1
        # We add 1 because we're counting the position of the first 1-bit
        # If no 1-bit is found, Python would return 32, but we'll see a max of 32-p leading zeros
        # in the remaining bits after removing the first p bits
        pattern = 1
        # Start with the highest bit position in the remaining bits (31-p)
        # and go down until we find a 1
        for bit_pos in range(31 - self._precision, -1, -1):
            if (remaining_bits >> bit_pos) & 1:
                # Found the first 1-bit, the pattern is position + 1
                pattern = bit_pos + 1
                break
            # If we've checked all bits and found none, pattern will remain 1

        # Alternative implementation using the position of the highest bit set:
        # pattern = 1
        # if remaining_bits != 0:
        #     pattern = 1 + (remaining_bits.bit_length() - 1)

        # Update the register if the new pattern is larger than the current value
        if pattern > self._registers[register_index]:
            self._registers[register_index] = pattern

    def query(self, *args: Any, **kwargs: Any) -> int:
        """
        Query the current state of the HyperLogLog estimator.

        This is a convenience method that calls estimate_cardinality.

        Returns:
            The estimated cardinality.
        """
        return self.estimate_cardinality()

    def estimate_cardinality(self) -> int:
        """
        Estimate the number of unique items in the stream.

        This method implements the full HyperLogLog algorithm with various corrections:
        1. Basic HyperLogLog formula for normal range
        2. Linear Counting for small cardinalities
        3. Large range correction for very large cardinalities

        Returns:
            The estimated number of unique items in the stream.
        """
        # Special case: empty estimator
        if self._is_empty:
            return 0

        # Compute the "raw" estimate using the HyperLogLog algorithm
        # First, calculate the harmonic mean of the register values
        # transformed by 2^(-register_value)
        sum_of_inverses = 0.0
        num_zero_registers = 0

        # Process each register
        for register_value in self._registers:
            # 2^(-register_value)
            sum_of_inverses += 2.0**-register_value
            # Count zero registers for linear counting if needed
            if register_value == 0:
                num_zero_registers += 1

        # The raw estimate formula is:
        # E = alpha_m * m^2 * (sum(2^(-M[j])))^(-1)
        # where alpha_m is a constant, m is the number of registers,
        # and M[j] is the value of register j
        raw_estimate = self._alpha * (self._m**2) * (1.0 / sum_of_inverses)

        # Apply corrections based on the estimate range
        # These corrections improve accuracy for small and large cardinalities

        # Small range correction using Linear Counting
        # If the estimate is small or many registers are still empty,
        # use linear counting for better accuracy
        if (
            raw_estimate <= self._m * self._THRESHOLD_LINEAR_COUNTING
            and num_zero_registers > 0
        ):
            # Linear Counting formula: m * ln(m / V)
            # where m is the number of registers and V is the number of zero registers
            return int(round(self._m * math.log(self._m / num_zero_registers)))

        # Large range correction (not typically needed with 32-bit hashes)
        # Only apply for very large streams approaching 2^32 unique values
        # This is the "large range correction" from the HyperLogLog paper
        elif raw_estimate > 2**32 / 30.0:
            # The correction for large cardinalities prevents overflow
            return int(round(-(2**32) * math.log(1.0 - raw_estimate / 2**32)))

        # Normal range: use the raw estimate directly
        else:
            return int(round(raw_estimate))

    def merge(self, other: "HyperLogLog[T]") -> "HyperLogLog[T]":
        """
        Merge this HyperLogLog with another one.

        This operation allows combining estimates from different streams.
        The merged estimator will have the same cardinality estimate as if
        all items from both streams were processed by a single estimator.

        Args:
            other: Another HyperLogLog estimator.

        Returns:
            A new merged HyperLogLog estimator.

        Raises:
            TypeError: If other is not a HyperLogLog.
            ValueError: If estimators have different precision.
        """
        # Check that other is a HyperLogLog
        self._check_same_type(other)

        # Check that precision parameters match
        if self._precision != other._precision:
            raise ValueError(
                f"Cannot merge HyperLogLog estimators with different precision: "
                f"{self._precision} and {other._precision}"
            )

        # Create a new estimator with the same parameters
        result = HyperLogLog[T](
            precision=self._precision,
            memory_limit_bytes=self._memory_limit_bytes,
            seed=self._seed,
        )

        # Merge the registers by taking the maximum value for each position
        for i in range(self._m):
            result._registers[i] = max(self._registers[i], other._registers[i])

        # Update the empty flag - result is empty only if both inputs are empty
        result._is_empty = self._is_empty and other._is_empty

        # Update the items processed count
        result._items_processed = self._combine_items_processed(other)

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the HyperLogLog estimator to a dictionary for serialization.

        Returns:
            A dictionary representation of the estimator.
        """
        data = self._base_dict()
        data.update(
            {
                "precision": self._precision,
                "registers": list(self._registers),
                "is_empty": self._is_empty,
                "seed": self._seed,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyperLogLog[T]":
        """
        Create a HyperLogLog estimator from a dictionary representation.

        Args:
            data: The dictionary containing the estimator state.

        Returns:
            A new HyperLogLog estimator.
        """
        # Create a new estimator with the saved precision
        estimator = cls(
            precision=data["precision"],
            memory_limit_bytes=data.get("memory_limit_bytes"),
            seed=data.get("seed", 0),
        )

        # Restore the register values
        estimator._registers = array.array("B", data["registers"])

        # Restore other state
        estimator._is_empty = data["is_empty"]
        estimator._items_processed = data["items_processed"]

        return estimator

    def estimate_size(self) -> int:
        """
        Estimate the current memory usage of this estimator in bytes.

        Returns:
            Estimated memory usage in bytes.
        """
        # Base size of the object
        size = sys.getsizeof(self)

        # Size of the registers array
        size += sys.getsizeof(self._registers)

        return size

    def error_bound(self) -> Dict[str, float]:
        """
        Calculate the theoretical error bounds for this estimator.

        Returns:
            A dictionary with the error bounds:
            - relative_error: The standard error (approximately 1.04/sqrt(m))
            - confidence_68pct: Error range for 68% confidence
            - confidence_95pct: Error range for 95% confidence
            - confidence_99pct: Error range for 99% confidence
        """
        # The standard error is approximately 1.04/sqrt(m)
        std_error = 1.04 / math.sqrt(self._m)

        return {
            "relative_error": std_error,
            "confidence_68pct": std_error,  # 1 sigma
            "confidence_95pct": std_error * 1.96,  # 2 sigma
            "confidence_99pct": std_error * 2.58,  # 3 sigma
        }

    def clear(self) -> None:
        """
        Reset the estimator to its initial state.

        This method clears all registers but keeps the same precision.
        """
        for i in range(self._m):
            self._registers[i] = 0

        self._is_empty = True
        self._items_processed = 0

    @classmethod
    def create_from_error_rate(
        cls,
        relative_error: float,
        memory_limit_bytes: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> "HyperLogLog[T]":
        """
        Create a HyperLogLog estimator with the desired error guarantees.

        Args:
            relative_error: The target relative error (standard error)
                           For example, 0.01 means a target error of 1%
            memory_limit_bytes: Optional maximum memory usage in bytes
            seed: Optional random seed for hash functions

        Returns:
            A new HyperLogLog estimator configured for the specified error bound

        Raises:
            ValueError: If relative_error is too small to achieve with valid precision,
                       or if relative_error is not between 0 and 1
        """
        if not (0 < relative_error < 1):
            raise ValueError("Relative error must be between 0 and 1")

        # Calculate required precision
        # Standard error = 1.04/sqrt(2^p), so we solve for p:
        # p = log2((1.04/relative_error)^2)
        precision = math.ceil(math.log2((1.04 / relative_error) ** 2))

        # Limit precision to valid range
        if precision < 4:
            precision = 4
        elif precision > 16:
            raise ValueError(
                f"Relative error of {relative_error} is too small to achieve "
                f"with maximum precision of 16. Minimum achievable error is approximately 0.026%."
            )

        return cls(
            precision=precision, memory_limit_bytes=memory_limit_bytes, seed=seed
        )

    @classmethod
    def create_from_memory_limit(
        cls,
        memory_bytes: int,
        seed: Optional[int] = None,
    ) -> "HyperLogLog[T]":
        """
        Create a HyperLogLog estimator optimized for a given memory limit.

        This factory method chooses the highest precision that fits within
        the specified memory budget, providing the best possible accuracy.

        Args:
            memory_bytes: Maximum memory usage in bytes
            seed: Optional random seed for hash functions

        Returns:
            A new HyperLogLog estimator optimized for the memory constraint

        Raises:
            ValueError: If memory_bytes is too small for even the minimum precision
        """
        if memory_bytes <= 0:
            raise ValueError("Memory limit must be positive")

        # Calculate the precision based on memory
        # Each register uses 1 byte, and we have 2^p registers
        # We also need to account for the overhead of the object itself
        # Estimate the base size of a HyperLogLog object with p=4 (minimum)
        min_estimator = cls(precision=4)
        base_size = min_estimator.estimate_size() - (1 << 4)  # Subtract the registers

        # Available space for registers
        available_bytes = memory_bytes - base_size

        # Calculate maximum precision that fits
        if available_bytes <= 16:  # 2^4 = 16 (minimum)
            raise ValueError(
                f"Memory limit of {memory_bytes} bytes is too small. "
                f"Minimum required is approximately {base_size + 16} bytes."
            )

        max_precision = min(16, math.floor(math.log2(available_bytes)))

        return cls(precision=max_precision, memory_limit_bytes=memory_bytes, seed=seed)

    def get_register_values(self) -> List[int]:
        """
        Get the current values of all registers (useful for debugging).

        Returns:
            A list containing the current register values.
        """
        return list(self._registers)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current state of the estimator.

        Returns:
            A dictionary with various statistics:
            - precision: The precision parameter (p)
            - num_registers: The number of registers (m)
            - empty_registers: The number of registers with value 0
            - max_register_value: The maximum value across all registers
            - estimate: The current cardinality estimate
            - relative_error: The theoretical relative error
            - memory_usage: Estimated memory usage in bytes
        """
        # Count empty registers and find maximum value
        empty_registers = 0
        max_value = 0
        for value in self._registers:
            if value == 0:
                empty_registers += 1
            max_value = max(max_value, value)

        return {
            "precision": self._precision,
            "num_registers": self._m,
            "empty_registers": empty_registers,
            "max_register_value": max_value,
            "estimate": self.estimate_cardinality(),
            "relative_error": self.error_bound()["relative_error"],
            "memory_usage": self.estimate_size(),
        }
