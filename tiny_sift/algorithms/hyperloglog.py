"""
Enhanced HyperLogLog implementation with comprehensive benchmarking hooks.

This implementation adds detailed statistics reporting, error bounds analysis,
and memory usage tracking to the existing HyperLogLog implementation.
"""

import array
import math
import sys
from typing import Any, Dict, List, Optional, TypeVar, Union, cast

from tiny_sift.core.base import CardinalityEstimator
from tiny_sift.core.hash import murmurhash3_32

T = TypeVar("T")  # Type for the items being processed


def _count_leading_zeros(x: int, bits: int = 32) -> int:
    """
    Count the number of leading zeros in the binary representation of x.

    Args:
        x: The integer to analyze
        bits: The total number of bits to consider (default: 32 for 32-bit integers)

    Returns:
        The number of leading zeros
    """
    if x == 0:
        return bits

    # Find the position of the highest bit set to 1 (1-indexed)
    position = x.bit_length()

    # Calculate leading zeros (bits - position)
    return bits - position


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
    _THRESHOLD_SMALL = 2.5  # For small cardinality correction

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
        3. Counts the number of leading zeros in the remaining bits + 1
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
        register_index = hash_value & self._index_mask

        # Get the remaining bits by shifting right by 'p' bits
        remaining_hash = hash_value >> self._precision

        # Count leading zeros in the remaining bits
        leading_zeros = _count_leading_zeros(remaining_hash, 32 - self._precision)

        # The rank is the position of the first 1-bit, so it's leading_zeros + 1
        rank = leading_zeros + 1

        # Update the register if the new rank is larger than the current value
        if rank > self._registers[register_index]:
            self._registers[register_index] = rank

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

        Returns:
            The estimated number of unique items in the stream.
        """
        # Special case: empty estimator
        if self._is_empty:
            return 0

        # Compute the "raw" estimate using the HyperLogLog algorithm
        # We use the harmonic mean of 2^(-register_value)
        sum_of_inverses = 0.0
        zero_registers = 0

        # Process each register
        for register_value in self._registers:
            sum_of_inverses += math.pow(2.0, -register_value)
            if register_value == 0:
                zero_registers += 1

        # Calculate the raw estimate: alpha * m^2 / sum(2^(-M[j]))
        raw_estimate = self._alpha * (self._m**2) / sum_of_inverses

        # Apply small range correction if necessary
        # If there are empty registers, use linear counting
        if raw_estimate <= self._THRESHOLD_SMALL * self._m and zero_registers > 0:
            # Linear counting formula: m * ln(m / V)
            # where m is the number of registers and V is the number of empty registers
            return int(round(self._m * math.log(self._m / zero_registers)))

        # No correction needed, return the raw estimate
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

    def clear(self) -> None:
        """
        Reset the estimator to its initial empty state.
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

        # For very precise memory limits, use a conservative approach
        # Start with minimum precision and increase until we hit the limit
        for precision in range(4, 17):  # 4 to 16
            # Create a test instance with this precision
            test_instance = cls(precision=precision, seed=seed)
            size = test_instance.estimate_size()

            # If this instance exceeds the memory limit, use the previous precision
            if size > memory_bytes:
                if precision > 4:
                    return cls(
                        precision=precision - 1,
                        memory_limit_bytes=memory_bytes,
                        seed=seed,
                    )
                else:
                    raise ValueError(
                        f"Memory limit of {memory_bytes} bytes is too small. "
                        f"Minimum size is approximately {size} bytes."
                    )

            # If this is the maximum precision, just return it
            if precision == 16:
                return cls(
                    precision=precision, memory_limit_bytes=memory_bytes, seed=seed
                )

        # If we've exhausted all precisions without exceeding the limit,
        # use the maximum precision
        return cls(precision=16, memory_limit_bytes=memory_bytes, seed=seed)

    def get_register_values(self) -> List[int]:
        """
        Get the current values of all registers.

        This method is useful for analyzing the internal state of the HyperLogLog
        estimator and for debugging or visualization purposes.

        Returns:
            A list containing the current register values.
        """
        return list(self._registers)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the current state of the HyperLogLog estimator.

        This method extends the base class implementation to include HyperLogLog-specific
        statistics such as register distribution, precision parameters, and error bounds.

        Returns:
            A dictionary containing various statistics about the estimator.
        """
        # Start with base class statistics
        stats = super().get_stats()

        # Add HyperLogLog specific parameters
        stats.update(
            {
                "precision": self._precision,
                "num_registers": self._m,
                "alpha_value": self._alpha,
            }
        )

        # Add register statistics if not empty
        if not self._is_empty:
            register_values = list(self._registers)
            empty_registers = register_values.count(0)
            max_register = max(register_values) if register_values else 0
            register_sum = sum(register_values)

            # Calculate register value distribution
            register_distribution = {}
            for value in range(max_register + 1):
                count = register_values.count(value)
                if count > 0:
                    register_distribution[str(value)] = (
                        count  # Ensure keys are strings for JSON compatibility
                    )

            # Register distribution stats
            stats.update(
                {
                    "empty_registers": empty_registers,
                    "empty_registers_pct": (
                        (empty_registers / self._m) * 100 if self._m > 0 else 0
                    ),
                    "max_register_value": max_register,
                    "avg_register_value": register_sum / self._m if self._m > 0 else 0,
                    "register_value_distribution": register_distribution,
                }
            )

            # Add error bounds from the error_bounds method
            error_bounds = self.error_bounds()
            stats.update(error_bounds)

            # Add cardinality estimate
            estimate = self.estimate_cardinality()
            stats["estimate"] = estimate

            # Add theoretical limits
            stats["theoretical_max_countable"] = 2 ** (32 - self._precision)
            stats["saturation_pct"] = min(
                100.0, (estimate / stats["theoretical_max_countable"]) * 100
            )

        # Ensure memory_usage is present (for backward compatibility)
        if "memory_usage" not in stats:
            stats["memory_usage"] = self.estimate_size()

        return stats

    def error_bounds(self) -> Dict[str, Union[str, float]]:
        """
        Calculate the theoretical error bounds for this estimator.

        This method provides detailed error bound information specific to HyperLogLog,
        including confidence intervals at different levels.

        Returns:
            A dictionary with the error bounds:
            - relative_error: The standard error (approximately 1.04/sqrt(m))
            - confidence_68pct: Error range for 68% confidence (1 sigma)
            - confidence_95pct: Error range for 95% confidence (2 sigma)
            - confidence_99pct: Error range for 99% confidence (3 sigma)
        """
        # Start with any values from the base class
        bounds = super().error_bounds()

        # Calculate the standard error: 1.04/sqrt(m)
        std_error = 1.04 / math.sqrt(self._m)

        # Update with HyperLogLog-specific error bounds
        bounds.update(
            {
                "relative_error": std_error,
                "confidence_68pct": std_error,  # 1 sigma
                "confidence_95pct": std_error * 1.96,  # 2 sigma
                "confidence_99pct": std_error * 2.58,  # 3 sigma
            }
        )

        return bounds

    def estimate_size(self) -> int:
        """
        Estimate the current memory usage of the HyperLogLog in bytes.

        This method extends the base class implementation to provide a more detailed
        breakdown of the memory used by the HyperLogLog data structure.

        Returns:
            Estimated size in bytes.
        """
        # Start with the base object size from the parent class
        size = super().estimate_size()

        # Add size of HyperLogLog-specific attributes
        # Note: some of these might already be included in the base calculation
        # but we add them here for completeness
        size += sys.getsizeof(self._precision)
        size += sys.getsizeof(self._m)
        size += sys.getsizeof(self._alpha)
        size += sys.getsizeof(self._index_mask)
        size += sys.getsizeof(self._is_empty)
        size += sys.getsizeof(self._seed)

        # Size of the registers array - this should be the largest component
        # and might not be fully accounted for in the base implementation
        size += sys.getsizeof(self._registers)

        # Some implementations include the buffer separately,
        # but getsizeof on array.array already includes its buffer

        # Account for min/max values if they exist and aren't in the base calculation
        if hasattr(self, "_min_val") and self._min_val is not None:
            size += sys.getsizeof(self._min_val)
        if hasattr(self, "_max_val") and self._max_val is not None:
            size += sys.getsizeof(self._max_val)

        return size

    def analyze_performance(self) -> Dict[str, Any]:
        """
        Perform a detailed analysis of the HyperLogLog estimator's performance.

        This method provides insights into the estimator's efficiency and
        accuracy characteristics based on its current state and theoretical properties.

        Returns:
            A dictionary containing performance analysis metrics.
        """
        analysis = {
            "algorithm": "HyperLogLog",
            "memory_efficiency": {
                "bits_per_item": (self.estimate_size() * 8)
                / max(1, self.estimate_cardinality()),
                "total_bytes": self.estimate_size(),
                "register_bytes": self._m * self._registers.itemsize,
                "overhead_bytes": self.estimate_size()
                - (self._m * self._registers.itemsize),
            },
            "accuracy": {
                "precision_parameter": self._precision,
                "standard_error": 1.04 / math.sqrt(self._m),
                "expected_accuracy": f"{(1 - (1.04 / math.sqrt(self._m))) * 100:.2f}%",
            },
            "saturation": {
                "empty_registers": sum(1 for r in self._registers if r == 0),
                "empty_register_pct": sum(1 for r in self._registers if r == 0)
                / self._m
                * 100,
                "max_register_value": max(self._registers) if not self._is_empty else 0,
                "theoretical_max_value": 32 - self._precision,
                "saturation_level": (
                    max(self._registers) / (32 - self._precision)
                    if not self._is_empty
                    else 0
                ),
            },
        }

        # Add recommendations based on the analysis
        analysis["recommendations"] = []

        # Check if too many registers are empty (underutilized)
        if (
            analysis["saturation"]["empty_register_pct"] > 50
            and self.items_processed > 1000
        ):
            analysis["recommendations"].append(
                "Consider reducing precision to save memory (many empty registers)"
            )

        # Check if max register value is close to theoretical maximum (potential saturation)
        if analysis["saturation"]["saturation_level"] > 0.9:
            analysis["recommendations"].append(
                "Warning: Approaching register value saturation, may affect accuracy for larger cardinalities"
            )

        # Check memory efficiency
        if (
            analysis["memory_efficiency"]["bits_per_item"] > 20
            and self.items_processed > 1000
        ):
            analysis["recommendations"].append(
                "Memory usage is high relative to cardinality, consider reducing precision"
            )

        return analysis
