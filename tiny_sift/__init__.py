"""
tiny-sift - Lightweight Stream Summarization Library

tiny-sift is a Python library for summarizing data streams with minimal memory footprint
using probabilistic data structures and algorithms.
"""

__version__ = "0.1.0"

# Import main classes to make them available at the top level
from tiny_sift.algorithms.reservoir import ReservoirSampling, WeightedReservoirSampling
from tiny_sift.core.base import (
    CardinalityEstimator,
    FrequencyEstimator,
    SampleMaintainer,
    StreamSummary,
    WindowAggregator,
)

__all__ = [
    # Core base classes
    "StreamSummary",
    "SampleMaintainer",
    "FrequencyEstimator",
    "CardinalityEstimator",
    "WindowAggregator",
    # Algorithm implementations
    "ReservoirSampling",
    "WeightedReservoirSampling",
]
