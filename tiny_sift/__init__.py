"""
tiny-sift - Lightweight Stream Summarization Library

tiny-sift is a Python library for summarizing data streams with minimal memory footprint
using probabilistic data structures and algorithms.
"""

__version__ = "0.1.0"

# Import main classes to make them available at the top level
from tiny_sift.algorithms.countmin import CountMinSketch
from tiny_sift.algorithms.histogram import ExponentialHistogram
from tiny_sift.algorithms.hyperloglog import HyperLogLog
from tiny_sift.algorithms.reservoir import ReservoirSampling, WeightedReservoirSampling
from tiny_sift.algorithms.spacesaving import SpaceSaving
from .algorithms.quantile_sketch import TDigest
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
    "CountMinSketch",
    "HyperLogLog",
    "ExponentialHistogram",
    "SpaceSaving",
    "TDigest",
]
