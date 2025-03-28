"""
TinySift - Lightweight Stream Summarization Library

TinySift is a Python library for summarizing data streams with minimal memory footprint
using probabilistic data structures and algorithms.
"""

__version__ = "0.1.0"

# Import main classes to make them available at the top level
from tinysift.algorithms.reservoir import ReservoirSampling, WeightedReservoirSampling
from tinysift.core.base import (
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