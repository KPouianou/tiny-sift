# TinySift

Lightweight stream summarization library designed to complement GigQ for processing data streams with minimal resources. This library implements probabilistic data structures and algorithms for efficient stream analysis with bounded memory usage.

## Features

- Zero external dependencies
- Memory-efficient probabilistic data structures
- Bounded memory footprint for processing unlimited streams
- Consistent API across all algorithms
- Full integration with GigQ for stream processing jobs

## Installation

```bash
pip install tinysift
```

## Quick Start

```python
from tinysift.algorithms.reservoir import ReservoirSampling

# Create a reservoir with size 100
reservoir = ReservoirSampling(size=100)

# Process a stream of data
for item in data_stream:
    reservoir.update(item)

# Get the current sample
sample = reservoir.get_sample()
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
