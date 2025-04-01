A lightweight stream summarization library that enables processing and analysis of massive data streams with minimal memory requirements. TinySift implements cutting-edge probabilistic data structures and algorithms designed for scenarios where conventional approaches would be prohibitively expensive or impossible due to memory constraints.

## Why TinySift?

Modern data applications often need to analyze streams that are too large to store in memory or process exhaustively. Whether you're tracking unique visitors on a high-traffic website, monitoring network traffic patterns, or analyzing sensor data from IoT devices, processing large datasets typically requires complex infrastructure setup or expensive cloud services.

TinySift solves this problem by providing implementations of advanced algorithms that:

- Process unlimited data in a single pass
- Maintain constant memory usage regardless of stream size
- Offer configurable tradeoffs between accuracy and memory requirements
- Provide probabilistic guarantees on result quality

## Features

- Requires only Python with no external dependencies
- Works with datasets much larger than available RAM
- Processes data in a single pass - ideal for streams or large files
- Provides configurable accuracy vs. memory usage tradeoffs
- Includes serialization for saving and loading processing state
- Supports combining results from separately processed data chunks

## When to Use TinySift

TinySift is ideal for applications where:

- The data stream is too large to store completely
- You need approximate answers to specific questions about the data
- Memory efficiency is a critical constraint
- You can tolerate small, bounded errors in exchange for massive reductions in resource usage
- You need to process data in real-time as it arrives

## Installation

```bash
pip install tinysift
```

## Quick Start

```python
from tiny_sift.algorithms.reservoir import ReservoirSampling

# Create a reservoir with size 100
reservoir = ReservoirSampling(size=100)

# Process a stream of data
for item in data_stream:
    reservoir.update(item)

# Get the current sample
sample = reservoir.get_sample()
```

## Algorithm Selection Guide

This guide helps you select the right algorithm based on your specific use case, memory constraints, and accuracy requirements.

### Algorithm Comparison

| Algorithm             | Primary Use Case                       | Memory Usage                           | Accuracy                                                       | Limitations                                    |
| --------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------- |
| Reservoir Sampling    | Maintain a random sample of fixed size | O(k) where k is sample size            | Exact sampling probability                                     | Cannot change sample size dynamically          |
| HyperLogLog           | Count unique elements                  | O(2^p) where p is precision parameter  | Error ~1.04/sqrt(2^p)                                          | Returns approximate counts only                |
| Count-Min Sketch      | Estimate item frequencies              | O(width × depth)                       | Overestimates by at most N/width with probability 1-2^(-depth) | Never underestimates but may overestimate      |
| Bloom Filter          | Set membership testing                 | O(m) bits for m-bit filter             | False positive rate controllable, no false negatives           | Cannot remove items, cannot count occurrences  |
| Counting Bloom Filter | Set membership with deletion support   | O(m×c) where c is counter bits         | Same as Bloom Filter                                           | Higher memory usage than standard Bloom Filter |
| Exponential Histogram | Sliding window statistics              | O(k log(N)) where k is error parameter | Error bound of ε                                               | Approximates statistics over sliding window    |
| Space-Saving          | Find top-k frequent items              | O(k) where k is number of counters     | Error at most N/k for any item                                 | Only tracks a subset of items                  |
| T-Digest              | Quantile estimation                    | O(k) where k is compression factor     | Better accuracy for extreme quantiles                          | Variable precision across distribution         |

### Decision Tree for Algorithm Selection

#### When memory is extremely limited:

- **Need to count unique items?** → HyperLogLog
- **Need to test set membership?** → Bloom Filter
- **Need frequency estimates?** → Count-Min Sketch

#### When accuracy is critical:

- **Need exact random sampling?** → Reservoir Sampling
- **Need sliding window accuracy?** → Exponential Histogram with smaller error bound
- **Need accurate percentiles?** → T-Digest with higher compression factor

#### When working with specific use cases:

- **Cache implementation with eviction?** → Counting Bloom Filter
- **Finding trending/popular items?** → Space-Saving
- **Monitoring metrics over time?** → Exponential Histogram

### Practical Usage Examples

#### Web Analytics

- **Unique visitors**: HyperLogLog
- **Popular pages**: Space-Saving
- **Session tracking**: Bloom Filter (for fast membership testing)
- **Response time percentiles**: T-Digest

#### Network Monitoring

- **Unique IP addresses**: HyperLogLog
- **Heavy hitters (IPs with most traffic)**: Space-Saving
- **Packet size distribution**: T-Digest
- **Protocol statistics over time windows**: Exponential Histogram

#### Database Systems

- **Query result caching**: Counting Bloom Filter
- **Approximate counts for query planning**: Count-Min Sketch
- **Sampling for approximate queries**: Reservoir Sampling
- **Cardinality estimation for joins**: HyperLogLog

#### IoT & Sensor Networks

- **Periodic sampling of sensor data**: Reservoir Sampling
- **Anomaly detection on sliding windows**: Exponential Histogram
- **Device tracking with minimal memory**: Bloom Filter
- **Estimation of value distribution**: T-Digest

## License

This project is licensed under the MIT License - see the LICENSE file for details.
