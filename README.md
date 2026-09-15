# HNSW

A modern lightweight C++ implementation of **Hierarchical Navigable Small World (HNSW)** for approximate nearest neighbor search.

## Highlights

1. Modern C++ implementation using templates and concepts.
2. HNSW index with configurable `M`, `efConstruction` and `efSearch`.
3. Incremental vector insertion and approximate `k-NN` search.
4. Multi-layer graph construction with bidirectional neighbor connections.
5. Generic metric interface using C++ concepts.
6. Benchmarking against brute-force nearest neighbor search.

## Current implementation

The main search and insertion logic is implemented around the standard HNSW layered graph structure.

## Benchmark

Current benchmark:

```text
HNSW benchmark
--------------
Vectors:        25000
Dimensions:     16
Queries:        1000
K:              10
M:              16
efConstruction: 200

efSearch    Recall@K       Brute (ms/query)    HNSW (ms/query)
-------------------------------------------------------------------
10          0.5873         9.0411              0.4394
20          0.7483         9.0487              0.5701
50          0.9067         9.0569              0.9606
100         0.9660         9.0451              1.5963
200         0.9871         9.0659              2.7731
```

As expected, increasing `efSearch` improves recall at the cost of search time.

For example, with `efSearch = 100`:

```text
Recall@10:        0.9660
Brute force:      9.0451 ms/query
HNSW:             1.5963 ms/query
```

## Roadmap

The next steps are mainly focused on performance, usability and robustness.

### Multithreading

* Parallel search
* Parallel insertion
* Faster index construction

### SIMD distance computation

* SSE
* AVX
* AVX2
* AVX-512
* FMA where supported

### Python bindings

* Python API for index creation, insertion and search
* Easier experimentation and benchmarking

### Exception handling

* Validation of configuration and input data
* Proper handling of invalid operations and runtime errors

### Custom distances

* User-defined metrics through the existing generic metric interface
* Possibility of optimized implementations for specific metrics

## Goals

The main goal is to evolve the current implementation into a fast and extensible HNSW library with:

```text
C++
 ├── HNSW
 ├── custom metrics
 ├── SIMD acceleration
 ├── multithreading
 ├── Python bindings
 ├── exception handling
 └── benchmarking
```

The benchmark will be extended as new optimizations are introduced, allowing the performance impact of each change to be measured.

