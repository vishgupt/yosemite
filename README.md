# HNSW - Hierarchical Navigable Small World

A high-performance Java implementation of the Hierarchical Navigable Small World (HNSW) algorithm for approximate nearest neighbor search in high-dimensional spaces.

## Overview

HNSW is a graph-based algorithm that organizes vectors in a hierarchical structure with multiple layers, enabling efficient approximate nearest neighbor (ANN) search. This implementation provides:

- **Fast Search**: O(log N) average time complexity for nearest neighbor queries
- **Efficient Insertion**: O(log N) average time complexity for adding new vectors
- **Memory Efficient**: O(N) space complexity for storing the graph
- **Scalable**: Handles millions of vectors in high-dimensional spaces
- **Flexible**: Supports custom distance metrics (Euclidean, Cosine similarity)

## Features

### Core Components

1. **Vector** - Immutable vector representation with distance calculations
   - Euclidean distance metric
   - Cosine similarity metric
   - Configurable dimensions

2. **Node** - Graph node representing a vector with multi-layer connections
   - Hierarchical layer structure
   - Layer-specific connections
   - Connection management (add, remove, query)

3. **HNSW** - Main index structure
   - Insert vectors with automatic level assignment
   - Search for k-nearest neighbors
   - Greedy layer-wise search algorithm
   - Connection pruning for bounded degree

4. **SearchRequest** - Encapsulates search parameters
   - Query vector
   - Number of results (topK)
   - Optional search depth limit

5. **SearchResult** - Represents search results
   - Vector ID
   - Distance from query
   - Comparable and sortable

## Installation

### Prerequisites
- Java 11 or higher
- Gradle 9.2.0 or higher

### Build

```bash
./gradlew build
```

### Run Tests

```bash
./gradlew test
```

## Usage

### Basic Example

```java
import com.yosemite.hnsm.*;
import java.util.List;

// Create HNSW index
HNSW hnsw = new HNSW(16, 1.0f / (float) Math.log(2.0));

// Insert vectors
float[] data1 = {1.0f, 2.0f, 3.0f};
Vector v1 = new Vector(1, data1);
hnsw.insert(v1);

float[] data2 = {1.1f, 2.1f, 3.1f};
Vector v2 = new Vector(2, data2);
hnsw.insert(v2);

// Search for nearest neighbors
float[] queryData = {1.05f, 2.05f, 3.05f};
Vector query = new Vector(0, queryData);
SearchRequest request = new SearchRequest(query, 5);
List<SearchResult> results = hnsw.search(request);

// Process results
for (SearchResult result : results) {
    System.out.println("Vector ID: " + result.getVectorId() + 
                       ", Distance: " + result.getDistance());
}
```

### Advanced Configuration

```java
// Create HNSW with custom parameters
int maxConnections = 32;  // Maximum connections per node (upper layers)
float levelMultiplier = 1.0f / (float) Math.log(2.0);  // Layer assignment parameter
HNSW hnsw = new HNSW(maxConnections, levelMultiplier);

// Search with custom parameters
Vector query = new Vector(0, queryData);
SearchRequest request = new SearchRequest(query, 10, 100);  // topK=10, maxSearchDepth=100
List<SearchResult> results = hnsw.search(request);
```

## Algorithm Details

### Insertion Algorithm

1. Assign random level to new node using exponential decay distribution
2. If first node, set as entry point
3. Search for nearest neighbors from entry point down to new node's level
4. At each layer from new node's level down to 0:
   - Find candidate neighbors using greedy search
   - Connect new node to candidates bidirectionally
   - Prune connections if degree exceeds maximum
5. Update entry point if new node has higher level

### Search Algorithm

1. Start from entry point (highest level node)
2. Traverse down through layers, finding nearest neighbor at each level
3. At layer 0, search for k nearest neighbors using greedy search
4. Return results sorted by distance

### Greedy Layer Search

The core search operation at each layer:

1. Initialize with entry points
2. Maintain priority queue of candidates ordered by distance
3. While candidates exist:
   - Pop closest candidate
   - Stop if candidate is farther than best result
   - Explore all unvisited neighbors
   - Add promising neighbors to candidates
   - Maintain top numCandidates results
4. Return results sorted by distance

## Architecture

### Class Hierarchy

```
com.yosemite.hnsm/
├── Vector.java           - Vector representation with distance metrics
├── Node.java             - Graph node with multi-layer connections
├── HNSW.java             - Main index structure
├── SearchRequest.java    - Search parameters
└── SearchResult.java     - Search result with distance
```

### Key Design Decisions

1. **Immutable Vectors**: Vectors are immutable to prevent accidental modifications
2. **Layer-based Connections**: Each node maintains separate connection sets per layer
3. **Bidirectional Edges**: Connections are maintained bidirectionally for consistency
4. **Greedy Search**: Uses greedy algorithm for efficiency at the cost of approximation
5. **Connection Pruning**: Maintains bounded degree by removing farthest connections

## Performance Characteristics

### Time Complexity
- **Insert**: O(log N) average case
- **Search**: O(log N) average case
- **Space**: O(N) for storing the graph

### Parameters

- **maxConnections**: Controls graph sparsity and search quality
  - Higher values: Better search quality, more memory
  - Typical: 16-32

- **levelMultiplier**: Controls layer assignment probability
  - Typical value: 1.0 / ln(2.0) ≈ 1.443
  - Higher values: More layers, slower search but better quality

## Testing

The project includes comprehensive unit tests covering:

- Vector distance and similarity calculations
- Node connection management
- HNSW insertion and search operations
- SearchRequest validation
- Edge cases and error handling

Test coverage: ~80%

Run tests with:
```bash
./gradlew test
```

## API Reference

### HNSW Class

```java
// Constructor
HNSW(int maxConnections, float levelMultiplier)

// Main operations
void insert(Vector vector)
List<SearchResult> search(SearchRequest request)

// Utility methods
int size()
boolean contains(int vectorId)
```

### Vector Class

```java
// Constructor
Vector(int id, float[] data)

// Accessors
int getId()
float[] getData()
int getDimension()

// Distance metrics
float distance(Vector other)
float cosineSimilarity(Vector other)
```

### SearchRequest Class

```java
// Constructors
SearchRequest(Vector queryVector, int topK)
SearchRequest(Vector queryVector, int topK, int maxSearchDepth)

// Accessors
Vector getQueryVector()
int getTopK()
int getMaxSearchDepth()
```

### SearchResult Class

```java
// Constructor
SearchResult(int vectorId, float distance)

// Accessors
int getVectorId()
float getDistance()

// Comparable interface
int compareTo(SearchResult other)
```

## Limitations and Future Work

### Current Limitations

1. **In-Memory Only**: All data must fit in memory
2. **No Persistence**: No built-in serialization/deserialization
3. **Single-Threaded**: Not thread-safe for concurrent operations
4. **Fixed Dimensionality**: All vectors must have same dimension

### Future Enhancements

1. Persistence layer for saving/loading indexes
2. Thread-safe concurrent operations
3. Additional distance metrics (Manhattan, Hamming)
4. Dynamic parameter tuning
5. Batch insertion optimization
6. Index compression techniques

## References

- Malkov, Y., & Yashunin, D. A. (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
- Original paper: https://arxiv.org/abs/1603.09320

## License

This project is part of the Yosemite workspace.

## Contributing

Contributions are welcome! Please ensure:
- All tests pass
- Code follows existing style
- New features include tests
- Documentation is updated

## Author

Vishal Gupta

## Changelog

### Version 1.0.0
- Initial implementation of HNSW algorithm
- Vector and Node classes
- SearchRequest and SearchResult classes
- Comprehensive test suite (~80% coverage)
- Full documentation and comments
