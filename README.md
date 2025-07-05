# SwiftVS - Swift Vector Database Framework

[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/Platform-iOS%20%7C%20macOS%20%7C%20tvOS%20%7C%20watchOS-blue.svg)](https://developer.apple.com/swift/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Swift-native vector database framework that provides efficient storage, indexing, and similarity search for high-dimensional vectors on Apple platforms.

## 🚀 Features

- **High-Dimensional Vector Storage**: Store vectors with any dimension along with flexible metadata
- **Multiple Distance Metrics**: Cosine similarity, Euclidean distance, and dot product
- **Persistent Storage**: SQLite-based storage with ACID compliance
- **Efficient Search**: Brute-force and Accelerate-optimized search algorithms
- **Thread-Safe**: Built with Swift's actor model for concurrent access
- **Metadata Support**: Rich metadata storage with JSON serialization
- **Import/Export**: JSON and CSV export capabilities
- **Performance**: SIMD-accelerated operations using Apple's Accelerate framework
- **Extensible**: Designed for future ANN indexing algorithms (HNSW, IVF)

## 📦 Installation

### Swift Package Manager

Add SwiftVS to your project using Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/SwiftVS.git", from: "1.0.0")
]
```

### Xcode Integration

1. Open your project in Xcode
2. Go to **File → Add Package Dependencies**
3. Enter the repository URL: `https://github.com/yourusername/SwiftVS.git`
4. Select the version and add to your target

## 🎯 Quick Start

### Basic Usage

```swift
import SwiftVS

// Create a vector database
let db = try VectorDatabase.create(at: "vectors.db")

// Create vectors with metadata
let vector1 = Vector(
    values: [0.1, 0.2, 0.3, 0.4],
    metadata: [
        "title": AnyCodable("Document 1"),
        "category": AnyCodable("AI/ML")
    ]
)

let vector2 = Vector(
    values: [0.5, 0.6, 0.7, 0.8],
    metadata: [
        "title": AnyCodable("Document 2"),
        "category": AnyCodable("Data Science")
    ]
)

// Insert vectors
try await db.insert(vector1)
try await db.insert(vector2)

// Perform similarity search
let queryVector = Vector(values: [0.2, 0.3, 0.4, 0.5])
let results = try await db.search(query: queryVector, k: 5)

// Process results
for result in results {
    let title = result.vector.metadata?["title"]?.value as? String ?? "Unknown"
    print("Title: \(title), Similarity: \(result.score)")
}
```

### Advanced Configuration

```swift
// Configure database with custom options
let config = VectorDatabaseConfig(
    databasePath: "vectors.db",
    expectedDimension: 128,
    searchType: .accelerated,
    useAccelerate: true,
    defaultSearchOptions: SearchOptions(
        metric: .cosine,
        normalizeQuery: true,
        normalizeStored: true
    )
)

let db = try VectorDatabase(config: config)
```

### Search with Custom Options

```swift
// Search with metadata filtering
let searchOptions = SearchOptions(
    metric: .cosine,
    normalizeQuery: true,
    metadataFilter: { vector in
        let category = vector.metadata?["category"]?.value as? String
        return category == "AI/ML"
    }
)

let filteredResults = try await db.search(
    query: queryVector,
    k: 10,
    options: searchOptions
)
```

## 🔧 Core Components

### Vector

The `Vector` struct represents a high-dimensional vector with optional metadata:

```swift
let vector = Vector(
    values: [0.1, 0.2, 0.3, 0.4],
    metadata: [
        "id": AnyCodable("doc_123"),
        "timestamp": AnyCodable(Date()),
        "category": AnyCodable("research")
    ]
)

// Vector operations
let magnitude = vector.magnitude
let normalized = try vector.normalized()
let similarity = try vector1.cosineSimilarity(with: vector2)
```

### VectorDatabase

The main interface for vector operations:

```swift
// CRUD operations
try await db.insert(vector)
try await db.insertBatch(vectors)
try await db.update(updatedVector)
try await db.delete(id: vectorId)

// Retrieval
let vector = try await db.get(id: vectorId)
let allVectors = try await db.getAll()
let count = try await db.count()

// Search operations
let results = try await db.search(query: queryVector, k: 10)
let thresholdResults = try await db.search(query: queryVector, threshold: 0.8)
let similarVectors = try await db.findSimilar(to: vectorId, k: 5)
```

### Distance Metrics

SwiftVS supports multiple distance metrics:

- **Cosine Similarity**: Measures the cosine of the angle between vectors
- **Euclidean Distance**: Measures the straight-line distance between vectors
- **Dot Product**: Measures the projection of one vector onto another

```swift
let cosineOptions = SearchOptions(metric: .cosine)
let euclideanOptions = SearchOptions(metric: .euclidean)
let dotProductOptions = SearchOptions(metric: .dotProduct)
```

## 📊 Performance Optimizations

### Accelerate Framework

SwiftVS automatically uses Apple's Accelerate framework when available for SIMD-accelerated operations:

```swift
// Check if Accelerate is available
if SwiftVS.isAccelerateAvailable() {
    // Use accelerated search
    let config = VectorDatabaseConfig(searchType: .accelerated)
    let db = try VectorDatabase(config: config)
}
```

### Batch Operations

Use batch operations for better performance:

```swift
// Batch insert
try await db.insertBatch(vectors)

// Batch search
let batchResults = try await db.batchSearch(queries: queryVectors, k: 10)
```

## 💾 Data Management

### Export/Import

SwiftVS supports JSON and CSV export formats:

```swift
// Export to JSON
try await db.exportToJSON(filePath: "vectors.json")

// Import from JSON
try await db.importFromJSON(filePath: "vectors.json")

// Export to CSV (values only)
try await db.exportToCSV(filePath: "vectors.csv")
```

### Database Statistics

Monitor your database with built-in statistics:

```swift
let stats = try await db.getStatistics()
print("Total vectors: \(stats.totalVectors)")
print("Unique dimensions: \(stats.uniqueDimensions)")
print("Average dimension: \(stats.averageDimension)")
```

## 🧪 Testing

Run the test suite:

```bash
swift test
```

Run the demo application:

```bash
swift run SwiftVSDemo
```

## 📈 Use Cases

### Document Similarity

```swift
// Store document embeddings
let documents = [
    ("AI Research Paper", [0.8, 0.2, 0.9, 0.1]),
    ("ML Tutorial", [0.7, 0.3, 0.8, 0.2]),
    ("Web Development Guide", [0.1, 0.9, 0.2, 0.8])
]

for (title, embedding) in documents {
    let vector = Vector(
        values: embedding,
        metadata: ["title": AnyCodable(title)]
    )
    try await db.insert(vector)
}

// Find similar documents
let queryDoc = Vector(values: [0.75, 0.25, 0.85, 0.15])
let similarDocs = try await db.search(query: queryDoc, k: 3)
```

### Recommendation Systems

```swift
// Store user preferences as vectors
let userVector = Vector(
    values: userPreferences,
    metadata: [
        "userId": AnyCodable(userId),
        "timestamp": AnyCodable(Date())
    ]
)

// Find similar users
let similarUsers = try await db.findSimilar(to: userVector.id, k: 10)
```

### Image Feature Matching

```swift
// Store image feature vectors
let imageVector = Vector(
    values: featureVector,
    metadata: [
        "imageId": AnyCodable(imageId),
        "fileName": AnyCodable(fileName),
        "tags": AnyCodable(tags)
    ]
)

// Search for similar images
let similarImages = try await db.search(
    query: queryImageVector,
    k: 20,
    options: SearchOptions(metric: .cosine)
)
```

## 🔮 Future Enhancements

- **Advanced Indexing**: HNSW, IVF, and other ANN algorithms
- **Distributed Storage**: Multi-node support for large datasets
- **Real-time Updates**: Streaming vector updates
- **Quantization**: Vector compression for memory efficiency
- **GPU Acceleration**: Metal Performance Shaders integration

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

SwiftVS is released under the MIT License. See [LICENSE](LICENSE) for details.

## 📞 Support

- **Documentation**: [GitHub Wiki](https://github.com/yourusername/SwiftVS/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/SwiftVS/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/SwiftVS/discussions)

---

Built with ❤️ for the Swift community 