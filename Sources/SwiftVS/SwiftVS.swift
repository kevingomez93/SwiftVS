// SwiftVS - A Swift-native vector database framework
// Copyright (c) 2024 SwiftVS Contributors
// Licensed under the MIT License

import Foundation

/// SwiftVS is a Swift-native vector database framework that provides
/// efficient storage, indexing, and similarity search for high-dimensional vectors.
///
/// ## Key Features
/// - Store high-dimensional vectors with optional metadata
/// - Efficient similarity search using cosine similarity, Euclidean distance, or dot product
/// - Persistent storage using SQLite
/// - Support for both brute-force and accelerated search algorithms
/// - Thread-safe operations using Swift's actor model
/// - Import/export functionality for data migration
/// - Comprehensive error handling and validation
///
/// ## Basic Usage
/// ```swift
/// // Create a vector database
/// let db = try VectorDatabase.create(at: "vectors.db")
/// 
/// // Create and insert vectors
/// let vector1 = Vector(values: [0.1, 0.2, 0.3, 0.4])
/// let vector2 = Vector(values: [0.5, 0.6, 0.7, 0.8])
/// 
/// try await db.insert(vector1)
/// try await db.insert(vector2)
/// 
/// // Perform similarity search
/// let results = try await db.search(query: vector1, k: 5)
/// ```
///
/// ## Advanced Usage
/// ```swift
/// // Configure the database with specific options
/// let config = VectorDatabaseConfig(
///     databasePath: "vectors.db",
///     expectedDimension: 128,
///     searchType: .accelerated,
///     useAccelerate: true
/// )
/// let db = try VectorDatabase(config: config)
/// 
/// // Search with custom options
/// let searchOptions = SearchOptions(
///     metric: .cosine,
///     normalizeQuery: true,
///     normalizeStored: true
/// )
/// let results = try await db.search(
///     query: queryVector,
///     k: 10,
///     options: searchOptions
/// )
/// ```
public struct SwiftVS {
    public static let version = "1.0.0"
    public static let description = "A Swift-native vector database framework"
    
    /// Creates a simple vector database with default configuration
    /// - Parameter databasePath: Path to the SQLite database file
    /// - Returns: A configured vector database
    /// - Throws: VectorStoreError if initialization fails
    public static func createDatabase(at databasePath: String) async throws -> VectorDatabase {
        return try await VectorDatabase.create(at: databasePath)
    }
    
    /// Creates an in-memory vector database for testing and prototyping
    /// - Returns: A configured in-memory vector database
    /// - Throws: VectorStoreError if initialization fails
    public static func createInMemoryDatabase() async throws -> VectorDatabase {
        return try await VectorDatabase.createInMemory()
    }
    
    /// Convenience method to create a vector from an array of values
    /// - Parameters:
    ///   - values: The vector values
    ///   - metadata: Optional metadata dictionary
    /// - Returns: A new Vector instance
    public static func createVector(
        values: [Float],
        metadata: [String: Any]? = nil
    ) -> Vector {
        let convertedMetadata = metadata?.compactMapValues { value in
            if let codableValue = value as? any Codable {
                return AnyCodable(codableValue)
            }
            return nil
        }
        return Vector(values: values, metadata: convertedMetadata)
    }
    
    /// Convenience method to create search options
    /// - Parameters:
    ///   - metric: The distance metric to use
    ///   - normalizeQuery: Whether to normalize the query vector
    ///   - normalizeStored: Whether to normalize stored vectors
    /// - Returns: A SearchOptions instance
    public static func createSearchOptions(
        metric: DistanceMetric = .cosine,
        normalizeQuery: Bool = false,
        normalizeStored: Bool = false
    ) -> SearchOptions {
        return SearchOptions(
            metric: metric,
            normalizeQuery: normalizeQuery,
            normalizeStored: normalizeStored
        )
    }
}

// MARK: - Public API
// All types are already public and available directly
// No need for re-exports since they're in the same module

// MARK: - Extension for Backwards Compatibility

extension SwiftVS {
    
    /// Legacy method for creating vectors (deprecated)
    @available(*, deprecated, message: "Use createVector(values:metadata:) instead")
    public static func vector(from values: [Float]) -> Vector {
        return Vector(values: values)
    }
    
    /// Legacy method for creating databases (deprecated)
    @available(*, deprecated, message: "Use createDatabase(at:) instead")
    public static func database(at path: String) async throws -> VectorDatabase {
        return try await VectorDatabase.create(at: path)
    }
}

// MARK: - Framework Information

extension SwiftVS {
    
    /// Get detailed information about the SwiftVS framework
    /// - Returns: Framework information
    public static func getFrameworkInfo() -> FrameworkInfo {
        return FrameworkInfo(
            name: "SwiftVS",
            version: version,
            description: description,
            supportedPlatforms: [.iOS, .macOS, .tvOS, .watchOS],
            features: [
                "High-dimensional vector storage",
                "Similarity search algorithms",
                "SQLite persistence",
                "Metadata support",
                "Accelerate framework integration",
                "Thread-safe operations",
                "Import/export functionality"
            ]
        )
    }
    
    /// Check if Accelerate framework is available
    /// - Returns: True if Accelerate is available, false otherwise
    public static func isAccelerateAvailable() -> Bool {
        #if canImport(Accelerate)
        return true
        #else
        return false
        #endif
    }
}

/// Framework information structure
public struct FrameworkInfo {
    public let name: String
    public let version: String
    public let description: String
    public let supportedPlatforms: [Platform]
    public let features: [String]
    
    public enum Platform: String, CaseIterable {
        case iOS = "iOS"
        case macOS = "macOS"
        case tvOS = "tvOS"
        case watchOS = "watchOS"
    }
} 