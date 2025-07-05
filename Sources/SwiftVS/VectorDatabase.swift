import Foundation

/// Configuration options for the vector database
public struct VectorDatabaseConfig {
    /// Path to the SQLite database file
    public let databasePath: String
    
    /// Expected vector dimension (optional constraint)
    public let expectedDimension: Int?
    
    /// Type of similarity search to use
    public let searchType: SearchType
    
    /// Whether to enable high-performance operations using Accelerate
    public let useAccelerate: Bool
    
    /// Default search options
    public let defaultSearchOptions: SearchOptions
    
    public enum SearchType {
        case bruteForce
        case accelerated
    }
    
    public init(
        databasePath: String,
        expectedDimension: Int? = nil,
        searchType: SearchType = .bruteForce,
        useAccelerate: Bool = true,
        defaultSearchOptions: SearchOptions = SearchOptions()
    ) {
        self.databasePath = databasePath
        self.expectedDimension = expectedDimension
        self.searchType = searchType
        self.useAccelerate = useAccelerate
        self.defaultSearchOptions = defaultSearchOptions
    }
}

/// Main vector database class that provides a unified interface for vector operations
public actor VectorDatabase {
    private let store: VectorStore
    private let searchEngine: any SimilaritySearchable
    private let config: VectorDatabaseConfig
    private var isInitialized = false
    
    /// Initialize the vector database
    /// - Parameter config: Configuration options
    /// - Throws: VectorStoreError if initialization fails
    public init(config: VectorDatabaseConfig) async throws {
        self.config = config
        self.store = try VectorStore(
            databasePath: config.databasePath,
            expectedDimension: config.expectedDimension
        )
        
        // Choose search engine based on configuration
        #if canImport(Accelerate)
        if config.useAccelerate && config.searchType == .accelerated {
            self.searchEngine = AcceleratedSimilaritySearch()
        } else {
            self.searchEngine = BruteForceSimilaritySearch()
        }
        #else
        self.searchEngine = BruteForceSimilaritySearch()
        #endif
        
        // Initialize the search engine with existing vectors
        try await loadVectorsIntoSearchEngine()
        self.isInitialized = true
    }
    
    /// Load all vectors from persistent storage into the search engine
    private func loadVectorsIntoSearchEngine() async throws {
        let vectors = try await store.getAll()
        
        if let bruteForceSearch = searchEngine as? BruteForceSimilaritySearch {
            await bruteForceSearch.addVectors(vectors)
        }
        
        #if canImport(Accelerate)
        if let acceleratedSearch = searchEngine as? AcceleratedSimilaritySearch {
            await acceleratedSearch.addVectors(vectors)
        }
        #endif
    }
    
    // MARK: - Vector CRUD Operations
    
    /// Insert a vector into the database
    /// - Parameter vector: The vector to insert
    /// - Throws: VectorStoreError if insertion fails
    public func insert(_ vector: Vector) async throws {
        try await store.insert(vector)
        
        // Update search engine
        if let bruteForceSearch = searchEngine as? BruteForceSimilaritySearch {
            await bruteForceSearch.addVectors([vector])
        }
        
        #if canImport(Accelerate)
        if let acceleratedSearch = searchEngine as? AcceleratedSimilaritySearch {
            await acceleratedSearch.addVectors([vector])
        }
        #endif
    }
    
    /// Insert multiple vectors into the database
    /// - Parameter vectors: The vectors to insert
    /// - Throws: VectorStoreError if insertion fails
    public func insertBatch(_ vectors: [Vector]) async throws {
        try await store.insertBatch(vectors)
        
        // Update search engine
        if let bruteForceSearch = searchEngine as? BruteForceSimilaritySearch {
            await bruteForceSearch.addVectors(vectors)
        }
        
        #if canImport(Accelerate)
        if let acceleratedSearch = searchEngine as? AcceleratedSimilaritySearch {
            await acceleratedSearch.addVectors(vectors)
        }
        #endif
    }
    
    /// Update a vector in the database
    /// - Parameter vector: The updated vector
    /// - Throws: VectorStoreError if update fails
    public func update(_ vector: Vector) async throws {
        try await store.update(vector)
        
        // Update search engine
        if let bruteForceSearch = searchEngine as? BruteForceSimilaritySearch {
            await bruteForceSearch.updateVectors([vector])
        }
        
        #if canImport(Accelerate)
        if let acceleratedSearch = searchEngine as? AcceleratedSimilaritySearch {
            await acceleratedSearch.updateVectors([vector])
        }
        #endif
    }
    
    /// Delete a vector from the database
    /// - Parameter vectorId: The ID of the vector to delete
    /// - Throws: VectorStoreError if deletion fails
    public func delete(id vectorId: UUID) async throws {
        try await store.delete(id: vectorId)
        
        // Update search engine
        if let bruteForceSearch = searchEngine as? BruteForceSimilaritySearch {
            await bruteForceSearch.removeVectors(withIds: [vectorId])
        }
        
        #if canImport(Accelerate)
        if let acceleratedSearch = searchEngine as? AcceleratedSimilaritySearch {
            await acceleratedSearch.removeVectors(withIds: [vectorId])
        }
        #endif
    }
    
    /// Get a vector by ID
    /// - Parameter vectorId: The ID of the vector to retrieve
    /// - Returns: The vector if found
    /// - Throws: VectorStoreError if vector not found
    public func get(id vectorId: UUID) async throws -> Vector {
        return try await store.get(id: vectorId)
    }
    
    /// Get all vectors from the database
    /// - Returns: Array of all vectors
    /// - Throws: VectorStoreError if retrieval fails
    public func getAll() async throws -> [Vector] {
        return try await store.getAll()
    }
    
    /// Get vectors with a specific dimension
    /// - Parameter dimension: The dimension to filter by
    /// - Returns: Array of vectors with the specified dimension
    /// - Throws: VectorStoreError if retrieval fails
    public func getVectors(withDimension dimension: Int) async throws -> [Vector] {
        return try await store.getVectors(withDimension: dimension)
    }
    
    /// Get vectors by a list of IDs
    /// - Parameter vectorIds: The IDs of vectors to retrieve
    /// - Returns: Array of found vectors
    /// - Throws: VectorStoreError if retrieval fails
    public func getVectors(withIds vectorIds: [UUID]) async throws -> [Vector] {
        return try await store.getVectors(withIds: vectorIds)
    }
    
    /// Get the count of vectors in the database
    /// - Returns: Number of vectors
    /// - Throws: VectorStoreError if count fails
    public func count() async throws -> Int {
        return try await store.count()
    }
    
    /// Check if a vector exists in the database
    /// - Parameter vectorId: The ID of the vector to check
    /// - Returns: True if vector exists, false otherwise
    /// - Throws: VectorStoreError if check fails
    public func exists(id vectorId: UUID) async throws -> Bool {
        return try await store.exists(id: vectorId)
    }
    
    /// Get vectors with pagination
    /// - Parameters:
    ///   - offset: Number of vectors to skip
    ///   - limit: Maximum number of vectors to return
    /// - Returns: Array of vectors
    /// - Throws: VectorStoreError if retrieval fails
    public func getVectors(offset: Int, limit: Int) async throws -> [Vector] {
        return try await store.getVectors(offset: offset, limit: limit)
    }
    
    /// Clear all vectors from the database
    /// - Throws: VectorStoreError if clearing fails
    public func clear() async throws {
        try await store.clear()
        
        // Clear search engine
        if let bruteForceSearch = searchEngine as? BruteForceSimilaritySearch {
            await bruteForceSearch.clear()
        }
        
        #if canImport(Accelerate)
        if let acceleratedSearch = searchEngine as? AcceleratedSimilaritySearch {
            await acceleratedSearch.clear()
        }
        #endif
    }
    
    // MARK: - Similarity Search Operations
    
    /// Perform a K-nearest neighbor search
    /// - Parameters:
    ///   - query: The query vector
    ///   - k: Number of nearest neighbors to return
    ///   - options: Search configuration options (uses default if not provided)
    /// - Returns: Array of search results sorted by similarity (best first)
    /// - Throws: VectorError if search fails
    public func search(
        query: Vector,
        k: Int,
        options: SearchOptions? = nil
    ) async throws -> [SimilaritySearchResult] {
        let searchOptions = options ?? config.defaultSearchOptions
        return try await searchEngine.search(query: query, k: k, options: searchOptions)
    }
    
    /// Perform a similarity search with a minimum similarity threshold
    /// - Parameters:
    ///   - query: The query vector
    ///   - threshold: Minimum similarity threshold
    ///   - maxResults: Maximum number of results to return
    ///   - options: Search configuration options (uses default if not provided)
    /// - Returns: Array of search results above the threshold
    /// - Throws: VectorError if search fails
    public func search(
        query: Vector,
        threshold: Float,
        maxResults: Int = Int.max,
        options: SearchOptions? = nil
    ) async throws -> [SimilaritySearchResult] {
        let searchOptions = options ?? config.defaultSearchOptions
        return try await searchEngine.search(
            query: query,
            threshold: threshold,
            maxResults: maxResults,
            options: searchOptions
        )
    }
    
    /// Perform a search using raw vector values
    /// - Parameters:
    ///   - queryValues: The query vector values
    ///   - k: Number of nearest neighbors to return
    ///   - options: Search configuration options (uses default if not provided)
    /// - Returns: Array of search results sorted by similarity (best first)
    /// - Throws: VectorError if search fails
    public func search(
        queryValues: [Float],
        k: Int,
        options: SearchOptions? = nil
    ) async throws -> [SimilaritySearchResult] {
        let queryVector = Vector(values: queryValues)
        return try await search(query: queryVector, k: k, options: options)
    }
    
    /// Perform a search using raw vector values with threshold
    /// - Parameters:
    ///   - queryValues: The query vector values
    ///   - threshold: Minimum similarity threshold
    ///   - maxResults: Maximum number of results to return
    ///   - options: Search configuration options (uses default if not provided)
    /// - Returns: Array of search results above the threshold
    /// - Throws: VectorError if search fails
    public func search(
        queryValues: [Float],
        threshold: Float,
        maxResults: Int = Int.max,
        options: SearchOptions? = nil
    ) async throws -> [SimilaritySearchResult] {
        let queryVector = Vector(values: queryValues)
        return try await search(query: queryVector, threshold: threshold, maxResults: maxResults, options: options)
    }
    
    // MARK: - Advanced Search Operations
    
    /// Find vectors similar to a given vector ID
    /// - Parameters:
    ///   - vectorId: ID of the vector to find similar vectors for
    ///   - k: Number of similar vectors to return (excluding the original vector)
    ///   - options: Search configuration options (uses default if not provided)
    /// - Returns: Array of similar vectors
    /// - Throws: VectorError if search fails
    public func findSimilar(
        to vectorId: UUID,
        k: Int,
        options: SearchOptions? = nil
    ) async throws -> [SimilaritySearchResult] {
        let queryVector = try await get(id: vectorId)
        let results = try await search(query: queryVector, k: k + 1, options: options)
        
        // Filter out the original vector from results
        return results.filter { $0.vector.id != vectorId }
    }
    
    /// Perform a batch search with multiple query vectors
    /// - Parameters:
    ///   - queries: Array of query vectors
    ///   - k: Number of nearest neighbors to return for each query
    ///   - options: Search configuration options (uses default if not provided)
    /// - Returns: Array of search results for each query
    /// - Throws: VectorError if search fails
    public func batchSearch(
        queries: [Vector],
        k: Int,
        options: SearchOptions? = nil
    ) async throws -> [[SimilaritySearchResult]] {
        var results: [[SimilaritySearchResult]] = []
        
        for query in queries {
            let queryResults = try await search(query: query, k: k, options: options)
            results.append(queryResults)
        }
        
        return results
    }
    
    // MARK: - Import/Export Operations
    
    /// Export all vectors to a JSON file
    /// - Parameter filePath: Path to save the JSON file
    /// - Throws: VectorStoreError if export fails
    public func exportToJSON(filePath: String) async throws {
        try await store.exportToJSON(filePath: filePath)
    }
    
    /// Import vectors from a JSON file
    /// - Parameter filePath: Path to the JSON file
    /// - Throws: VectorStoreError if import fails
    public func importFromJSON(filePath: String) async throws {
        try await store.importFromJSON(filePath: filePath)
        
        // Reload search engine after import
        try await loadVectorsIntoSearchEngine()
    }
    
    /// Export all vectors to a CSV file
    /// - Parameter filePath: Path to save the CSV file
    /// - Throws: VectorStoreError if export fails
    public func exportToCSV(filePath: String) async throws {
        try await store.exportToCSV(filePath: filePath)
    }
    
    // MARK: - Database Information
    
    /// Get database configuration
    /// - Returns: The current configuration
    public func getConfig() -> VectorDatabaseConfig {
        return config
    }
    
    /// Get database statistics
    /// - Returns: Database statistics
    /// - Throws: VectorStoreError if retrieval fails
    public func getStatistics() async throws -> DatabaseStatistics {
        let totalVectors = try await store.count()
        let dimensions = try await store.getAll().map { $0.dimension }
        let uniqueDimensions = Set(dimensions).count
        let averageDimension = dimensions.isEmpty ? 0 : dimensions.reduce(0, +) / dimensions.count
        
        return DatabaseStatistics(
            totalVectors: totalVectors,
            uniqueDimensions: uniqueDimensions,
            averageDimension: averageDimension,
            searchEngineType: config.searchType,
            databasePath: config.databasePath
        )
    }
    
    /// Check if the database is properly initialized
    /// - Returns: True if initialized, false otherwise
    public func isReady() -> Bool {
        return isInitialized
    }
}

/// Database statistics structure
public struct DatabaseStatistics {
    /// Total number of vectors in the database
    public let totalVectors: Int
    
    /// Number of unique dimensions in the database
    public let uniqueDimensions: Int
    
    /// Average dimension of vectors in the database
    public let averageDimension: Int
    
    /// Type of search engine being used
    public let searchEngineType: VectorDatabaseConfig.SearchType
    
    /// Path to the database file
    public let databasePath: String
    
    public init(
        totalVectors: Int,
        uniqueDimensions: Int,
        averageDimension: Int,
        searchEngineType: VectorDatabaseConfig.SearchType,
        databasePath: String
    ) {
        self.totalVectors = totalVectors
        self.uniqueDimensions = uniqueDimensions
        self.averageDimension = averageDimension
        self.searchEngineType = searchEngineType
        self.databasePath = databasePath
    }
}

// MARK: - Convenience Extensions

public extension VectorDatabase {
    /// Create a vector database with a simple configuration
    /// - Parameters:
    ///   - databasePath: Path to the SQLite database file
    ///   - expectedDimension: Expected vector dimension (optional)
    /// - Returns: A configured vector database
    /// - Throws: VectorStoreError if initialization fails
    static func create(
        at databasePath: String,
        expectedDimension: Int? = nil
    ) async throws -> VectorDatabase {
        let config = VectorDatabaseConfig(
            databasePath: databasePath,
            expectedDimension: expectedDimension
        )
        return try await VectorDatabase(config: config)
    }
    
    /// Create an in-memory vector database for testing
    /// - Returns: A configured in-memory vector database
    /// - Throws: VectorStoreError if initialization fails
    static func createInMemory() async throws -> VectorDatabase {
        let config = VectorDatabaseConfig(databasePath: ":memory:")
        return try await VectorDatabase(config: config)
    }
} 