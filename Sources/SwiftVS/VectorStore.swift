import Foundation
import SQLite

/// Errors that can occur during vector storage operations
public enum VectorStoreError: LocalizedError, Equatable {
    case databaseNotInitialized
    case vectorNotFound(UUID)
    case invalidVectorData
    case databaseError(String)
    case dimensionMismatch(expected: Int, actual: Int)
    
    public var errorDescription: String? {
        switch self {
        case .databaseNotInitialized:
            return "Database not initialized"
        case .vectorNotFound(let id):
            return "Vector with ID \(id) not found"
        case .invalidVectorData:
            return "Invalid vector data"
        case .databaseError(let message):
            return "Database error: \(message)"
        case .dimensionMismatch(let expected, let actual):
            return "Dimension mismatch: expected \(expected), got \(actual)"
        }
    }
}

/// SQLite-based vector storage implementation
public actor VectorStore {
    private var db: Connection?
    private let databasePath: String
    private let expectedDimension: Int?
    
    // Table definitions
    private let vectors = Table("vectors")
    private let id = Expression<String>("id")
    private let values = Expression<Data>("values")
    private let metadata = Expression<Data?>("metadata")
    private let dimension = Expression<Int>("dimension")
    private let createdAt = Expression<Date>("created_at")
    private let updatedAt = Expression<Date>("updated_at")
    
    /// Initialize the vector store
    /// - Parameters:
    ///   - databasePath: Path to the SQLite database file
    ///   - expectedDimension: Expected vector dimension (optional constraint)
    public init(databasePath: String, expectedDimension: Int? = nil) throws {
        self.databasePath = databasePath
        self.expectedDimension = expectedDimension
        
        // Create database directory if it doesn't exist
        let databaseURL = URL(fileURLWithPath: databasePath)
        let parentDirectory = databaseURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: parentDirectory,
            withIntermediateDirectories: true,
            attributes: nil
        )
        
        self.db = try Connection(databasePath)
        
        // Create tables
        try db?.run(vectors.create(ifNotExists: true) { t in
            t.column(id, primaryKey: true)
            t.column(values)
            t.column(metadata)
            t.column(dimension)
            t.column(createdAt)
            t.column(updatedAt)
        })
        
        // Create index on dimension for performance
        try db?.run("CREATE INDEX IF NOT EXISTS idx_dimension ON vectors(dimension)")
    }
    
    /// Insert a vector into the store
    /// - Parameter vector: The vector to insert
    /// - Throws: VectorStoreError if insertion fails
    public func insert(_ vector: Vector) async throws {
        guard let db = db else {
            throw VectorStoreError.databaseNotInitialized
        }
        
        // Validate dimension if constraint is set
        if let expectedDim = expectedDimension, vector.dimension != expectedDim {
            throw VectorStoreError.dimensionMismatch(expected: expectedDim, actual: vector.dimension)
        }
        
        do {
            let valuesData = try serializeFloatArray(vector.values)
            let metadataData = try vector.metadata.map { try serializeMetadata($0) }
            
            try db.run(vectors.insert(
                id <- vector.id.uuidString,
                values <- valuesData,
                metadata <- metadataData,
                dimension <- vector.dimension,
                createdAt <- vector.createdAt,
                updatedAt <- vector.updatedAt
            ))
        } catch {
            throw VectorStoreError.databaseError("Failed to insert vector: \(error.localizedDescription)")
        }
    }
    
    /// Insert multiple vectors into the store
    /// - Parameter vectors: The vectors to insert
    /// - Throws: VectorStoreError if insertion fails
    public func insertBatch(_ vectors: [Vector]) async throws {
        guard db != nil else {
            throw VectorStoreError.databaseNotInitialized
        }
        
        do {
            for vector in vectors {
                try await insert(vector)
            }
        } catch {
            throw VectorStoreError.databaseError("Failed to insert batch: \(error.localizedDescription)")
        }
    }
    
    /// Update a vector in the store
    /// - Parameter vector: The updated vector
    /// - Throws: VectorStoreError if update fails
    public func update(_ vector: Vector) async throws {
        guard let db = db else {
            throw VectorStoreError.databaseNotInitialized
        }
        
        // Validate dimension if constraint is set
        if let expectedDim = expectedDimension, vector.dimension != expectedDim {
            throw VectorStoreError.dimensionMismatch(expected: expectedDim, actual: vector.dimension)
        }
        
        do {
            let valuesData = try serializeFloatArray(vector.values)
            let metadataData = try vector.metadata.map { try serializeMetadata($0) }
            
            let vectorRow = vectors.filter(id == vector.id.uuidString)
            let changes = try db.run(vectorRow.update(
                values <- valuesData,
                metadata <- metadataData,
                dimension <- vector.dimension,
                updatedAt <- vector.updatedAt
            ))
            
            if changes == 0 {
                throw VectorStoreError.vectorNotFound(vector.id)
            }
        } catch let error as VectorStoreError {
            throw error
        } catch {
            throw VectorStoreError.databaseError("Failed to update vector: \(error.localizedDescription)")
        }
    }
    
    /// Delete a vector from the store
    /// - Parameter vectorId: The ID of the vector to delete
    /// - Throws: VectorStoreError if deletion fails
    public func delete(id vectorId: UUID) async throws {
        guard let db = db else {
            throw VectorStoreError.databaseNotInitialized
        }
        
        do {
            let vectorRow = vectors.filter(id == vectorId.uuidString)
            let changes = try db.run(vectorRow.delete())
            
            if changes == 0 {
                throw VectorStoreError.vectorNotFound(vectorId)
            }
        } catch let error as VectorStoreError {
            throw error
        } catch {
            throw VectorStoreError.databaseError("Failed to delete vector: \(error.localizedDescription)")
        }
    }
    
    /// Get a vector by ID
    /// - Parameter vectorId: The ID of the vector to retrieve
    /// - Returns: The vector if found
    /// - Throws: VectorStoreError if vector not found
    public func get(id vectorId: UUID) async throws -> Vector {
        guard let db = db else {
            throw VectorStoreError.databaseNotInitialized
        }
        
        do {
            let vectorRow = vectors.filter(id == vectorId.uuidString)
            guard let row = try db.pluck(vectorRow) else {
                throw VectorStoreError.vectorNotFound(vectorId)
            }
            
            return try deserializeVector(from: row)
        } catch let error as VectorStoreError {
            throw error
        } catch {
            throw VectorStoreError.databaseError("Failed to get vector: \(error.localizedDescription)")
        }
    }
    
    /// Get all vectors from the store
    /// - Returns: Array of all vectors
    /// - Throws: VectorStoreError if retrieval fails
    public func getAll() async throws -> [Vector] {
        guard let db = db else {
            throw VectorStoreError.databaseNotInitialized
        }
        
        do {
            var results: [Vector] = []
            for row in try db.prepare(vectors) {
                let vector = try deserializeVector(from: row)
                results.append(vector)
            }
            return results
        } catch {
            throw VectorStoreError.databaseError("Failed to get all vectors: \(error.localizedDescription)")
        }
    }
    
    /// Get vectors with a specific dimension
    /// - Parameter dimension: The dimension to filter by
    /// - Returns: Array of vectors with the specified dimension
    /// - Throws: VectorStoreError if retrieval fails
    public func getVectors(withDimension dimension: Int) async throws -> [Vector] {
        guard let db = db else {
            throw VectorStoreError.databaseNotInitialized
        }
        
        do {
            var results: [Vector] = []
            let query = vectors.filter(self.dimension == dimension)
            for row in try db.prepare(query) {
                let vector = try deserializeVector(from: row)
                results.append(vector)
            }
            return results
        } catch {
            throw VectorStoreError.databaseError("Failed to get vectors by dimension: \(error.localizedDescription)")
        }
    }
    
    /// Get vectors by a list of IDs
    /// - Parameter vectorIds: The IDs of vectors to retrieve
    /// - Returns: Array of found vectors
    /// - Throws: VectorStoreError if retrieval fails
    public func getVectors(withIds vectorIds: [UUID]) async throws -> [Vector] {
        guard let db = db else {
            throw VectorStoreError.databaseNotInitialized
        }
        
        do {
            var results: [Vector] = []
            let idStrings = vectorIds.map { $0.uuidString }
            let query = vectors.filter(idStrings.contains(id))
            
            for row in try db.prepare(query) {
                let vector = try deserializeVector(from: row)
                results.append(vector)
            }
            return results
        } catch {
            throw VectorStoreError.databaseError("Failed to get vectors by IDs: \(error.localizedDescription)")
        }
    }
    
    /// Get the count of vectors in the store
    /// - Returns: Number of vectors
    /// - Throws: VectorStoreError if count fails
    public func count() async throws -> Int {
        guard let db = db else {
            throw VectorStoreError.databaseNotInitialized
        }
        
        do {
            return try db.scalar(vectors.count)
        } catch {
            throw VectorStoreError.databaseError("Failed to count vectors: \(error.localizedDescription)")
        }
    }
    
    /// Clear all vectors from the store
    /// - Throws: VectorStoreError if clearing fails
    public func clear() async throws {
        guard let db = db else {
            throw VectorStoreError.databaseNotInitialized
        }
        
        do {
            try db.run(vectors.delete())
        } catch {
            throw VectorStoreError.databaseError("Failed to clear vectors: \(error.localizedDescription)")
        }
    }
    
    /// Check if a vector exists in the store
    /// - Parameter vectorId: The ID of the vector to check
    /// - Returns: True if vector exists, false otherwise
    /// - Throws: VectorStoreError if check fails
    public func exists(id vectorId: UUID) async throws -> Bool {
        guard let db = db else {
            throw VectorStoreError.databaseNotInitialized
        }
        
        do {
            let vectorRow = vectors.filter(id == vectorId.uuidString)
            return try db.pluck(vectorRow) != nil
        } catch {
            throw VectorStoreError.databaseError("Failed to check vector existence: \(error.localizedDescription)")
        }
    }
    
    /// Get vectors with pagination
    /// - Parameters:
    ///   - offset: Number of vectors to skip
    ///   - limit: Maximum number of vectors to return
    /// - Returns: Array of vectors
    /// - Throws: VectorStoreError if retrieval fails
    public func getVectors(offset: Int, limit: Int) async throws -> [Vector] {
        guard let db = db else {
            throw VectorStoreError.databaseNotInitialized
        }
        
        do {
            var results: [Vector] = []
            let query = vectors.limit(limit, offset: offset)
            for row in try db.prepare(query) {
                let vector = try deserializeVector(from: row)
                results.append(vector)
            }
            return results
        } catch {
            throw VectorStoreError.databaseError("Failed to get vectors with pagination: \(error.localizedDescription)")
        }
    }
    
    // MARK: - Private Helper Methods
    
    private func serializeFloatArray(_ array: [Float]) throws -> Data {
        return array.withUnsafeBytes { bytes in
            Data(bytes)
        }
    }
    
    private func deserializeFloatArray(_ data: Data) throws -> [Float] {
        return data.withUnsafeBytes { bytes in
            let floatBuffer = bytes.bindMemory(to: Float.self)
            return Array(floatBuffer)
        }
    }
    
    private func serializeMetadata(_ metadata: [String: AnyCodable]) throws -> Data {
        return try JSONEncoder().encode(metadata)
    }
    
    private func deserializeMetadata(_ data: Data) throws -> [String: AnyCodable] {
        return try JSONDecoder().decode([String: AnyCodable].self, from: data)
    }
    
    private func deserializeVector(from row: Row) throws -> Vector {
        let vectorId = UUID(uuidString: row[id])!
        let vectorValues = try deserializeFloatArray(row[values])
        let vectorMetadata = try row[metadata].map { try deserializeMetadata($0) }
        let vectorCreatedAt = row[createdAt]
        let vectorUpdatedAt = row[updatedAt]
        
        return Vector(
            id: vectorId,
            values: vectorValues,
            metadata: vectorMetadata,
            createdAt: vectorCreatedAt,
            updatedAt: vectorUpdatedAt
        )
    }
}

// MARK: - Export/Import Support

public extension VectorStore {
    
    /// Export vectors to a JSON file
    /// - Parameter filePath: Path to save the JSON file
    /// - Throws: VectorStoreError if export fails
    func exportToJSON(filePath: String) async throws {
        let vectors = try await getAll()
        let data = try JSONEncoder().encode(vectors)
        try data.write(to: URL(fileURLWithPath: filePath))
    }
    
    /// Import vectors from a JSON file
    /// - Parameter filePath: Path to the JSON file
    /// - Throws: VectorStoreError if import fails
    func importFromJSON(filePath: String) async throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: filePath))
        let vectors = try JSONDecoder().decode([Vector].self, from: data)
        try await insertBatch(vectors)
    }
    
    /// Export vectors to a CSV file (values only, no metadata)
    /// - Parameter filePath: Path to save the CSV file
    /// - Throws: VectorStoreError if export fails
    func exportToCSV(filePath: String) async throws {
        let vectors = try await getAll()
        var csvContent = "id,dimension,values\n"
        
        for vector in vectors {
            let valuesString = vector.values.map { "\($0)" }.joined(separator: ";")
            csvContent += "\(vector.id.uuidString),\(vector.dimension),\"\(valuesString)\"\n"
        }
        
        try csvContent.write(to: URL(fileURLWithPath: filePath), atomically: true, encoding: .utf8)
    }
} 