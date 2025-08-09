//
//  SwiftVSTests.swift
//  SwiftVSTests
//
//  Created by Kevin Gomez on 04/07/2025.
//

import Testing
import Foundation
@testable import SwiftVS

struct SwiftVSTests {
    
    // MARK: - Vector Tests
    
    @Test func testVectorInitialization() async throws {
        let values: [Float] = [1.0, 2.0, 3.0, 4.0]
        let metadata = ["category": AnyCodable("test"), "id": AnyCodable(123)]
        
        let vector = Vector(values: values, metadata: metadata)
        
        #expect(vector.values == values)
        #expect(vector.dimension == 4)
        #expect(vector.metadata?["category"]?.value as? String == "test")
        #expect(vector.metadata?["id"]?.value as? Int == 123)
    }
    
    @Test func testVectorMagnitude() async throws {
        let vector = Vector(values: [3.0, 4.0])
        #expect(vector.magnitude == 5.0)
    }
    
    @Test func testVectorDotProduct() async throws {
        let vector1 = Vector(values: [1.0, 2.0, 3.0])
        let vector2 = Vector(values: [4.0, 5.0, 6.0])
        
        let dotProduct = try vector1.dotProduct(with: vector2)
        #expect(dotProduct == 32.0) // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }
    
    @Test func testVectorCosineSimilarity() async throws {
        let vector1 = Vector(values: [1.0, 0.0])
        let vector2 = Vector(values: [1.0, 0.0])
        
        let similarity = try vector1.cosineSimilarity(with: vector2)
        #expect(similarity == 1.0) // Identical vectors should have similarity 1.0
    }
    
    @Test func testVectorEuclideanDistance() async throws {
        let vector1 = Vector(values: [0.0, 0.0])
        let vector2 = Vector(values: [3.0, 4.0])
        
        let distance = try vector1.euclideanDistance(to: vector2)
        #expect(distance == 5.0) // 3-4-5 triangle
    }
    
    @Test func testVectorNormalization() async throws {
        let vector = Vector(values: [3.0, 4.0])
        let normalized = try vector.normalized()
        
        #expect(abs(normalized.magnitude - 1.0) < 0.001) // Should be unit vector
        #expect(abs(normalized.values[0] - 0.6) < 0.001) // 3/5 = 0.6
        #expect(abs(normalized.values[1] - 0.8) < 0.001) // 4/5 = 0.8
    }
    
    @Test func testVectorDimensionMismatch() async throws {
        let vector1 = Vector(values: [1.0, 2.0])
        let vector2 = Vector(values: [1.0, 2.0, 3.0])
        
        #expect(throws: VectorError.dimensionMismatch(2, 3)) {
            try vector1.dotProduct(with: vector2)
        }
    }
    
    // MARK: - VectorStore Tests
    
    @Test func testVectorStoreInitialization() async throws {
        let tempPath = NSTemporaryDirectory() + "test_vectors_\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempPath) }
        
        let store = try VectorStore(databasePath: tempPath)
        let count = try await store.count()
        #expect(count == 0)
    }
    
    @Test func testVectorStoreInsertAndRetrieve() async throws {
        let tempPath = NSTemporaryDirectory() + "test_vectors_\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempPath) }
        
        let store = try VectorStore(databasePath: tempPath)
        let vector = Vector(values: [1.0, 2.0, 3.0])
        
        try await store.insert(vector)
        let retrieved = try await store.get(id: vector.id)
        
        #expect(retrieved.id == vector.id)
        #expect(retrieved.values == vector.values)
        #expect(retrieved.dimension == vector.dimension)
    }
    
    @Test func testVectorStoreUpdate() async throws {
        let tempPath = NSTemporaryDirectory() + "test_vectors_\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempPath) }
        
        let store = try VectorStore(databasePath: tempPath)
        let vector = Vector(values: [1.0, 2.0, 3.0])
        
        try await store.insert(vector)
        
        let updatedVector = vector.updated(values: [4.0, 5.0, 6.0])
        try await store.update(updatedVector)
        
        let retrieved = try await store.get(id: vector.id)
        #expect(retrieved.values == [4.0, 5.0, 6.0])
    }
    
    @Test func testVectorStoreDelete() async throws {
        let tempPath = NSTemporaryDirectory() + "test_vectors_\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempPath) }
        
        let store = try VectorStore(databasePath: tempPath)
        let vector = Vector(values: [1.0, 2.0, 3.0])
        
        try await store.insert(vector)
        #expect(try await store.exists(id: vector.id))
        
        try await store.delete(id: vector.id)
        #expect(try await store.exists(id: vector.id) == false)
    }
    
    @Test func testVectorStoreBatch() async throws {
        let tempPath = NSTemporaryDirectory() + "test_vectors_\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: tempPath) }
        
        let store = try VectorStore(databasePath: tempPath)
        let vectors = [
            Vector(values: [1.0, 2.0]),
            Vector(values: [3.0, 4.0]),
            Vector(values: [5.0, 6.0])
        ]
        
        try await store.insertBatch(vectors)
        
        let count = try await store.count()
        #expect(count == 3)
        
        let allVectors = try await store.getAll()
        #expect(allVectors.count == 3)
    }
    
    // MARK: - Similarity Search Tests
    
    @Test func testBruteForceSearch() async throws {
        let search = BruteForceSimilaritySearch()
        
        let vectors = [
            Vector(values: [1.0, 0.0]),
            Vector(values: [0.0, 1.0]),
            Vector(values: [1.0, 1.0])
        ]
        
        await search.addVectors(vectors)
        
        let queryVector = Vector(values: [1.0, 0.0])
        let results = try await search.search(query: queryVector, k: 2)
        
        #expect(results.count == 2)
        #expect(results[0].vector.values == [1.0, 0.0]) // Should find exact match first
        #expect(results[0].score == 1.0) // Perfect cosine similarity
    }
    
    @Test func testSearchWithThreshold() async throws {
        let search = BruteForceSimilaritySearch()
        
        let vectors = [
            Vector(values: [1.0, 0.0]),
            Vector(values: [0.0, 1.0]),
            Vector(values: [1.0, 1.0])
        ]
        
        await search.addVectors(vectors)
        
        let queryVector = Vector(values: [1.0, 0.0])
        let results = try await search.search(query: queryVector, threshold: 0.5, maxResults: 10)
        
        #expect(results.count >= 1) // Should find at least the perfect match
    }
    
    @Test func testSearchWithMetadataFilter() async throws {
        let search = BruteForceSimilaritySearch()
        
        let vectors = [
            Vector(values: [1.0, 0.0], metadata: ["category": AnyCodable("A")]),
            Vector(values: [0.0, 1.0], metadata: ["category": AnyCodable("B")]),
            Vector(values: [1.0, 1.0], metadata: ["category": AnyCodable("A")])
        ]
        
        await search.addVectors(vectors)
        
        let searchOptions = SearchOptions(
            metadataFilter: { vector in
                vector.metadata?["category"]?.value as? String == "A"
            }
        )
        
        let queryVector = Vector(values: [1.0, 0.0])
        let results = try await search.search(query: queryVector, k: 10, options: searchOptions)
        
        #expect(results.count == 2) // Should only find vectors with category "A"
        for result in results {
            #expect(result.vector.metadata?["category"]?.value as? String == "A")
        }
    }
    
    @Test func testSearchWithDifferentMetrics() async throws {
        let search = BruteForceSimilaritySearch()
        
        let vectors = [
            Vector(values: [1.0, 0.0]),
            Vector(values: [0.0, 1.0])
        ]
        
        await search.addVectors(vectors)
        
        let queryVector = Vector(values: [1.0, 0.0])
        
        // Test cosine similarity
        let cosineResults = try await search.search(
            query: queryVector,
            k: 2,
            options: SearchOptions(metric: .cosine)
        )
        
        // Test Euclidean distance
        let euclideanResults = try await search.search(
            query: queryVector,
            k: 2,
            options: SearchOptions(metric: .euclidean)
        )
        
        // Test dot product
        let dotProductResults = try await search.search(
            query: queryVector,
            k: 2,
            options: SearchOptions(metric: .dotProduct)
        )
        
        #expect(cosineResults.count == 2)
        #expect(euclideanResults.count == 2)
        #expect(dotProductResults.count == 2)
    }
    
    // MARK: - VectorDatabase Integration Tests
    
    @Test func testVectorDatabaseBasicOperations() async throws {
        let db = try await VectorDatabase.createInMemory()
        
        let vector1 = Vector(values: [1.0, 2.0, 3.0])
        let vector2 = Vector(values: [4.0, 5.0, 6.0])
        
        try await db.insert(vector1)
        try await db.insert(vector2)
        
        let count = try await db.count()
        #expect(count == 2)
        
        let retrieved = try await db.get(id: vector1.id)
        #expect(retrieved.values == vector1.values)
        
        let searchResults = try await db.search(query: vector1, k: 2)
        #expect(searchResults.count == 2)
    }
    
    @Test func testVectorDatabaseSearchWithOptions() async throws {
        let config = VectorDatabaseConfig(
            databasePath: ":memory:",
            searchType: .bruteForce,
            defaultSearchOptions: SearchOptions(metric: .cosine)
        )
        let db = try await VectorDatabase(config: config)
        
        let vectors = [
            Vector(values: [1.0, 0.0]),
            Vector(values: [0.0, 1.0]),
            Vector(values: [1.0, 1.0])
        ]
        
        try await db.insertBatch(vectors)
        
        let queryVector = Vector(values: [1.0, 0.0])
        let results = try await db.search(query: queryVector, k: 2)
        
        #expect(results.count == 2)
        #expect(results[0].score >= results[1].score) // Should be sorted by similarity
    }
    
    @Test func testVectorDatabaseFindSimilar() async throws {
        let db = try await VectorDatabase.createInMemory()
        
        let vector1 = Vector(values: [1.0, 0.0])
        let vector2 = Vector(values: [0.8, 0.2])
        let vector3 = Vector(values: [0.0, 1.0])
        
        try await db.insertBatch([vector1, vector2, vector3])
        
        let similarVectors = try await db.findSimilar(to: vector1.id, k: 2)
        
        #expect(similarVectors.count == 2)
        // Should not include the original vector
        #expect(similarVectors.allSatisfy { $0.vector.id != vector1.id })
    }
    
    @Test func testVectorDatabaseBatchSearch() async throws {
        let db = try await VectorDatabase.createInMemory()
        
        let storedVectors = [
            Vector(values: [1.0, 0.0]),
            Vector(values: [0.0, 1.0]),
            Vector(values: [1.0, 1.0])
        ]
        
        try await db.insertBatch(storedVectors)
        
        let queryVectors = [
            Vector(values: [1.0, 0.0]),
            Vector(values: [0.0, 1.0])
        ]
        
        let batchResults = try await db.batchSearch(queries: queryVectors, k: 2)
        
        #expect(batchResults.count == 2)
        #expect(batchResults[0].count == 2)
        #expect(batchResults[1].count == 2)
    }
    
    // MARK: - Export/Import Tests
    
    @Test func testJSONExportImport() async throws {
        let tempPath = NSTemporaryDirectory() + "test_export.json"
        defer { try? FileManager.default.removeItem(atPath: tempPath) }
        
        let db = try await VectorDatabase.createInMemory()
        
        let vectors = [
            Vector(values: [1.0, 2.0], metadata: ["type": AnyCodable("test")]),
            Vector(values: [3.0, 4.0], metadata: ["type": AnyCodable("sample")])
        ]
        
        try await db.insertBatch(vectors)
        
        // Export
        try await db.exportToJSON(filePath: tempPath)
        
        // Create new database and import
        let newDb = try await VectorDatabase.createInMemory()
        try await newDb.importFromJSON(filePath: tempPath)
        
        let importedVectors = try await newDb.getAll()
        #expect(importedVectors.count == 2)
    }
    
    // MARK: - Error Handling Tests
    
    @Test func testVectorErrorHandling() async throws {
        let store = try VectorStore(databasePath: ":memory:")
        
        let nonExistentId = UUID()
        
        await #expect(throws: VectorStoreError.vectorNotFound(nonExistentId)) {
            try await store.get(id: nonExistentId)
        }
    }
    
    @Test func testDimensionValidation() async throws {
        let store = try VectorStore(databasePath: ":memory:", expectedDimension: 3)
        
        let validVector = Vector(values: [1.0, 2.0, 3.0])
        let invalidVector = Vector(values: [1.0, 2.0])
        
        try await store.insert(validVector) // Should succeed
        
        await #expect(throws: VectorStoreError.dimensionMismatch(expected: 3, actual: 2)) {
            try await store.insert(invalidVector)
        }
    }
    
    // MARK: - Performance Tests
    
    @Test func testLargeVectorOperations() async throws {
        let db = try await VectorDatabase.createInMemory()
        
        // Create 100 vectors of dimension 50
        let vectors = (0..<100).map { i in
            let values = (0..<50).map { j in Float(i * 50 + j) / 1000.0 }
            return Vector(values: values)
        }
        
        try await db.insertBatch(vectors)
        
        let queryVector = Vector(values: Array(repeating: 0.5, count: 50))
        let results = try await db.search(query: queryVector, k: 10)
        
        #expect(results.count == 10)
        #expect(results.allSatisfy { $0.vector.dimension == 50 })
    }
    
    // MARK: - Framework Information Tests
    
    @Test func testFrameworkInfo() async throws {
        let info = SwiftVS.getFrameworkInfo()
        
        #expect(info.name == "SwiftVS")
        #expect(info.version == "1.0.0")
        #expect(info.features.count > 0)
        #expect(info.supportedPlatforms.contains(.iOS))
        #expect(info.supportedPlatforms.contains(.macOS))
    }
    
    @Test func testConvenienceMethods() async throws {
        let db = try await SwiftVS.createInMemoryDatabase()
        #expect(try await db.count() == 0)
        
        let vector = SwiftVS.createVector(values: [1.0, 2.0, 3.0])
        #expect(vector.dimension == 3)
        
        let searchOptions = SwiftVS.createSearchOptions(metric: .cosine)
        #expect(searchOptions.metric == .cosine)
    }
    
    // MARK: - AnyCodable Tests
    
    @Test func testAnyCodableEncoding() async throws {
        let metadata: [String: AnyCodable] = [
            "string": AnyCodable("hello"),
            "int": AnyCodable(42),
            "double": AnyCodable(3.14),
            "bool": AnyCodable(true)
        ]
        
        let data = try JSONEncoder().encode(metadata)
        let decoded = try JSONDecoder().decode([String: AnyCodable].self, from: data)
        
        #expect(decoded["string"]?.value as? String == "hello")
        #expect(decoded["int"]?.value as? Int == 42)
        #expect(decoded["double"]?.value as? Double == 3.14)
        #expect(decoded["bool"]?.value as? Bool == true)
    }
}
