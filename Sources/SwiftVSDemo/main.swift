import Foundation
import SwiftVS

// MARK: - Demo Application

@main
struct SwiftVSDemo {
    static func main() async {
        print("🚀 SwiftVS Demo - Vector Database Framework")
        print("=" * 50)
        
        do {
            try await runDemo()
        } catch {
            print("❌ Error: \(error)")
        }
    }
    
    static func runDemo() async throws {
        // 1. Create a vector database
        print("\n1. Creating vector database...")
        let db = try await VectorDatabase.createInMemory()
        
        // 2. Create sample vectors representing document embeddings
        print("\n2. Creating sample document vectors...")
        let documents = [
            ("Machine Learning", [0.8, 0.2, 0.9, 0.1, 0.7]),
            ("Artificial Intelligence", [0.9, 0.1, 0.8, 0.2, 0.6]),
            ("Data Science", [0.7, 0.3, 0.8, 0.2, 0.8]),
            ("Web Development", [0.2, 0.8, 0.1, 0.9, 0.3]),
            ("Mobile Apps", [0.1, 0.9, 0.2, 0.8, 0.4]),
            ("Database Systems", [0.6, 0.4, 0.7, 0.3, 0.9])
        ]
        
        var vectors: [Vector] = []
        for (title, values) in documents {
            let metadata: [String: AnyCodable] = [
                "title": AnyCodable(title),
                "category": AnyCodable(getCategoryForTitle(title)),
                "timestamp": AnyCodable(ISO8601DateFormatter().string(from: Date()))
            ]
            let vector = Vector(values: values.map { Float($0) }, metadata: metadata)
            vectors.append(vector)
        }
        
        // 3. Insert vectors into database
        print("\n3. Inserting vectors into database...")
        try await db.insertBatch(vectors)
        
        let count = try await db.count()
        print("   ✅ Inserted \(count) vectors")
        
        // 4. Perform similarity search
        print("\n4. Performing similarity search...")
        let queryVector = Vector(values: [0.85, 0.15, 0.9, 0.1, 0.65]) // AI/ML-like vector
        
        let searchResults = try await db.search(query: queryVector, k: 3)
        print("   🔍 Query vector: [0.85, 0.15, 0.9, 0.1, 0.65]")
        print("   📊 Top 3 most similar documents:")
        
        for (index, result) in searchResults.enumerated() {
            let title = result.vector.metadata?["title"]?.value as? String ?? "Unknown"
            let category = result.vector.metadata?["category"]?.value as? String ?? "Unknown"
            print("   \(index + 1). \(title) (Category: \(category))")
            print("      Similarity: \(String(format: "%.3f", result.score))")
        }
        
        // 5. Search with different metrics
        print("\n5. Comparing different distance metrics...")
        
        let cosineOptions = SearchOptions(metric: .cosine)
        let euclideanOptions = SearchOptions(metric: .euclidean)
        let dotProductOptions = SearchOptions(metric: .dotProduct)
        
        let cosineResults = try await db.search(query: queryVector, k: 2, options: cosineOptions)
        let euclideanResults = try await db.search(query: queryVector, k: 2, options: euclideanOptions)
        let dotProductResults = try await db.search(query: queryVector, k: 2, options: dotProductOptions)
        
        print("   📈 Cosine Similarity - Top result: \(getTitle(cosineResults[0]))")
        print("   📐 Euclidean Distance - Top result: \(getTitle(euclideanResults[0]))")
        print("   🔢 Dot Product - Top result: \(getTitle(dotProductResults[0]))")
        
        // 6. Search with metadata filtering
        print("\n6. Searching with metadata filtering...")
        
        let techFilter: @Sendable (Vector) -> Bool = { vector in
            let category = vector.metadata?["category"]?.value as? String
            return category == "Technology"
        }
        
        let filteredOptions = SearchOptions(
            metric: .cosine,
            metadataFilter: techFilter
        )
        
        let filteredResults = try await db.search(query: queryVector, k: 5, options: filteredOptions)
        print("   🔍 Technology-only search results:")
        for (index, result) in filteredResults.enumerated() {
            let title = getTitle(result)
            print("   \(index + 1). \(title) (Score: \(String(format: "%.3f", result.score)))")
        }
        
        // 7. Find similar vectors to existing ones
        print("\n7. Finding similar vectors to existing documents...")
        if let firstVector = vectors.first {
            let similarResults = try await db.findSimilar(to: firstVector.id, k: 2)
            let originalTitle = getTitle(firstVector)
            print("   🔗 Vectors similar to '\(originalTitle)':")
            
            for (index, result) in similarResults.enumerated() {
                let title = getTitle(result)
                print("   \(index + 1). \(title) (Score: \(String(format: "%.3f", result.score)))")
            }
        }
        
        // 8. Batch search
        print("\n8. Performing batch search...")
        let queryVectors = [
            Vector(values: [0.8, 0.2, 0.9, 0.1, 0.7]), // ML-like
            Vector(values: [0.2, 0.8, 0.1, 0.9, 0.3])  // Web dev-like
        ]
        
        let batchResults = try await db.batchSearch(queries: queryVectors, k: 2)
        print("   📦 Batch search results:")
        for (queryIndex, results) in batchResults.enumerated() {
            print("   Query \(queryIndex + 1) results:")
            for (index, result) in results.enumerated() {
                let title = getTitle(result)
                print("     \(index + 1). \(title)")
            }
        }
        
        // 9. Database statistics
        print("\n9. Database statistics...")
        let stats = try await db.getStatistics()
        print("   📊 Total vectors: \(stats.totalVectors)")
        print("   📏 Unique dimensions: \(stats.uniqueDimensions)")
        print("   📊 Average dimension: \(stats.averageDimension)")
        print("   🔧 Search engine: \(stats.searchEngineType)")
        
        // 10. Export/Import demonstration
        print("\n10. Export/Import demonstration...")
        let tempPath = NSTemporaryDirectory() + "demo_export.json"
        defer { try? FileManager.default.removeItem(atPath: tempPath) }
        
        try await db.exportToJSON(filePath: tempPath)
        print("   💾 Exported vectors to JSON")
        
        let newDb = try await VectorDatabase.createInMemory()
        try await newDb.importFromJSON(filePath: tempPath)
        let importedCount = try await newDb.count()
        print("   📥 Imported \(importedCount) vectors to new database")
        
        // 11. Framework information
        print("\n11. Framework information...")
        let info = SwiftVS.getFrameworkInfo()
        print("   🏷️  Framework: \(info.name) v\(info.version)")
        print("   📝 Description: \(info.description)")
        print("   🎯 Features: \(info.features.joined(separator: ", "))")
        print("   📱 Platforms: \(info.supportedPlatforms.map { $0.rawValue }.joined(separator: ", "))")
        print("   ⚡ Accelerate available: \(SwiftVS.isAccelerateAvailable())")
        
        print("\n✅ Demo completed successfully!")
    }
    
    // Helper functions
    static func getCategoryForTitle(_ title: String) -> String {
        switch title {
        case "Machine Learning", "Artificial Intelligence", "Data Science":
            return "AI/ML"
        case "Web Development", "Mobile Apps":
            return "Technology"
        case "Database Systems":
            return "Data"
        default:
            return "Other"
        }
    }
    
    static func getTitle(_ vector: Vector) -> String {
        return vector.metadata?["title"]?.value as? String ?? "Unknown"
    }
    
    static func getTitle(_ result: SimilaritySearchResult) -> String {
        return result.vector.metadata?["title"]?.value as? String ?? "Unknown"
    }
}

// Extension to repeat strings
extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
} 