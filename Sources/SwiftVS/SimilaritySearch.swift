import Foundation

/// Represents a search result with a vector and its similarity score
public struct SimilaritySearchResult: Sendable {
    /// The vector that was found
    public let vector: Vector
    
    /// The similarity score (higher is more similar for cosine similarity)
    public let score: Float
    
    /// The distance (lower is more similar for distance metrics)
    public let distance: Float?
    
    public init(vector: Vector, score: Float, distance: Float? = nil) {
        self.vector = vector
        self.score = score
        self.distance = distance
    }
}

/// Distance metrics supported by the similarity search
public enum DistanceMetric: String, CaseIterable, Sendable {
    case cosine = "cosine"
    case euclidean = "euclidean"
    case dotProduct = "dot_product"
    
    public var description: String {
        switch self {
        case .cosine:
            return "Cosine Similarity"
        case .euclidean:
            return "Euclidean Distance"
        case .dotProduct:
            return "Dot Product"
        }
    }
}

/// Options for configuring similarity search
public struct SearchOptions: Sendable {
    /// The distance metric to use
    public let metric: DistanceMetric
    
    /// Whether to normalize query vector before search
    public let normalizeQuery: Bool
    
    /// Whether to normalize stored vectors before comparison
    public let normalizeStored: Bool
    
    /// Optional metadata filter to apply during search
    public let metadataFilter: (@Sendable (Vector) -> Bool)?
    
    public init(
        metric: DistanceMetric = .cosine,
        normalizeQuery: Bool = false,
        normalizeStored: Bool = false,
        metadataFilter: (@Sendable (Vector) -> Bool)? = nil
    ) {
        self.metric = metric
        self.normalizeQuery = normalizeQuery
        self.normalizeStored = normalizeStored
        self.metadataFilter = metadataFilter
    }
}

/// Protocol for similarity search implementations
public protocol SimilaritySearchable {
    /// Performs a K-nearest neighbor search
    /// - Parameters:
    ///   - query: The query vector
    ///   - k: Number of nearest neighbors to return
    ///   - options: Search configuration options
    /// - Returns: Array of search results sorted by similarity (best first)
    func search(
        query: Vector,
        k: Int,
        options: SearchOptions
    ) async throws -> [SimilaritySearchResult]
    
    /// Performs a similarity search with a minimum similarity threshold
    /// - Parameters:
    ///   - query: The query vector
    ///   - threshold: Minimum similarity threshold
    ///   - maxResults: Maximum number of results to return
    ///   - options: Search configuration options
    /// - Returns: Array of search results above the threshold
    func search(
        query: Vector,
        threshold: Float,
        maxResults: Int,
        options: SearchOptions
    ) async throws -> [SimilaritySearchResult]
}

/// Brute-force similarity search implementation
public actor BruteForceSimilaritySearch: SimilaritySearchable {
    private var vectors: [Vector] = []
    
    public init() {}
    
    /// Adds vectors to the search index
    /// - Parameter vectors: Vectors to add
    public func addVectors(_ vectors: [Vector]) {
        self.vectors.append(contentsOf: vectors)
    }
    
    /// Removes vectors from the search index
    /// - Parameter vectorIds: IDs of vectors to remove
    public func removeVectors(withIds vectorIds: Set<UUID>) {
        self.vectors.removeAll { vectorIds.contains($0.id) }
    }
    
    /// Updates vectors in the search index
    /// - Parameter vectors: Updated vectors
    public func updateVectors(_ vectors: [Vector]) {
        let updateMap = Dictionary(uniqueKeysWithValues: vectors.map { ($0.id, $0) })
        
        for i in 0..<self.vectors.count {
            if let updated = updateMap[self.vectors[i].id] {
                self.vectors[i] = updated
            }
        }
    }
    
    /// Clears all vectors from the search index
    public func clear() {
        self.vectors.removeAll()
    }
    
    /// Gets the current number of vectors in the index
    public func count() -> Int {
        return vectors.count
    }
    
    public func search(
        query: Vector,
        k: Int,
        options: SearchOptions = SearchOptions()
    ) async throws -> [SimilaritySearchResult] {
        guard k > 0 else { return [] }
        
        var queryVector = query
        if options.normalizeQuery {
            queryVector = try query.normalized()
        }
        
        // Filter vectors by metadata if filter is provided
        let candidateVectors = options.metadataFilter.map { filter in
            vectors.filter(filter)
        } ?? vectors
        
        // Calculate similarities
        var results: [SimilaritySearchResult] = []
        
        for vector in candidateVectors {
            let result = try await calculateSimilarity(
                query: queryVector,
                candidate: vector,
                metric: options.metric,
                normalizeStored: options.normalizeStored
            )
            results.append(result)
        }
        
        // Sort by similarity score (descending for similarity, ascending for distance)
        switch options.metric {
        case .cosine, .dotProduct:
            results.sort { $0.score > $1.score }
        case .euclidean:
            results.sort { $0.distance ?? Float.infinity < $1.distance ?? Float.infinity }
        }
        
        // Return top K results
        return Array(results.prefix(k))
    }
    
    public func search(
        query: Vector,
        threshold: Float,
        maxResults: Int = Int.max,
        options: SearchOptions = SearchOptions()
    ) async throws -> [SimilaritySearchResult] {
        var queryVector = query
        if options.normalizeQuery {
            queryVector = try query.normalized()
        }
        
        // Filter vectors by metadata if filter is provided
        let candidateVectors = options.metadataFilter.map { filter in
            vectors.filter(filter)
        } ?? vectors
        
        // Calculate similarities
        var results: [SimilaritySearchResult] = []
        
        for vector in candidateVectors {
            let result = try await calculateSimilarity(
                query: queryVector,
                candidate: vector,
                metric: options.metric,
                normalizeStored: options.normalizeStored
            )
            
            // Apply threshold filter
            let passesThreshold: Bool
            switch options.metric {
            case .cosine, .dotProduct:
                passesThreshold = result.score >= threshold
            case .euclidean:
                passesThreshold = (result.distance ?? Float.infinity) <= threshold
            }
            
            if passesThreshold {
                results.append(result)
            }
        }
        
        // Sort by similarity score (descending for similarity, ascending for distance)
        switch options.metric {
        case .cosine, .dotProduct:
            results.sort { $0.score > $1.score }
        case .euclidean:
            results.sort { $0.distance ?? Float.infinity < $1.distance ?? Float.infinity }
        }
        
        // Return results up to maxResults
        return Array(results.prefix(maxResults))
    }
    
    private func calculateSimilarity(
        query: Vector,
        candidate: Vector,
        metric: DistanceMetric,
        normalizeStored: Bool
    ) async throws -> SimilaritySearchResult {
        var candidateVector = candidate
        if normalizeStored {
            candidateVector = try candidate.normalized()
        }
        
        switch metric {
        case .cosine:
            let similarity = try query.cosineSimilarity(with: candidateVector)
            return SimilaritySearchResult(vector: candidate, score: similarity)
            
        case .euclidean:
            let distance = try query.euclideanDistance(to: candidateVector)
            // Convert distance to similarity score (inverse relationship)
            let similarity = 1.0 / (1.0 + distance)
            return SimilaritySearchResult(vector: candidate, score: similarity, distance: distance)
            
        case .dotProduct:
            let dotProduct = try query.dotProduct(with: candidateVector)
            return SimilaritySearchResult(vector: candidate, score: dotProduct)
        }
    }
}

// MARK: - Accelerate Framework Support

#if canImport(Accelerate)
import Accelerate

/// High-performance similarity search using Accelerate framework
public actor AcceleratedSimilaritySearch: SimilaritySearchable {
    private var vectors: [Vector] = []
    
    public init() {}
    
    /// Adds vectors to the search index
    /// - Parameter vectors: Vectors to add
    public func addVectors(_ vectors: [Vector]) {
        self.vectors.append(contentsOf: vectors)
    }
    
    /// Removes vectors from the search index
    /// - Parameter vectorIds: IDs of vectors to remove
    public func removeVectors(withIds vectorIds: Set<UUID>) {
        self.vectors.removeAll { vectorIds.contains($0.id) }
    }
    
    /// Updates vectors in the search index
    /// - Parameter vectors: Updated vectors
    public func updateVectors(_ vectors: [Vector]) {
        let updateMap = Dictionary(uniqueKeysWithValues: vectors.map { ($0.id, $0) })
        
        for i in 0..<self.vectors.count {
            if let updated = updateMap[self.vectors[i].id] {
                self.vectors[i] = updated
            }
        }
    }
    
    /// Clears all vectors from the search index
    public func clear() {
        self.vectors.removeAll()
    }
    
    /// Gets the current number of vectors in the index
    public func count() -> Int {
        return vectors.count
    }
    
    public func search(
        query: Vector,
        k: Int,
        options: SearchOptions = SearchOptions()
    ) async throws -> [SimilaritySearchResult] {
        guard k > 0 else { return [] }
        
        var queryVector = query
        if options.normalizeQuery {
            queryVector = try query.normalized()
        }
        
        // Filter vectors by metadata if filter is provided
        let candidateVectors = options.metadataFilter.map { filter in
            vectors.filter(filter)
        } ?? vectors
        
        guard !candidateVectors.isEmpty else { return [] }
        
        // Use Accelerate for high-performance computation
        let similarities = try await calculateSimilaritiesAccelerated(
            query: queryVector,
            candidates: candidateVectors,
            metric: options.metric,
            normalizeStored: options.normalizeStored
        )
        
        // Create results
        var results: [SimilaritySearchResult] = []
        for (index, similarity) in similarities.enumerated() {
            let result = SimilaritySearchResult(
                vector: candidateVectors[index],
                score: similarity.score,
                distance: similarity.distance
            )
            results.append(result)
        }
        
        // Sort by similarity score (descending for similarity, ascending for distance)
        switch options.metric {
        case .cosine, .dotProduct:
            results.sort { $0.score > $1.score }
        case .euclidean:
            results.sort { $0.distance ?? Float.infinity < $1.distance ?? Float.infinity }
        }
        
        // Return top K results
        return Array(results.prefix(k))
    }
    
    public func search(
        query: Vector,
        threshold: Float,
        maxResults: Int = Int.max,
        options: SearchOptions = SearchOptions()
    ) async throws -> [SimilaritySearchResult] {
        // For threshold-based search, get all results and then filter
        let allResults = try await search(query: query, k: vectors.count, options: options)
        
        let filteredResults = allResults.filter { result in
            switch options.metric {
            case .cosine, .dotProduct:
                return result.score >= threshold
            case .euclidean:
                return (result.distance ?? Float.infinity) <= threshold
            }
        }
        
        return Array(filteredResults.prefix(maxResults))
    }
    
    private func calculateSimilaritiesAccelerated(
        query: Vector,
        candidates: [Vector],
        metric: DistanceMetric,
        normalizeStored: Bool
    ) async throws -> [(score: Float, distance: Float?)] {
        let dimension = query.dimension
        let candidateCount = candidates.count
        
        // Prepare data matrices
        let queryData = query.values
        var candidateData: [Float] = []
        
        for candidate in candidates {
            var candidateValues = candidate.values
            if normalizeStored {
                let norm = sqrt(vDSP.sum(vDSP.square(candidateValues)))
                if norm > 0 {
                    vDSP.divide(candidateValues, norm, result: &candidateValues)
                }
            }
            candidateData.append(contentsOf: candidateValues)
        }
        
        var results: [(score: Float, distance: Float?)] = []
        
        switch metric {
        case .cosine:
            // Calculate cosine similarities using dot product
            let queryMagnitude = sqrt(vDSP.sum(vDSP.square(queryData)))
            
            for i in 0..<candidateCount {
                let candidateSlice = Array(candidateData[i * dimension..<(i + 1) * dimension])
                let candidateMagnitude = sqrt(vDSP.sum(vDSP.square(candidateSlice)))
                
                let dotProduct = vDSP.dot(queryData, candidateSlice)
                let similarity = dotProduct / (queryMagnitude * candidateMagnitude)
                results.append((score: similarity, distance: nil))
            }
            
        case .euclidean:
            for i in 0..<candidateCount {
                let candidateSlice = Array(candidateData[i * dimension..<(i + 1) * dimension])
                var differences = [Float](repeating: 0, count: dimension)
                vDSP.subtract(candidateSlice, queryData, result: &differences)
                
                let distance = sqrt(vDSP.sum(vDSP.square(differences)))
                let similarity = 1.0 / (1.0 + distance)
                
                results.append((score: similarity, distance: distance))
            }
            
        case .dotProduct:
            for i in 0..<candidateCount {
                let candidateSlice = Array(candidateData[i * dimension..<(i + 1) * dimension])
                let dotProduct = vDSP.dot(queryData, candidateSlice)
                results.append((score: dotProduct, distance: nil))
            }
        }
        
        return results
    }
}
#endif 