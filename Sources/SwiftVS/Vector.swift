import Foundation

/// A vector with associated metadata for storage and retrieval in SwiftVS
public struct Vector: Codable, Equatable, Sendable {
    
    /// Unique identifier for the vector
    public let id: UUID
    
    /// The vector data as an array of floating-point values
    public let values: [Float]
    
    /// Optional metadata associated with the vector
    public let metadata: [String: AnyCodable]?
    
    /// Timestamp when the vector was created
    public let createdAt: Date
    
    /// Timestamp when the vector was last updated
    public let updatedAt: Date
    
    /// The dimensionality of the vector
    public var dimension: Int {
        return values.count
    }
    
    /// The L2 (Euclidean) norm of the vector
    public var magnitude: Float {
        return sqrt(values.reduce(0) { $0 + $1 * $1 })
    }
    
    /// Creates a new vector with the given values and optional metadata
    /// - Parameters:
    ///   - values: The vector values
    ///   - metadata: Optional metadata dictionary
    ///   - id: Optional custom ID (generates UUID if not provided)
    public init(values: [Float], metadata: [String: AnyCodable]? = nil, id: UUID = UUID()) {
        self.id = id
        self.values = values
        self.metadata = metadata
        let now = Date()
        self.createdAt = now
        self.updatedAt = now
    }
    
    /// Creates an updated copy of this vector with new values and/or metadata
    /// - Parameters:
    ///   - values: New vector values (uses existing if nil)
    ///   - metadata: New metadata (uses existing if nil)
    /// - Returns: A new Vector instance with updated values
    public func updated(values: [Float]? = nil, metadata: [String: AnyCodable]? = nil) -> Vector {
        return Vector(
            id: self.id,
            values: values ?? self.values,
            metadata: metadata ?? self.metadata,
            createdAt: self.createdAt,
            updatedAt: Date()
        )
    }
    
    /// Internal initializer for creating vectors with specific timestamps
    internal init(id: UUID, values: [Float], metadata: [String: AnyCodable]?, createdAt: Date, updatedAt: Date) {
        self.id = id
        self.values = values
        self.metadata = metadata
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}

// MARK: - Vector Operations

public extension Vector {
    
    /// Computes the dot product with another vector
    /// - Parameter other: The other vector
    /// - Returns: The dot product as a Float
    /// - Throws: VectorError.dimensionMismatch if vectors have different dimensions
    func dotProduct(with other: Vector) throws -> Float {
        guard self.dimension == other.dimension else {
            throw VectorError.dimensionMismatch(self.dimension, other.dimension)
        }
        
        return zip(self.values, other.values).reduce(0) { $0 + $1.0 * $1.1 }
    }
    
    /// Computes the cosine similarity with another vector
    /// - Parameter other: The other vector
    /// - Returns: The cosine similarity as a Float (between -1 and 1)
    /// - Throws: VectorError.dimensionMismatch if vectors have different dimensions
    func cosineSimilarity(with other: Vector) throws -> Float {
        let dotProduct = try self.dotProduct(with: other)
        let magnitudes = self.magnitude * other.magnitude
        
        guard magnitudes > 0 else {
            throw VectorError.zeroMagnitude
        }
        
        return dotProduct / magnitudes
    }
    
    /// Computes the Euclidean distance to another vector
    /// - Parameter other: The other vector
    /// - Returns: The Euclidean distance as a Float
    /// - Throws: VectorError.dimensionMismatch if vectors have different dimensions
    func euclideanDistance(to other: Vector) throws -> Float {
        guard self.dimension == other.dimension else {
            throw VectorError.dimensionMismatch(self.dimension, other.dimension)
        }
        
        let squaredDifferences = zip(self.values, other.values).map { pow($0 - $1, 2) }
        return sqrt(squaredDifferences.reduce(0, +))
    }
    
    /// Normalizes the vector to unit length
    /// - Returns: A new Vector with normalized values
    /// - Throws: VectorError.zeroMagnitude if the vector has zero magnitude
    func normalized() throws -> Vector {
        let mag = self.magnitude
        guard mag > 0 else {
            throw VectorError.zeroMagnitude
        }
        
        let normalizedValues = self.values.map { $0 / mag }
        return Vector(values: normalizedValues, metadata: self.metadata, id: self.id)
    }
}

// MARK: - AnyCodable Support

/// A type-erased codable value for flexible metadata storage
public struct AnyCodable: Codable, Equatable, Sendable {
    public let value: any Sendable
    
    public init<T: Codable & Sendable>(_ value: T) {
        self.value = value
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        
        if let bool = try? container.decode(Bool.self) {
            self.value = bool
        } else if let int = try? container.decode(Int.self) {
            self.value = int
        } else if let double = try? container.decode(Double.self) {
            self.value = double
        } else if let string = try? container.decode(String.self) {
            self.value = string
        } else if let array = try? container.decode([AnyCodable].self) {
            self.value = array
        } else if let dictionary = try? container.decode([String: AnyCodable].self) {
            self.value = dictionary
        } else {
            throw DecodingError.typeMismatch(AnyCodable.self, DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Unsupported type"))
        }
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        
        switch value {
        case let bool as Bool:
            try container.encode(bool)
        case let int as Int:
            try container.encode(int)
        case let double as Double:
            try container.encode(double)
        case let string as String:
            try container.encode(string)
        case let array as [AnyCodable]:
            try container.encode(array)
        case let dictionary as [String: AnyCodable]:
            try container.encode(dictionary)
        default:
            throw EncodingError.invalidValue(value, EncodingError.Context(codingPath: encoder.codingPath, debugDescription: "Unsupported type"))
        }
    }
    
    public static func == (lhs: AnyCodable, rhs: AnyCodable) -> Bool {
        switch (lhs.value, rhs.value) {
        case let (lhs as Bool, rhs as Bool):
            return lhs == rhs
        case let (lhs as Int, rhs as Int):
            return lhs == rhs
        case let (lhs as Double, rhs as Double):
            return lhs == rhs
        case let (lhs as String, rhs as String):
            return lhs == rhs
        case let (lhs as [AnyCodable], rhs as [AnyCodable]):
            return lhs == rhs
        case let (lhs as [String: AnyCodable], rhs as [String: AnyCodable]):
            return lhs == rhs
        default:
            return false
        }
    }
}

// MARK: - Errors

/// Errors that can occur during vector operations
public enum VectorError: LocalizedError, Equatable {
    case dimensionMismatch(Int, Int)
    case zeroMagnitude
    case invalidDimension(Int)
    
    public var errorDescription: String? {
        switch self {
        case .dimensionMismatch(let dim1, let dim2):
            return "Vector dimension mismatch: \(dim1) vs \(dim2)"
        case .zeroMagnitude:
            return "Cannot perform operation on zero-magnitude vector"
        case .invalidDimension(let dim):
            return "Invalid vector dimension: \(dim)"
        }
    }
} 