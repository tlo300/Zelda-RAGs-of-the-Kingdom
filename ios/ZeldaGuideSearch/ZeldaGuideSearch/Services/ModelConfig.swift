// ModelConfig.swift
// Search-app configuration — embedding and vector search parameters only.
// Named ModelConfig so VectorSearchService compiles without modification.

import Foundation

struct ModelConfig {
    static var ragTopK: Int { 5 }
    static var minVectorSimilarity: Float { 0.3 }
    static var embeddingDimensions: Int { 384 }
    static var embeddingModelFilename: String { "MiniLMEmbedder.mlmodelc" }
    static var knowledgeBaseFilename: String { "knowledge_base.db" }
}
