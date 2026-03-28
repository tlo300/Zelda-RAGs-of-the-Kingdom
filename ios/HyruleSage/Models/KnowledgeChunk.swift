// KnowledgeChunk.swift
// A single retrieved knowledge chunk returned by VectorSearchService, including its
// cosine similarity score relative to the query embedding.

import Foundation

struct KnowledgeChunk: Sendable {
    let id: Int
    let chunkText: String
    let source: String
    let pageTitle: String
    let similarityScore: Float
}
