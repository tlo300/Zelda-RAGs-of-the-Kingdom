// VectorSearchService.swift
// Opens knowledge_base.db from the app bundle and performs vector similarity search
// using sqlite-vec's vec0 virtual table. Falls back to FTS5 keyword search when a
// text query is provided and vector search returns no results.

import Foundation
import os
import SQLiteVec

private let vsLog = Logger(subsystem: "com.tlo300.HyruleSage", category: "VectorSearch")

enum VectorSearchError: Error, LocalizedError {
    case databaseNotFound(String)

    var errorDescription: String? {
        switch self {
        case .databaseNotFound(let name):
            return "Knowledge base '\(name)' not found in the app bundle. " +
                   "The app may be incomplete — please reinstall."
        }
    }
}

actor VectorSearchService {

    private let db: Database

    /// Opens knowledge_base.db from Bundle.main. Throws `VectorSearchError.databaseNotFound`
    /// (with a user-visible message) if the file is absent — never crashes or silently returns empty.
    init() throws {
        let filename = ModelConfig.knowledgeBaseFilename
        let nameURL = URL(fileURLWithPath: filename)
        let baseName = nameURL.deletingPathExtension().lastPathComponent
        let ext = nameURL.pathExtension
        guard let url = Bundle.main.url(forResource: baseName, withExtension: ext) else {
            throw VectorSearchError.databaseNotFound(filename)
        }
        try SQLiteVec.initialize()
        db = try Database(.uri(url.path), readonly: true)
    }

    /// Designated initialiser for testing — injects a pre-configured database.
    init(database: Database) {
        db = database
    }

    // MARK: - Public API

    /// Returns the top-K chunks ranked by cosine similarity to `query`.
    func search(query: [Float], topK: Int) async throws -> [KnowledgeChunk] {
        return try await vectorSearch(query: query, topK: topK)
    }

    /// Returns the top-K chunks by cosine similarity. Falls back to FTS5 keyword search
    /// when vector search returns no results OR when the best similarity score is below
    /// `ModelConfig.minVectorSimilarity`.
    func search(queryText: String, queryEmbedding: [Float], topK: Int) async throws -> [KnowledgeChunk] {
        let results: [KnowledgeChunk]
        do {
            results = try await vectorSearch(query: queryEmbedding, topK: topK)
        } catch {
            vsLog.error("Vector search error: \(error.localizedDescription, privacy: .public) — falling back to FTS5")
            return try await ftsSearch(queryText: queryText, topK: topK)
        }
        let bestSimilarity = results.map(\.similarityScore).max() ?? 0
        vsLog.notice("Vector search: \(results.count, privacy: .public) results, best similarity \(bestSimilarity, privacy: .public)")
        for (i, chunk) in results.enumerated() {
            vsLog.debug("  [\(i + 1, privacy: .public)] \(chunk.similarityScore, privacy: .public) \(chunk.pageTitle, privacy: .public)")
        }
        if bestSimilarity >= ModelConfig.minVectorSimilarity { return results }
        vsLog.notice("Best similarity \(bestSimilarity, privacy: .public) below threshold \(ModelConfig.minVectorSimilarity, privacy: .public) — falling back to FTS5")
        let ftsResults = try await ftsSearch(queryText: queryText, topK: topK)
        vsLog.notice("FTS5 fallback: \(ftsResults.count, privacy: .public) results")
        for (i, chunk) in ftsResults.enumerated() {
            vsLog.debug("  [\(i + 1, privacy: .public)] \(chunk.pageTitle, privacy: .public)")
        }
        return ftsResults
    }

    // MARK: - Private helpers

    private func vectorSearch(query: [Float], topK: Int) async throws -> [KnowledgeChunk] {
        let sql = """
            WITH knn AS (
                SELECT rowid, distance
                FROM chunk_embeddings
                WHERE embedding MATCH ?
                AND k = \(topK)
                ORDER BY distance
            )
            SELECT knn.rowid AS id, c.chunk_text, c.source, c.page_title, knn.distance
            FROM knn
            JOIN chunks c ON knn.rowid = c.id
            """
        let rows = try await db.query(sql, params: [query])
        return rows.compactMap { decodeVectorRow($0) }
    }

    private func ftsSearch(queryText: String, topK: Int) async throws -> [KnowledgeChunk] {
        let sql = """
            SELECT c.id, c.chunk_text, c.source, c.page_title
            FROM chunks_fts
            JOIN chunks c ON chunks_fts.rowid = c.id
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT \(topK)
            """
        let rows = try await db.query(sql, params: [queryText])
        return rows.compactMap { decodeFTSRow($0) }
    }

    private func decodeVectorRow(_ row: [String: any Sendable]) -> KnowledgeChunk? {
        guard
            let chunkText = row["chunk_text"] as? String,
            let source    = row["source"] as? String,
            let pageTitle = row["page_title"] as? String
        else { return nil }
        let id = intValue(row["id"]) ?? 0
        let distance: Double
        if let d = row["distance"] as? Double {
            distance = d
        } else if let f = row["distance"] as? Float {
            distance = Double(f)
        } else {
            return nil
        }
        let similarity = Float(max(0.0, 1.0 - (distance * distance / 2.0)))
        return KnowledgeChunk(id: id, chunkText: chunkText, source: source,
                              pageTitle: pageTitle, similarityScore: similarity)
    }

    private func decodeFTSRow(_ row: [String: any Sendable]) -> KnowledgeChunk? {
        guard
            let chunkText = row["chunk_text"] as? String,
            let source    = row["source"] as? String,
            let pageTitle = row["page_title"] as? String
        else { return nil }
        let id = intValue(row["id"]) ?? 0
        return KnowledgeChunk(id: id, chunkText: chunkText, source: source,
                              pageTitle: pageTitle, similarityScore: 0.0)
    }

    private func intValue(_ value: (any Sendable)?) -> Int? {
        switch value {
        case let v as Int:   return v
        case let v as Int64: return Int(v)
        case let v as Int32: return Int(v)
        default:             return nil
        }
    }
}
