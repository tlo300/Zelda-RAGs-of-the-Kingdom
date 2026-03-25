// VectorSearchServiceTests.swift
// XCTest unit tests for VectorSearchService. Uses an in-memory SQLite+vec0 fixture to
// verify correct cosine-similarity ranking and FTS5 keyword fallback behaviour.

import XCTest
import SQLiteVec
@testable import ZeldaGuide

final class VectorSearchServiceTests: XCTestCase {

    private var service: VectorSearchService!

    override func setUp() async throws {
        try await super.setUp()
        try SQLiteVec.initialize()
        let db = try Database(.inMemory)
        try await populateFixture(db)
        service = VectorSearchService(database: db)
    }

    // MARK: - Ranking tests

    func testVectorSearchRanksClosestFirst() async throws {
        // Query is strongly biased toward dimension 0 → chunk "Chunk One" should rank first
        let query = makeQueryVector(primary: 0, secondary: 1, primaryWeight: 0.99)
        let results = try await service.search(query: query, topK: 3)
        XCTAssertEqual(results.count, 3)
        XCTAssertEqual(results[0].pageTitle, "Chunk One",
                       "Closest chunk must rank first")
        XCTAssertEqual(results[1].pageTitle, "Chunk Two",
                       "Second-closest must rank second")
        XCTAssertGreaterThan(results[0].similarityScore, results[1].similarityScore)
    }

    func testVectorSearchRanksSecondChunkCorrectly() async throws {
        // Query biased toward dimension 1 → chunk "Chunk Two" should rank first
        let query = makeQueryVector(primary: 1, secondary: 0, primaryWeight: 0.99)
        let results = try await service.search(query: query, topK: 3)
        XCTAssertEqual(results[0].pageTitle, "Chunk Two")
    }

    // MARK: - Field correctness

    func testVectorSearchReturnsCorrectFields() async throws {
        let query = makeQueryVector(primary: 2, secondary: 0, primaryWeight: 0.99)
        let results = try await service.search(query: query, topK: 1)
        let chunk = try XCTUnwrap(results.first)
        XCTAssertEqual(chunk.pageTitle, "Chunk Three")
        XCTAssertFalse(chunk.chunkText.isEmpty)
        XCTAssertFalse(chunk.source.isEmpty)
        XCTAssertGreaterThanOrEqual(chunk.similarityScore, 0.0)
        XCTAssertLessThanOrEqual(chunk.similarityScore, 1.0)
    }

    // MARK: - topK limit

    func testTopKLimitsResultCount() async throws {
        let query = makeQueryVector(primary: 0, secondary: 1, primaryWeight: 0.6)
        let results = try await service.search(query: query, topK: 2)
        XCTAssertEqual(results.count, 2)
    }

    // MARK: - FTS5 fallback

    func testFTSFallbackWhenVecTableIsEmpty() async throws {
        // Build a separate service whose vec0 table has no rows → vector search returns 0
        // results → FTS5 fallback should kick in.
        try SQLiteVec.initialize()
        let db = try Database(.inMemory)
        try await populateFTSOnlyFixture(db)
        let ftsService = VectorSearchService(database: db)

        let dummyEmbedding = [Float](repeating: 0.1, count: ModelConfig.embeddingDimensions)
        let results = try await ftsService.search(
            queryText: "Hylian Shield",
            queryEmbedding: dummyEmbedding,
            topK: 5
        )
        XCTAssertFalse(results.isEmpty, "FTS5 fallback must return results when vec0 is empty")
        XCTAssertTrue(results.contains { $0.chunkText.contains("Hylian") },
                      "FTS5 result must contain the queried keyword")
    }

    // MARK: - Fixture helpers

    /// Populates an in-memory DB with three chunks and matching unit-vector embeddings.
    private func populateFixture(_ db: Database) async throws {
        let dims = ModelConfig.embeddingDimensions
        try await db.execute("""
            CREATE VIRTUAL TABLE chunk_embeddings USING vec0(embedding float[\(dims)])
            """)
        try await db.execute("""
            CREATE TABLE chunks (
                id          INTEGER PRIMARY KEY,
                chunk_text  TEXT    NOT NULL,
                source      TEXT    NOT NULL,
                page_title  TEXT    NOT NULL,
                chunk_index INTEGER NOT NULL
            )
            """)
        try await db.execute("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                chunk_text,
                content='chunks',
                content_rowid='id'
            )
            """)

        let fixtures: [(id: Int, text: String, source: String, title: String)] = [
            (1, "Hylian Shield found in Hyrule Castle dungeon.", "zelda_wiki",    "Chunk One"),
            (2, "Ancient Blade crafted from Bokoblin Fang.",     "compendium",    "Chunk Two"),
            (3, "Akkala Shrine located in Akkala Highlands.",    "zelda_dungeon", "Chunk Three"),
        ]
        for f in fixtures {
            try await db.execute(
                "INSERT INTO chunks (id, chunk_text, source, page_title, chunk_index) VALUES (?, ?, ?, ?, 0)",
                params: [f.id, f.text, f.source, f.title]
            )
            try await db.execute(
                "INSERT INTO chunks_fts (rowid, chunk_text) VALUES (?, ?)",
                params: [f.id, f.text]
            )
            // Each chunk gets a unit vector along its own dimension for predictable ranking.
            let embedding = makeUnitVector(primaryDimension: f.id - 1)
            try await db.execute(
                "INSERT INTO chunk_embeddings (rowid, embedding) VALUES (?, ?)",
                params: [f.id, embedding]
            )
        }
    }

    /// Populates an in-memory DB with FTS content only — vec0 table is empty.
    private func populateFTSOnlyFixture(_ db: Database) async throws {
        let dims = ModelConfig.embeddingDimensions
        try await db.execute("""
            CREATE VIRTUAL TABLE chunk_embeddings USING vec0(embedding float[\(dims)])
            """)
        try await db.execute("""
            CREATE TABLE chunks (
                id          INTEGER PRIMARY KEY,
                chunk_text  TEXT    NOT NULL,
                source      TEXT    NOT NULL,
                page_title  TEXT    NOT NULL,
                chunk_index INTEGER NOT NULL
            )
            """)
        try await db.execute("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                chunk_text,
                content='chunks',
                content_rowid='id'
            )
            """)
        try await db.execute(
            "INSERT INTO chunks (id, chunk_text, source, page_title, chunk_index) VALUES (1, 'Hylian Shield found in Hyrule Castle.', 'zelda_wiki', 'Hylian Shield', 0)"
        )
        try await db.execute(
            "INSERT INTO chunks_fts (rowid, chunk_text) VALUES (1, 'Hylian Shield found in Hyrule Castle.')"
        )
        // Intentionally no rows in chunk_embeddings.
    }

    // MARK: - Vector construction

    /// Returns a unit vector (1.0 at `dim`, 0.0 everywhere else).
    private func makeUnitVector(primaryDimension dim: Int) -> [Float] {
        var v = [Float](repeating: 0.0, count: ModelConfig.embeddingDimensions)
        v[dim] = 1.0
        return v
    }

    /// Returns a vector with `primaryWeight` at `primary` and `1 − primaryWeight` at `secondary`.
    private func makeQueryVector(primary: Int, secondary: Int, primaryWeight: Float) -> [Float] {
        var v = [Float](repeating: 0.0, count: ModelConfig.embeddingDimensions)
        v[primary]   = primaryWeight
        v[secondary] = 1.0 - primaryWeight
        return v
    }
}
