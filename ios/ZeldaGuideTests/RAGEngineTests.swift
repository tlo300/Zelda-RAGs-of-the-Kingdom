// RAGEngineTests.swift
// XCTest unit tests for RAGEngine.
// All dependencies (embedder, vector search, LLM) are injected via mocks so tests
// run without real Core ML models or a bundled knowledge_base.db.

import CoreML
import XCTest
import SQLiteVec
@testable import ZeldaGuide

// MARK: - Mock EmbeddingService implementations

/// Returns a unit vector at the specified dimension. Used to drive predictable ranking
/// against the in-memory fixture DB.
struct MockEmbeddingService: EmbeddingService {
    let vector: [Float]

    init(primaryDimension: Int = 0) {
        var v = [Float](repeating: 0.0, count: ModelConfig.embeddingDimensions)
        v[primaryDimension] = 1.0
        self.vector = v
    }

    func embed(_ text: String) async throws -> [Float] { vector }
}

/// Always throws `EmbeddingError.invalidOutput` to simulate a Core ML failure.
struct FailingEmbeddingService: EmbeddingService {
    let message: String
    func embed(_ text: String) async throws -> [Float] {
        throw EmbeddingError.invalidOutput(message)
    }
}

/// Returns a vector with NaN at index 0 without throwing, to test the engine's own NaN guard.
struct NaNEmbeddingService: EmbeddingService {
    func embed(_ text: String) async throws -> [Float] {
        var v = [Float](repeating: 0.0, count: ModelConfig.embeddingDimensions)
        v[0] = .nan
        return v
    }
}

// MARK: - Stateful mock LLM predictor

/// Outputs tokens from `sequence` in order, then emits `eosToken` to end generation.
/// Uses a class so that the step counter is shared across calls (actors call predict
/// multiple times per generate() invocation).
final class SequencePredictor: LLMPredictor, @unchecked Sendable {
    private let sequence: [Int32]
    private let eosToken: Int32
    private var step = 0

    init(sequence: [Int32], eosToken: Int32 = 254) {
        self.sequence = sequence
        self.eosToken = eosToken
    }

    func predict(inputIDs: [Int32]) async throws -> [Float] {
        await Task.yield()
        let token = step < sequence.count ? sequence[step] : eosToken
        step += 1
        var logits = [Float](repeating: -1.0, count: 256)
        logits[Int(token)] = 10.0
        return logits
    }
}

/// Maps specific token IDs to words; encode returns a single-token sequence.
struct WordTokenizer: LLMTokenizer {
    var eosTokenID: Int32 { 254 }
    let map: [Int32: String]

    func encode(_ text: String) -> [Int32] { [0] }

    func decode(tokenID: Int32) -> String { map[tokenID] ?? "" }
}

// MARK: - Tests

final class RAGEngineTests: XCTestCase {

    // MARK: - End-to-end: Hylian Shield answer mentions Hyrule Castle

    func testHylianShieldAnswerMentionsHyruleCastle() async throws {
        // Embedder returns unit vector at dim 0 → retrieves the Hylian Shield chunk first.
        let embedder = MockEmbeddingService(primaryDimension: 0)

        try SQLiteVec.initialize()
        let db = try Database(.inMemory)
        try await populateFixture(db)
        let vectorSearch = VectorSearchService(database: db)

        // LLM outputs "Hyrule" then " Castle" then EOS.
        let predictor = SequencePredictor(sequence: [1, 2])
        let tokenizer = WordTokenizer(map: [1: "Hyrule", 2: " Castle"])
        let llm = LLMService(predictor: predictor, tokenizer: tokenizer)

        let engine = RAGEngine(embedder: embedder, vectorSearch: vectorSearch, llm: llm)

        var tokens = [String]()
        for await token in await engine.answer(question: "Where is the Hylian Shield?") {
            tokens.append(token)
        }

        let answer = tokens.joined()
        XCTAssertTrue(answer.contains("Hyrule Castle"),
                      "Answer must mention 'Hyrule Castle', got: \(answer)")
    }

    // MARK: - Zero results still streams from LLM

    func testZeroResultsStillStreamsFromLLM() async throws {
        let embedder = MockEmbeddingService()

        // Empty DB: schema present, no rows → vector search and FTS5 both return 0 results.
        try SQLiteVec.initialize()
        let db = try Database(.inMemory)
        try await populateEmptyFixture(db)
        let vectorSearch = VectorSearchService(database: db)

        let predictor = SequencePredictor(sequence: [42])
        let tokenizer = WordTokenizer(map: [42: "unknown"])
        let llm = LLMService(predictor: predictor, tokenizer: tokenizer)

        let engine = RAGEngine(embedder: embedder, vectorSearch: vectorSearch, llm: llm)

        var tokens = [String]()
        for await token in await engine.answer(question: "xyzzy unfindable query") {
            tokens.append(token)
        }

        XCTAssertFalse(tokens.isEmpty,
                       "Engine must stream LLM output even when 0 chunks are retrieved")
        XCTAssertFalse(tokens.first?.starts(with: "[Error") == true,
                       "Zero results must not surface as an error — LLM handles the no-context case")
    }

    // MARK: - Embedding failure surfaces error through the stream

    func testEmbeddingFailureSurfacesErrorThroughStream() async throws {
        let embedder = FailingEmbeddingService(message: "simulated Core ML failure")

        try SQLiteVec.initialize()
        let db = try Database(.inMemory)
        let vectorSearch = VectorSearchService(database: db)
        let llm = LLMService(predictor: MockPredictor(), tokenizer: MockTokenizer())

        let engine = RAGEngine(embedder: embedder, vectorSearch: vectorSearch, llm: llm)

        var tokens = [String]()
        for await token in await engine.answer(question: "test") {
            tokens.append(token)
        }

        XCTAssertEqual(tokens.count, 1,
                       "Embedding failure must yield exactly one error token then finish")
        XCTAssertTrue(tokens.first?.starts(with: "[Error") == true,
                      "Error token must start with '[Error', got: \(tokens.first ?? "")")
    }

    // MARK: - NaN embedding surfaces error through the stream

    func testNaNEmbeddingSurfacesErrorThroughStream() async throws {
        let embedder = NaNEmbeddingService()

        try SQLiteVec.initialize()
        let db = try Database(.inMemory)
        let vectorSearch = VectorSearchService(database: db)
        let llm = LLMService(predictor: MockPredictor(), tokenizer: MockTokenizer())

        let engine = RAGEngine(embedder: embedder, vectorSearch: vectorSearch, llm: llm)

        var tokens = [String]()
        for await token in await engine.answer(question: "test") {
            tokens.append(token)
        }

        XCTAssertEqual(tokens.count, 1,
                       "NaN embedding must yield exactly one error token then finish")
        XCTAssertTrue(tokens.first?.starts(with: "[Error") == true,
                      "Error token must start with '[Error', got: \(tokens.first ?? "")")
    }

    // MARK: - Source chunks exposed for attribution UI

    func testSourceChunksPopulatedAfterAnswer() async throws {
        let embedder = MockEmbeddingService(primaryDimension: 0)

        try SQLiteVec.initialize()
        let db = try Database(.inMemory)
        try await populateFixture(db)
        let vectorSearch = VectorSearchService(database: db)
        let llm = LLMService(predictor: MockPredictor(), tokenizer: MockTokenizer())

        let engine = RAGEngine(embedder: embedder, vectorSearch: vectorSearch, llm: llm)

        // Drain the stream fully before reading sourceChunks.
        for await _ in await engine.answer(question: "Where is the Hylian Shield?") { }

        let chunks = await engine.sourceChunks
        XCTAssertFalse(chunks.isEmpty,
                       "sourceChunks must be populated after answer() completes")
        XCTAssertEqual(chunks[0].pageTitle, "Chunk One",
                       "Top chunk must be the one closest to the query embedding (dim 0)")
    }

    // MARK: - CoreMLEmbeddingService.buildInputArrays shape contract

    func testBuildInputArraysShapeIsAlways1x128() throws {
        let cases = ["", "Hi", "What's the best first shrine to visit?",
                     String(repeating: "a", count: 200)]
        for text in cases {
            let (ids, mask) = try CoreMLEmbeddingService.buildInputArrays(for: text)
            XCTAssertEqual(ids.shape,  [1, 128], "inputIDs shape must be [1,128] for: \(text.prefix(20))")
            XCTAssertEqual(mask.shape, [1, 128], "attentionMask shape must be [1,128] for: \(text.prefix(20))")
        }
    }

    func testBuildInputArraysPaddingPositionsAreZero() throws {
        // "Hi" = 2 UTF-8 bytes → positions 2…127 must be zero-padded
        let (ids, mask) = try CoreMLEmbeddingService.buildInputArrays(for: "Hi")
        for i in 2..<128 {
            XCTAssertEqual(ids[i].int32Value,  0, "inputIDs[\(i)] should be 0 (padding)")
            XCTAssertEqual(mask[i].int32Value, 0, "attentionMask[\(i)] should be 0 (padding)")
        }
        XCTAssertEqual(mask[0].int32Value, 1, "attentionMask[0] should be 1 (real token)")
        XCTAssertEqual(mask[1].int32Value, 1, "attentionMask[1] should be 1 (real token)")
    }

    func testBuildInputArraysLongTextTruncatedTo128() throws {
        let longText = String(repeating: "a", count: 200)
        let (ids, mask) = try CoreMLEmbeddingService.buildInputArrays(for: longText)
        XCTAssertEqual(ids.shape, [1, 128])
        // All 128 positions are real tokens — mask should be all 1s
        for i in 0..<128 {
            XCTAssertEqual(mask[i].int32Value, 1, "attentionMask[\(i)] should be 1 (no padding)")
        }
    }

    func testBuildInputArraysEmptyTextIsAllZeroPadded() throws {
        let (ids, mask) = try CoreMLEmbeddingService.buildInputArrays(for: "")
        for i in 0..<128 {
            XCTAssertEqual(ids[i].int32Value,  0, "inputIDs[\(i)] should be 0 for empty input")
            XCTAssertEqual(mask[i].int32Value, 0, "attentionMask[\(i)] should be 0 for empty input")
        }
    }

    // MARK: - Fixture helpers

    /// Populates three chunks with predictable unit-vector embeddings (same schema as
    /// VectorSearchServiceTests so the same query vectors produce the same ranking).
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
            var embedding = [Float](repeating: 0.0, count: dims)
            embedding[f.id - 1] = 1.0
            try await db.execute(
                "INSERT INTO chunk_embeddings (rowid, embedding) VALUES (?, ?)",
                params: [f.id, embedding]
            )
        }
    }

    /// Schema-only fixture with no rows: both vector search and FTS5 return 0 results.
    private func populateEmptyFixture(_ db: Database) async throws {
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
    }
}
