// ChatViewModelTests.swift
// XCTest unit tests for ChatViewModel.
// Uses a real RAGEngine with injected mocks so no Core ML models or knowledge_base.db
// are needed. Mock types from LLMServiceTests and RAGEngineTests are reused from the
// same test target.

import XCTest
import SQLiteVec
@testable import ZeldaGuide

@MainActor
final class ChatViewModelTests: XCTestCase {

    // MARK: - Helpers

    /// Builds a ChatViewModel backed by a real RAGEngine wired to in-memory mocks.
    /// `tokens` controls what the LLM emits before EOS.
    private func makeViewModel(tokens: [Int32] = [42]) async throws -> ChatViewModel {
        try SQLiteVec.initialize()
        let db = try Database(.inMemory)
        try await populateEmptySchema(db)
        let vectorSearch = VectorSearchService(database: db)
        let predictor    = SequencePredictor(sequence: tokens)
        let tokenizer    = WordTokenizer(map: [42: "sword"])
        let llm          = LLMService(predictor: predictor, tokenizer: tokenizer)
        let engine       = RAGEngine(embedder: MockEmbeddingService(), vectorSearch: vectorSearch, llm: llm)
        return ChatViewModel(ragEngine: engine)
    }

    /// Populates an empty schema (vec0 + chunks + FTS5 tables, no rows).
    private func populateEmptySchema(_ db: Database) async throws {
        let dims = ModelConfig.embeddingDimensions
        try await db.execute(
            "CREATE VIRTUAL TABLE chunk_embeddings USING vec0(embedding float[\(dims)])")
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

    // MARK: - Tests

    func testEmptyInputDoesNotSend() async throws {
        let vm = try await makeViewModel()
        vm.inputText = "   "
        vm.send()
        XCTAssertTrue(vm.messages.isEmpty, "Empty/whitespace input must not add messages")
        XCTAssertFalse(vm.isGenerating)
    }

    func testSendAppendsUserAndAssistantMessages() async throws {
        let vm = try await makeViewModel()
        vm.inputText = "Where is the Hylian Shield?"
        vm.send()

        XCTAssertEqual(vm.messages.count, 2, "send() must append one user and one assistant message")
        XCTAssertEqual(vm.messages[0].role, .user)
        XCTAssertEqual(vm.messages[0].text, "Where is the Hylian Shield?")
        XCTAssertEqual(vm.messages[1].role, .assistant)
        XCTAssertTrue(vm.isGenerating, "isGenerating must be true while stream is in flight")

        vm.cancelGeneration()
    }

    func testInputClearedAfterSend() async throws {
        let vm = try await makeViewModel()
        vm.inputText = "test question"
        vm.send()
        XCTAssertEqual(vm.inputText, "", "inputText must be cleared immediately after send()")
        vm.cancelGeneration()
    }

    func testSecondSendIgnoredWhileGenerating() async throws {
        let vm = try await makeViewModel(tokens: Array(repeating: 42, count: 500))
        vm.inputText = "first"
        vm.send()
        vm.inputText = "second"
        vm.send()

        XCTAssertEqual(vm.messages.count, 2,
            "A second send() while generating must be ignored — only two messages expected")

        vm.cancelGeneration()
    }

    func testCancelStopsGeneration() async throws {
        let vm = try await makeViewModel(tokens: Array(repeating: 42, count: 500))
        vm.inputText = "test"
        vm.send()
        XCTAssertTrue(vm.isGenerating)

        vm.cancelGeneration()

        XCTAssertFalse(vm.isGenerating, "isGenerating must be false immediately after cancelGeneration()")
        XCTAssertFalse(vm.isLoading)
    }

    func testStreamCompletesAndSetsIsGeneratingFalse() async throws {
        let vm = try await makeViewModel(tokens: [42])  // One token then EOS
        vm.inputText = "test"
        vm.send()

        // Poll until generation finishes (short sequence, should complete quickly).
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while vm.isGenerating {
            try await Task.sleep(for: .milliseconds(20))
            if ContinuousClock.now > deadline {
                XCTFail("Generation did not complete within 5 seconds")
                return
            }
        }

        XCTAssertFalse(vm.isGenerating)
        XCTAssertEqual(vm.messages.last?.isStreaming, false)
        XCTAssertFalse(vm.messages.last?.text.isEmpty ?? true,
            "Completed assistant message must have non-empty text")
    }
}
