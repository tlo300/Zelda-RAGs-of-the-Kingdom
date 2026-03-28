// LLMServiceTests.swift
// XCTest unit tests for LLMService.
// Uses injected mock predictor and tokenizer so tests run without the real Core ML model.

import UIKit
import XCTest
@testable import ZeldaGuide

// MARK: - Mock implementations

/// Predictor that always returns logits where a fixed answer token wins.
struct MockPredictor: LLMPredictor {
    var answerToken: Int32 = 42
    var vocabSize: Int     = 256

    func predict(inputIDs: [Int32]) async throws -> [Float] {
        await Task.yield()  // Ensure the actor releases between steps for cancellation tests.
        var logits = [Float](repeating: -1.0, count: vocabSize)
        logits[Int(answerToken)] = 10.0
        return logits
    }

    func reset() {}
}

/// Tokenizer that splits on whitespace and maps token 42 → "sword".
struct MockTokenizer: LLMTokenizer {
    var eosTokenID: Int32 { 255 }

    func encode(_ text: String) -> [Int32] {
        text.split(separator: " ").enumerated().map { Int32($0.offset + 1) }
    }

    func decode(tokenID: Int32) -> String {
        tokenID == 42 ? "sword" : ""
    }
}

// MARK: - Tests

final class LLMServiceTests: XCTestCase {

    private var service: LLMService!

    override func setUp() async throws {
        try await super.setUp()
        service = LLMService(predictor: MockPredictor(), tokenizer: MockTokenizer())
    }

    // MARK: - Non-empty output

    func testGenerateProducesNonEmptyOutput() async throws {
        var tokens = [String]()
        for await token in await service.generate(prompt: "What is the Hylian Shield?") {
            tokens.append(token)
            if tokens.count >= 5 { break }
        }
        XCTAssertFalse(tokens.isEmpty, "generate() must produce at least one token")
        XCTAssertTrue(tokens.allSatisfy { !$0.isEmpty }, "Every yielded token must be non-empty")
    }

    // MARK: - Max output tokens respected

    func testMaxOutputTokensIsRespected() async throws {
        var count = 0
        for await _ in await service.generate(prompt: "test") {
            count += 1
        }
        XCTAssertLessThanOrEqual(count, ModelConfig.maxOutputTokens,
                                 "Token count must not exceed ModelConfig.maxOutputTokens")
    }

    // MARK: - Prior generation is cancelled on new generate()

    func testNewGenerationCancelsPrior() async throws {
        let stream1 = await service.generate(prompt: "first")
        var iter1 = stream1.makeAsyncIterator()

        // Consume at least one token so task1 is running.
        let firstToken = await iter1.next()
        XCTAssertNotNil(firstToken, "stream1 must produce at least one token before cancellation")

        // Starting a second generation must cancel the first.
        let stream2 = await service.generate(prompt: "second")

        // stream2 must produce output.
        var tokens2 = [String]()
        for await token in stream2 {
            tokens2.append(token)
            if tokens2.count >= 3 { break }
        }
        XCTAssertFalse(tokens2.isEmpty, "stream2 must produce tokens after cancelling stream1")

        // stream1 must terminate, not hang.
        let finished = expectation(description: "stream1 terminates after cancellation")
        Task {
            while await iter1.next() != nil { }
            finished.fulfill()
        }
        await fulfillment(of: [finished], timeout: 5.0)
    }

    // MARK: - Memory warning pauses generation

    func testMemoryWarningPausesGeneration() async throws {
        let stream = await service.generate(prompt: "Where is the Hylian Shield?")
        var iter = stream.makeAsyncIterator()

        // Consume one token to confirm generation is running.
        let firstToken = await iter.next()
        XCTAssertNotNil(firstToken)

        // Fire a simulated memory warning on the main thread.
        await MainActor.run {
            NotificationCenter.default.post(
                name: UIApplication.didReceiveMemoryWarningNotification,
                object: nil
            )
        }

        // Drain the stream — it must terminate (not hang) after the warning.
        let finished = expectation(description: "stream finishes after memory warning")
        Task {
            while await iter.next() != nil { }
            finished.fulfill()
        }
        await fulfillment(of: [finished], timeout: 5.0)
    }

    // MARK: - Not-loaded error surfaces through the stream

    func testNotLoadedSurfacesErrorThroughStream() async throws {
        let unloaded = LLMService()   // no load() called, no mock injected
        var tokens = [String]()
        for await token in await unloaded.generate(prompt: "test") {
            tokens.append(token)
        }
        XCTAssertEqual(tokens.count, 1, "Should yield exactly one error message then finish")
        XCTAssertTrue(tokens.first?.starts(with: "[Error") == true,
                      "Error message must start with '[Error'")
    }
}
