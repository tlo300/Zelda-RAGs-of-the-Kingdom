// BPETokenizerTests.swift
// XCTest unit tests for BPETokenizer.
// Unit tests use BPETokenizer.init(data:) with a minimal in-memory vocab.
// Integration tests that require the real tokenizer.json are skipped when it is absent.

import XCTest
@testable import ZeldaGuide

// MARK: - Minimal tokenizer JSON

/// Minimal vocab covering "Hello" BPE merges and Qwen2.5 special tokens.
/// Ġ (U+0120) is the byte-level encoding of byte 32 (space).
private let kMinimalJSON = """
{
  "model": {
    "type": "BPE",
    "vocab": {
      "H": 10, "e": 11, "l": 12, "o": 13, "d": 14,
      "r": 15, "w": 16, "s": 17, "y": 18,
      "He": 100, "Hel": 101, "Hell": 103, "Hello": 102,
      "\u{0120}": 200, "world": 201, "\u{0120}world": 202
    },
    "merges": [
      "H e",
      "He l",
      "Hel l",
      "Hell o",
      "\u{0120} world"
    ]
  },
  "added_tokens": [
    {"id": 151643, "content": "<|endoftext|>", "special": true},
    {"id": 151644, "content": "<|im_start|>", "special": true},
    {"id": 151645, "content": "<|im_end|>", "special": true}
  ]
}
""".data(using: .utf8)!

// MARK: - Tests

final class BPETokenizerTests: XCTestCase {

    private var tok: BPETokenizer!

    override func setUpWithError() throws {
        try super.setUpWithError()
        tok = try BPETokenizer(data: kMinimalJSON)
    }

    // MARK: - EOS token

    func testEOSTokenID() {
        XCTAssertEqual(tok.eosTokenID, 151645)
    }

    // MARK: - Special token encoding (critical: these drive ChatML framing)

    func testEncodeImStart() throws {
        XCTAssertEqual(tok.encode("<|im_start|>"), [151644],
                       "<|im_start|> must encode to token 151644")
    }

    func testEncodeImEnd() throws {
        XCTAssertEqual(tok.encode("<|im_end|>"), [151645],
                       "<|im_end|> must encode to token 151645")
    }

    func testEncodeEndOfText() throws {
        XCTAssertEqual(tok.encode("<|endoftext|>"), [151643])
    }

    // MARK: - Special token decoding

    func testDecodeSpecialTokensReturnEmpty() {
        XCTAssertEqual(tok.decode(tokenID: 151644), "",
                       "Special tokens must decode to empty string (not emitted to UI)")
        XCTAssertEqual(tok.decode(tokenID: 151645), "")
        XCTAssertEqual(tok.decode(tokenID: 151643), "")
    }

    func testDecodeUnknownTokenReturnsEmpty() {
        XCTAssertEqual(tok.decode(tokenID: 99999), "")
    }

    // MARK: - BPE merge algorithm

    func testEncodeSingleWordProducesCorrectID() throws {
        // "Hello" → [102] after applying 4 merges from the minimal vocab
        XCTAssertEqual(tok.encode("Hello"), [102])
    }

    func testDecodeIDProducesCorrectString() {
        XCTAssertEqual(tok.decode(tokenID: 102), "Hello")
    }

    func testRoundTripASCII() throws {
        let text = "Hello"
        let ids = tok.encode(text)
        let decoded = ids.map { tok.decode(tokenID: $0) }.joined()
        XCTAssertEqual(decoded, text)
    }

    // MARK: - Special token split within a larger string

    func testSpecialTokenAtStartOfContext() throws {
        // "<|im_start|>Hello" → [151644, 102]
        let ids = tok.encode("<|im_start|>Hello")
        XCTAssertEqual(ids.first, 151644, "First token must be <|im_start|>")
        XCTAssertEqual(ids.last, 102,   "Last token must be the 'Hello' BPE token")
        XCTAssertEqual(ids.count, 2)
    }

    func testSpecialTokenBetweenWords() throws {
        let ids = tok.encode("Hello<|im_end|>")
        XCTAssertEqual(ids, [102, 151645])
    }

    // MARK: - Real tokenizer integration (skipped when tokenizer.json is absent)

    func testRealTokenizerSpecialTokens() throws {
        guard let url = Bundle.main.url(forResource: "tokenizer", withExtension: "json"),
              let real = try? BPETokenizer(url: url),
              !real.encode("Hello").isEmpty   // skip if placeholder has empty vocab
        else {
            throw XCTSkip("Real tokenizer.json not present — run build-ipa CI to inject it")
        }
        XCTAssertEqual(real.encode("<|im_start|>"), [151644])
        XCTAssertEqual(real.encode("<|im_end|>"),   [151645])
        XCTAssertEqual(real.eosTokenID, 151645)
    }

    func testRealTokenizerRoundTripASCII() throws {
        guard let url = Bundle.main.url(forResource: "tokenizer", withExtension: "json"),
              let real = try? BPETokenizer(url: url),
              !real.encode("Hello").isEmpty
        else {
            throw XCTSkip("Real tokenizer.json not present — run build-ipa CI to inject it")
        }
        let text = "The Hylian Shield is found in Hyrule Castle."
        let ids = real.encode(text)
        XCTAssertFalse(ids.isEmpty, "encode must produce tokens for non-empty ASCII text")
        let decoded = ids.map { real.decode(tokenID: $0) }.joined()
        XCTAssertEqual(decoded, text, "round-trip encode→decode must reproduce the original text")
    }

    func testRealTokenizerMultiByte() throws {
        guard let url = Bundle.main.url(forResource: "tokenizer", withExtension: "json"),
              let real = try? BPETokenizer(url: url),
              !real.encode("Hello").isEmpty
        else {
            throw XCTSkip("Real tokenizer.json not present — run build-ipa CI to inject it")
        }
        // "café" contains é (U+00E9), a two-byte UTF-8 sequence (0xC3 0xA9)
        let text = "café"
        let ids = real.encode(text)
        XCTAssertFalse(ids.isEmpty, "encode must produce tokens for text with multi-byte chars")
        let decoded = ids.map { real.decode(tokenID: $0) }.joined()
        XCTAssertEqual(decoded, text, "multi-byte UTF-8 must round-trip through encode/decode")
    }
}
