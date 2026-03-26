// WordPieceTokenizerTests.swift
// XCTest unit tests for WordPieceTokenizer.
// Minimal-vocab tests run without any bundle files.
// Integration tests that require the real vocab.txt are skipped when it is absent.

import XCTest
@testable import ZeldaGuide

// MARK: - Minimal vocab helper

/// 20-token vocab covering "hello", "world", and punctuation for unit tests.
/// Line index equals the token ID, matching the vocab.txt format.
private let kMinimalVocabLines: [String] = [
    "[PAD]",       // 0
    "[UNK]",       // 1
    "[CLS]",       // 2  (overridden by WordPieceTokenizer static constants to 101/102/100)
    "[SEP]",       // 3
    "hello",       // 4
    "world",       // 5
    "korok",       // 6
    "seed",        // 7
    "hylian",      // 8
    "shield",      // 9
    "##lo",        // 10
    "hel",         // 11
    "##ld",        // 12
    "##ian",       // 13
    "hy",          // 14
    "##l",         // 15
    "[MASK]",      // 16
    "the",         // 17
    "a",           // 18
    "is",          // 19
]

// MARK: - Tests

final class WordPieceTokenizerTests: XCTestCase {

    private var tok: WordPieceTokenizer!

    override func setUpWithError() throws {
        try super.setUpWithError()
        tok = WordPieceTokenizer(lines: kMinimalVocabLines)
    }

    // MARK: - Special token IDs are constants, not vocab-dependent

    func testCLSIDIsAlways101() {
        XCTAssertEqual(WordPieceTokenizer.clsID, 101)
    }

    func testSEPIDIsAlways102() {
        XCTAssertEqual(WordPieceTokenizer.sepID, 102)
    }

    func testUNKIDIsAlways100() {
        XCTAssertEqual(WordPieceTokenizer.unkID, 100)
    }

    // MARK: - Output shape contract

    func testEncodeAlwaysReturnsMaxLength() {
        let cases = ["", "hello", "hello world", String(repeating: "hello ", count: 50)]
        for text in cases {
            let (ids, mask) = tok.encode(text)
            XCTAssertEqual(ids.count, 128, "ids must always be length 128 for: '\(text.prefix(20))'")
            XCTAssertEqual(mask.count, 128, "mask must always be length 128 for: '\(text.prefix(20))'")
        }
    }

    func testFirstTokenIsAlwaysCLS() {
        for text in ["", "hello", "hello world"] {
            let (ids, _) = tok.encode(text)
            XCTAssertEqual(ids[0], WordPieceTokenizer.clsID,
                           "First token must be [CLS]=101 for '\(text)'")
        }
    }

    func testCLSMaskIsAlways1() {
        let (_, mask) = tok.encode("")
        XCTAssertEqual(mask[0], 1, "[CLS] must have mask=1 even for empty input")
    }

    // MARK: - Empty text

    func testEmptyTextGivesCLSSEPThenPadding() {
        let (ids, mask) = tok.encode("")
        XCTAssertEqual(ids[0], WordPieceTokenizer.clsID,  "ids[0] must be [CLS]")
        XCTAssertEqual(ids[1], WordPieceTokenizer.sepID,  "ids[1] must be [SEP]")
        XCTAssertEqual(mask[0], 1, "mask[0] must be 1 ([CLS])")
        XCTAssertEqual(mask[1], 1, "mask[1] must be 1 ([SEP])")
        for i in 2..<128 {
            XCTAssertEqual(ids[i], 0,  "ids[\(i)] must be 0 (padding)")
            XCTAssertEqual(mask[i], 0, "mask[\(i)] must be 0 (padding)")
        }
    }

    // MARK: - Whole-word lookup

    func testKnownWordProducesExpectedTokenID() {
        // "hello" is at index 4 in the minimal vocab
        let (ids, _) = tok.encode("hello")
        XCTAssertEqual(ids[1], 4, "ids[1] must be the 'hello' token (ID 4 in minimal vocab)")
        XCTAssertEqual(ids[2], WordPieceTokenizer.sepID, "ids[2] must be [SEP]")
    }

    func testMultiWordInput() {
        // "hello world" → CLS, hello(4), world(5), SEP, padding
        let (ids, mask) = tok.encode("hello world")
        XCTAssertEqual(ids[0], WordPieceTokenizer.clsID)
        XCTAssertEqual(ids[1], 4,  "hello = 4")
        XCTAssertEqual(ids[2], 5,  "world = 5")
        XCTAssertEqual(ids[3], WordPieceTokenizer.sepID)
        XCTAssertEqual(mask[1], 1)
        XCTAssertEqual(mask[2], 1)
        XCTAssertEqual(mask[4], 0, "padding after SEP must be 0")
    }

    // MARK: - Case folding

    func testUppercaseInputIsLowercased() {
        let lower = tok.encode("hello").ids
        let upper = tok.encode("HELLO").ids
        XCTAssertEqual(lower, upper, "Encoding must be case-insensitive")
    }

    // MARK: - WordPiece subword segmentation

    func testWordPieceSubwordLookup() {
        // "hylian" is in minimal vocab as whole word (ID 8) even though "hy"+"##l"+"##ian" also exist.
        // WordPiece tries longest prefix first, so it matches "hylian" directly.
        let (ids, _) = tok.encode("hylian")
        XCTAssertEqual(ids[1], 8, "hylian must match as a single token (longest prefix wins)")
    }

    // MARK: - Unknown token

    func testUnknownWordProducesUnkID() {
        // "xyz" is not in the minimal vocab and has no valid subword decomposition either.
        let (ids, _) = tok.encode("xyz")
        XCTAssertEqual(ids[1], WordPieceTokenizer.unkID, "Unknown word must produce [UNK]=100")
    }

    // MARK: - Truncation

    func testLongInputTruncatedTo126Subwords() {
        // 200 repetitions of "hello " → only 126 tokens (128 - 2 for CLS/SEP) should be kept.
        let longText = String(repeating: "hello ", count: 200)
        let (ids, mask) = tok.encode(longText)
        XCTAssertEqual(ids.count, 128, "Output must always be 128 tokens")
        // Position 127 is the last position — must be padding (SEP is at position 127 or earlier)
        // SEP must appear somewhere before position 127 (the truncation boundary)
        let sepIdx = ids.firstIndex(of: WordPieceTokenizer.sepID)
        XCTAssertNotNil(sepIdx, "[SEP] must always appear in the output")
        XCTAssertLessThanOrEqual(sepIdx!, 127, "[SEP] must be within the 128-token window")
        XCTAssertEqual(ids[0], WordPieceTokenizer.clsID, "First token must still be [CLS]")
        // All positions after SEP must be padding
        if let sep = sepIdx {
            for i in (sep + 1)..<128 {
                XCTAssertEqual(ids[i], 0, "ids[\(i)] after [SEP] must be 0 (padding)")
                XCTAssertEqual(mask[i], 0, "mask[\(i)] after [SEP] must be 0 (padding)")
            }
        }
    }

    // MARK: - Attention mask consistency

    func testMaskMatchesRealTokenPositions() {
        // "hello world" has 4 real tokens: CLS, hello, world, SEP
        let (_, mask) = tok.encode("hello world")
        XCTAssertEqual(mask[0], 1)
        XCTAssertEqual(mask[1], 1)
        XCTAssertEqual(mask[2], 1)
        XCTAssertEqual(mask[3], 1)
        for i in 4..<128 {
            XCTAssertEqual(mask[i], 0, "mask[\(i)] should be 0 (padding)")
        }
    }

    // MARK: - Real vocab.txt integration tests (skipped when vocab.txt is absent)

    func testRealVocabKorokSeed() throws {
        guard let url = Bundle.main.url(forResource: "vocab", withExtension: "txt"),
              let real = try? WordPieceTokenizer(url: url)
        else {
            throw XCTSkip("vocab.txt not present in bundle — add it to Resources to enable this test")
        }
        // "korok seed" → lowercase: "korok seed"
        // WordPiece "korok" → ko (12849) + ##rok (27923)
        // WordPiece "seed"  → seed (6534)
        // Full sequence: [CLS=101, 12849, 27923, 6534, SEP=102, 0, 0, ...]
        let (ids, mask) = real.encode("korok seed")
        XCTAssertEqual(ids[0], 101,   "ids[0] must be [CLS]=101")
        XCTAssertEqual(ids[1], 12849, "ids[1] must be 'ko' (12849)")
        XCTAssertEqual(ids[2], 27923, "ids[2] must be '##rok' (27923)")
        XCTAssertEqual(ids[3], 6534,  "ids[3] must be 'seed' (6534)")
        XCTAssertEqual(ids[4], 102,   "ids[4] must be [SEP]=102")
        XCTAssertEqual(mask[0], 1)
        XCTAssertEqual(mask[1], 1)
        XCTAssertEqual(mask[2], 1)
        XCTAssertEqual(mask[3], 1)
        XCTAssertEqual(mask[4], 1)
        for i in 5..<128 {
            XCTAssertEqual(ids[i], 0,  "ids[\(i)] must be padding (0)")
            XCTAssertEqual(mask[i], 0, "mask[\(i)] must be padding (0)")
        }
    }

    func testRealVocabKnownEnglishWords() throws {
        guard let url = Bundle.main.url(forResource: "vocab", withExtension: "txt"),
              let real = try? WordPieceTokenizer(url: url)
        else {
            throw XCTSkip("vocab.txt not present in bundle")
        }
        // Common English words must all be in the BERT vocabulary as single tokens.
        for word in ["shield", "sword", "shrine", "quest", "map", "link", "zelda"] {
            let (ids, _) = real.encode(word)
            XCTAssertNotEqual(ids[1], WordPieceTokenizer.unkID,
                              "'\(word)' must not tokenize to [UNK]")
        }
    }

    func testRealVocabPunctuationIsSplit() throws {
        guard let url = Bundle.main.url(forResource: "vocab", withExtension: "txt"),
              let real = try? WordPieceTokenizer(url: url)
        else {
            throw XCTSkip("vocab.txt not present in bundle")
        }
        // "hylian shield?" — the "?" should be split off and become a separate token.
        let withPunct = real.encode("hylian shield?").ids
        let withoutPunct = real.encode("hylian shield").ids
        XCTAssertNotEqual(withPunct, withoutPunct,
                          "Punctuation must produce different tokenization from plain text")
    }
}
