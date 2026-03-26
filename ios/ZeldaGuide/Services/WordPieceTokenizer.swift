// WordPieceTokenizer.swift
// BERT WordPiece tokenizer for on-device query embedding.
// Encodes query text to token IDs compatible with all-MiniLM-L6-v2 (bert-base-uncased vocab).

import Foundation

/// Encodes query strings to BERT WordPiece token IDs: [CLS] + subwords + [SEP] + zero-padding.
/// Vocab is loaded from vocab.txt (bert-base-uncased, 30 522 tokens) in the app bundle.
struct WordPieceTokenizer {

    /// Standard BERT special-token IDs (bert-base-uncased vocab, same for all BERT-family models).
    static let clsID: Int32 = 101
    static let sepID: Int32 = 102
    static let unkID: Int32 = 100

    private let vocab: [String: Int32]

    // MARK: - Init

    /// Loads vocab.txt from the main bundle. Throws `WordPieceError.vocabNotFound` if absent.
    init() throws {
        guard let url = Bundle.main.url(forResource: "vocab", withExtension: "txt") else {
            throw WordPieceError.vocabNotFound
        }
        try self.init(url: url)
    }

    init(url: URL) throws {
        let text = try String(contentsOf: url, encoding: .utf8)
        self.init(lines: text.components(separatedBy: "\n"))
    }

    /// Creates a tokenizer from an in-memory list of vocab lines.
    /// Line index equals the token ID (0-based), matching the vocab.txt format.
    init(lines: [String]) {
        var v = [String: Int32]()
        v.reserveCapacity(lines.count)
        for (i, line) in lines.enumerated() {
            let token = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if !token.isEmpty { v[token] = Int32(i) }
        }
        vocab = v
    }

    // MARK: - Public API

    /// Encodes `text` and returns exactly `maxLength` token IDs and a matching attention mask.
    /// Layout: [CLS] + WordPiece subwords (truncated) + [SEP] + zero-padding.
    /// Attention mask is 1 for real tokens (including CLS/SEP), 0 for padding.
    func encode(_ text: String, maxLength: Int = 128) -> (ids: [Int32], mask: [Int32]) {
        let tokens = tokenize(text)
        // Reserve 2 positions for CLS and SEP.
        let truncated = Array(tokens.prefix(maxLength - 2))

        var ids = [Int32]()
        ids.reserveCapacity(maxLength)
        ids.append(Self.clsID)
        ids.append(contentsOf: truncated)
        ids.append(Self.sepID)
        let realLen = ids.count

        while ids.count < maxLength { ids.append(0) }

        let mask = (0..<maxLength).map { $0 < realLen ? Int32(1) : Int32(0) }
        return (ids, mask)
    }

    // MARK: - Private

    private func tokenize(_ text: String) -> [Int32] {
        var result = [Int32]()
        for word in basicTokenize(text) {
            result.append(contentsOf: wordPiece(word))
        }
        return result
    }

    /// Lowercases text, inserts spaces around punctuation and Chinese characters, splits on whitespace.
    /// This mirrors BERT's BasicTokenizer (do_lower_case=True, tokenize_chinese_chars=True).
    private func basicTokenize(_ text: String) -> [String] {
        var buf = ""
        buf.reserveCapacity(text.count * 2)
        for scalar in text.unicodeScalars {
            if isPunctuation(scalar) || isChinese(scalar) {
                buf.append(" ")
                buf.unicodeScalars.append(scalar)
                buf.append(" ")
            } else {
                buf.unicodeScalars.append(scalar)
            }
        }
        return buf.lowercased()
            .components(separatedBy: .whitespaces)
            .filter { !$0.isEmpty }
    }

    /// Applies WordPiece subword segmentation to a single pre-tokenized word.
    /// Returns `[unkID]` if any character cannot be matched by any subword in the vocabulary.
    private func wordPiece(_ word: String) -> [Int32] {
        guard !word.isEmpty else { return [] }
        if let id = vocab[word] { return [id] }

        var result = [Int32]()
        var startIdx = word.startIndex
        var isFirst = true

        while startIdx < word.endIndex {
            var endIdx = word.endIndex
            var matched = false

            while endIdx > startIdx {
                let slice = word[startIdx..<endIdx]
                let candidate = isFirst ? String(slice) : "##" + slice
                if let id = vocab[candidate] {
                    result.append(id)
                    startIdx = endIdx
                    isFirst = false
                    matched = true
                    break
                }
                endIdx = word.index(before: endIdx)
            }

            if !matched { return [Self.unkID] }
        }

        return result
    }

    /// Returns true for ASCII punctuation ranges and Unicode punctuation characters.
    /// Matches BERT's is_punctuation() implementation.
    private func isPunctuation(_ scalar: Unicode.Scalar) -> Bool {
        let cp = scalar.value
        if (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
           (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126) { return true }
        return CharacterSet.punctuationCharacters.contains(scalar)
    }

    /// Returns true for CJK Unified Ideographs and CJK Compatibility Ideographs.
    private func isChinese(_ scalar: Unicode.Scalar) -> Bool {
        let cp = scalar.value
        return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
               (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0xF900 && cp <= 0xFAFF)
    }
}

// MARK: - Error

enum WordPieceError: Error, LocalizedError {
    case vocabNotFound

    var errorDescription: String? {
        "vocab.txt not found in the app bundle — required for on-device query embedding"
    }
}
