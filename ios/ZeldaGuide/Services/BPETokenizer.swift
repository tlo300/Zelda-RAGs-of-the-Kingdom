// BPETokenizer.swift
// Reads tokenizer.json (HuggingFace BPE format) and implements Qwen2.5-compatible
// byte-level BPE encode and decode. Replaces the BundledTokenizer placeholder.

import Foundation

/// Reads tokenizer.json from the app bundle and performs Qwen2.5 BPE encode/decode.
/// Conforms to LLMTokenizer for injection into LLMService.
struct BPETokenizer: LLMTokenizer {

    // MARK: - Stored properties

    private let vocab: [String: Int32]          // byte-level unicode string → token ID
    private let idToToken: [Int32: String]      // token ID → byte-level unicode string
    private let mergeRank: [String: Int]        // "a b" merge rule → priority (lower = applied first)
    private let specialContent: [String: Int32] // e.g. "<|im_start|>" → 151644
    private let specialTokenIDs: Set<Int32>     // fast lookup for decode
    private let byteToUnicode: [UInt8: Unicode.Scalar]
    private let unicodeToByte: [Unicode.Scalar: UInt8]

    // GPT-4 / tiktoken cl100k_base split pattern used by Qwen2.5's ByteLevel pre-tokenizer.
    // Compiled once at class load; crash-on-failure is intentional — invalid regex is a bug.
    private static let prePattern: NSRegularExpression = {
        let apos = "\u{2019}"  // RIGHT SINGLE QUOTATION MARK (U+2019)
        let pat = "(?i:\(apos)s|\(apos)t|\(apos)re|\(apos)ve|\(apos)m|\(apos)ll|\(apos)d)"
                + "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
                + "|\\p{N}{1,3}"
                + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
                + "|\\s*[\\r\\n]+"
                + "|\\s+(?!\\S)"
                + "|\\s+"
        return try! NSRegularExpression(pattern: pat, options: [])
    }()

    // MARK: - LLMTokenizer

    var eosTokenID: Int32 { 151645 }  // <|im_end|>

    /// Encodes `text` to Qwen2.5 token IDs, truncated to `ModelConfig.maxContextTokens`.
    func encode(_ text: String) -> [Int32] {
        var result = [Int32]()
        for seg in splitOnSpecialTokens(text) {
            if seg.isSpecial {
                if let id = specialContent[seg.text] { result.append(id) }
            } else {
                result.append(contentsOf: encodeRegular(seg.text))
            }
        }
        return Array(result.prefix(ModelConfig.maxContextTokens))
    }

    /// Returns the UTF-8 text for a single BPE token ID, or `""` for special/unknown tokens.
    func decode(tokenID: Int32) -> String {
        guard !specialTokenIDs.contains(tokenID) else { return "" }
        guard let tokenStr = idToToken[tokenID] else { return "" }
        var bytes = [UInt8]()
        for scalar in tokenStr.unicodeScalars {
            if let byte = unicodeToByte[scalar] { bytes.append(byte) }
        }
        return String(bytes: bytes, encoding: .utf8) ?? ""
    }

    // MARK: - Init

    init(url: URL) throws {
        try self.init(data: try Data(contentsOf: url))
    }

    init(data: Data) throws {
        let json = try JSONDecoder().decode(TokenizerJSON.self, from: data)

        vocab = json.model.vocab.mapValues { Int32($0) }

        var idMap = [Int32: String]()
        for (token, id) in vocab { idMap[id] = token }
        idToToken = idMap

        var rankMap = [String: Int]()
        for (i, merge) in json.model.merges.enumerated() { rankMap[merge] = i }
        mergeRank = rankMap

        var contentMap = [String: Int32]()
        var specialSet = Set<Int32>()
        for tok in json.addedTokens where tok.special == true {
            let id = Int32(tok.id)
            contentMap[tok.content] = id
            specialSet.insert(id)
        }
        specialContent = contentMap
        specialTokenIDs = specialSet

        (byteToUnicode, unicodeToByte) = Self.buildByteUnicode()
    }

    // MARK: - Private: encode

    private struct Segment { let text: String; let isSpecial: Bool }

    /// Splits `text` into alternating special-token and regular-text segments.
    private func splitOnSpecialTokens(_ text: String) -> [Segment] {
        guard !specialContent.isEmpty else {
            return [Segment(text: text, isSpecial: false)]
        }
        var segments = [Segment]()
        var remaining = text[...]
        while !remaining.isEmpty {
            var earliest: (range: Range<String.Index>, content: String)?
            for content in specialContent.keys {
                if let range = remaining.range(of: content) {
                    if earliest == nil || range.lowerBound < earliest!.range.lowerBound {
                        earliest = (range, content)
                    }
                }
            }
            guard let found = earliest else {
                segments.append(Segment(text: String(remaining), isSpecial: false))
                break
            }
            if found.range.lowerBound > remaining.startIndex {
                segments.append(Segment(
                    text: String(remaining[remaining.startIndex..<found.range.lowerBound]),
                    isSpecial: false))
            }
            segments.append(Segment(text: found.content, isSpecial: true))
            remaining = remaining[found.range.upperBound...]
        }
        return segments
    }

    /// Encodes a regular (non-special) text segment using ByteLevel BPE.
    private func encodeRegular(_ text: String) -> [Int32] {
        guard !text.isEmpty else { return [] }
        var result = [Int32]()
        for preToken in preSplit(text) {
            let bytes = Array(preToken.utf8)
            var symbols: [String] = bytes.compactMap { byteToUnicode[$0].map { String($0) } }
            guard !symbols.isEmpty else { continue }
            symbols = applyMerges(symbols)
            result.append(contentsOf: symbols.compactMap { vocab[$0] })
        }
        return result
    }

    /// Splits `text` using the ByteLevel pre-tokenizer regex.
    private func preSplit(_ text: String) -> [String] {
        let nsText = text as NSString
        let range = NSRange(location: 0, length: nsText.length)
        return Self.prePattern.matches(in: text, options: [], range: range)
            .map { nsText.substring(with: $0.range) }
    }

    /// Applies BPE merge rules to a symbol list until no more merges apply.
    private func applyMerges(_ input: [String]) -> [String] {
        var syms = input
        while syms.count > 1 {
            var bestRank = Int.max
            var bestIdx = -1
            for i in 0..<(syms.count - 1) {
                if let rank = mergeRank[syms[i] + " " + syms[i + 1]], rank < bestRank {
                    bestRank = rank; bestIdx = i
                }
            }
            guard bestIdx >= 0 else { break }
            syms[bestIdx] += syms[bestIdx + 1]
            syms.remove(at: bestIdx + 1)
        }
        return syms
    }

    // MARK: - Private: byte↔unicode

    /// Builds the GPT-2 byte_to_unicode mapping and its inverse.
    /// Bytes 33–126, 161–172, 174–255 map to their own Unicode scalar.
    /// The remaining 68 bytes map to U+0100 upward (in 0…255 iteration order).
    private static func buildByteUnicode() -> ([UInt8: Unicode.Scalar], [Unicode.Scalar: UInt8]) {
        var b2u = [UInt8: Unicode.Scalar]()
        var u2b = [Unicode.Scalar: UInt8]()
        var nextScalar: UInt32 = 256  // U+0100 (Ā)
        for bi in 0..<256 {
            let b = UInt8(bi)
            let direct = (33...126 ~= bi) || (161...172 ~= bi) || (174...255 ~= bi)
            let scalar: Unicode.Scalar = direct
                ? Unicode.Scalar(UInt32(bi))!
                : { let s = Unicode.Scalar(nextScalar)!; nextScalar += 1; return s }()
            b2u[b] = scalar
            u2b[scalar] = b
        }
        return (b2u, u2b)
    }
}

// MARK: - JSON model (private)

private struct TokenizerJSON: Decodable {
    struct Model: Decodable {
        let vocab: [String: Int]
        // HuggingFace tokenizers can store merges as either ["a b", ...] or [["a","b"], ...].
        // Decode as [[String]] and join with a space to normalise both formats.
        let merges: [String]

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            vocab = try c.decode([String: Int].self, forKey: .vocab)
            if let pairs = try? c.decode([[String]].self, forKey: .merges) {
                merges = pairs.map { $0.joined(separator: " ") }
            } else {
                merges = try c.decode([String].self, forKey: .merges)
            }
        }

        enum CodingKeys: String, CodingKey { case vocab, merges }
    }
    struct AddedToken: Decodable {
        let id: Int
        let content: String
        let special: Bool?
    }
    let model: Model
    let addedTokens: [AddedToken]

    enum CodingKeys: String, CodingKey {
        case model
        case addedTokens = "added_tokens"
    }
}
