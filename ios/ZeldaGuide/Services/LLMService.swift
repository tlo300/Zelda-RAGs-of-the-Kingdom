// LLMService.swift
// Loads the Core ML Qwen2.5 LLM and generates streaming text output via AsyncStream<String>.
// All model filenames and generation parameters come from ModelConfig — never hardcoded.
//
// Generation uses a KV-cache for O(n) decode performance:
//   - Prefill:  one forward pass over the full prompt → returns logits + initial KV-cache
//   - Decode:   one token per step using the cached KV state → ~200x faster than full-sequence

import CoreML
import Foundation
import UIKit
import os

private let llmLog = Logger(subsystem: "com.tlo300.ZeldaGuide", category: "LLMService")

// MARK: - Errors

enum LLMError: Error, LocalizedError {
    case modelNotFound(String)
    case loadFailed(Error)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "LLM '\(name)' not found in the app bundle. " +
                   "Ensure the .mlpackage was placed in Resources before building."
        case .loadFailed(let error):
            return "Failed to load LLM: \(error.localizedDescription)"
        }
    }
}

// MARK: - Protocols (injectable for testing)

/// Encodes a prompt string to token IDs and decodes a single token ID to a text fragment.
protocol LLMTokenizer: Sendable {
    /// Returns token IDs for the given text (truncated to maxContextTokens if needed).
    func encode(_ text: String) -> [Int32]
    /// Returns the text fragment for a single token ID, or an empty string for unknown tokens.
    func decode(tokenID: Int32) -> String
    /// Token ID that signals end-of-sequence.
    var eosTokenID: Int32 { get }
}

/// Runs a forward pass of the language model using the KV-cache interface.
/// - `inputIDs`:  token IDs to process (full prompt for prefill, [nextToken] for decode)
/// - `pastKV`:    accumulated KV-cache from previous calls; nil on the very first call
/// Returns `(logits, presentKV)` where `presentKV` is the updated cache to pass next time.
protocol LLMPredictor: Sendable {
    func predict(inputIDs: [Int32], pastKV: MLMultiArray?) async throws -> ([Float], MLMultiArray)
}

// MARK: - Production implementation

/// Wraps a loaded `MLModel` to produce last-token logits via the KV-cache interface.
struct CoreMLPredictor: LLMPredictor {
    private let model: MLModel

    init(model: MLModel) { self.model = model }

    func predict(inputIDs: [Int32], pastKV: MLMultiArray?) async throws -> ([Float], MLMultiArray) {
        let seqLen  = inputIDs.count
        let pastLen = pastKV?.shape[3].intValue ?? 0
        let totalLen = pastLen + seqLen

        // input_ids: [1, seqLen]
        let inputArr = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .int32)
        for (i, id) in inputIDs.enumerated() { inputArr[i] = NSNumber(value: id) }

        // attention_mask: [1, totalLen] — all ones (attend to every past + current token)
        let maskArr = try MLMultiArray(shape: [1, NSNumber(value: totalLen)], dataType: .int32)
        for i in 0..<totalLen { maskArr[i] = 1 }

        // past_kv: [L, 2, H, pastLen, D] — empty on first call, accumulated thereafter
        let kvArr: MLMultiArray
        if let pastKV {
            kvArr = pastKV
        } else {
            kvArr = try MLMultiArray(
                shape: [
                    NSNumber(value: ModelConfig.llmNumLayers),
                    2,
                    NSNumber(value: ModelConfig.llmNumKVHeads),
                    0,
                    NSNumber(value: ModelConfig.llmHeadDim),
                ],
                dataType: .float16
            )
        }

        // position_ids: [1, seqLen] — RoPE positions [pastLen, pastLen+1, ..., totalLen-1]
        let posArr = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .int32)
        for i in 0..<seqLen { posArr[i] = NSNumber(value: pastLen + i) }

        let features = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids":      MLFeatureValue(multiArray: inputArr),
            "attention_mask": MLFeatureValue(multiArray: maskArr),
            "past_kv":        MLFeatureValue(multiArray: kvArr),
            "position_ids":   MLFeatureValue(multiArray: posArr),
        ])
        let result = try await model.prediction(from: features)

        // logits: [1, 1, vocab]
        guard let logitsMLA = result.featureValue(for: "logits")?.multiArrayValue else {
            throw LLMError.loadFailed(NSError(
                domain: "LLMService", code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Model output missing 'logits' key"]))
        }
        let vocabSize = logitsMLA.shape[2].intValue
        let logits    = (0..<vocabSize).map { logitsMLA[$0].floatValue }

        // present_kv: [L, 2, H, totalLen, D]
        guard let presentKV = result.featureValue(for: "present_kv")?.multiArrayValue else {
            throw LLMError.loadFailed(NSError(
                domain: "LLMService", code: -2,
                userInfo: [NSLocalizedDescriptionKey: "Model output missing 'present_kv' key"]))
        }

        return (logits, presentKV)
    }
}

// MARK: - LLMService

/// Loads the Core ML Qwen2.5 model and streams generated tokens via `AsyncStream<String>`.
actor LLMService {

    private var predictor: (any LLMPredictor)?
    private var tokenizer: (any LLMTokenizer)?
    private var currentGeneration: Task<Void, Never>?
    private var isMemoryPressured = false

    init() {}

    /// Designated initialiser for testing — injects a predictor and tokenizer directly.
    init(predictor: any LLMPredictor, tokenizer: any LLMTokenizer) {
        self.predictor = predictor
        self.tokenizer = tokenizer
    }

    // MARK: - Loading

    /// Loads the LLM from the app bundle asynchronously.
    /// Throws `LLMError.modelNotFound` with a clear, human-readable message if the
    /// .mlpackage is absent — never crashes or silently hangs.
    func load() async throws {
        guard predictor == nil else { return }

        let filename = ModelConfig.modelFilename
        let url      = URL(fileURLWithPath: filename)
        let baseName = url.deletingPathExtension().lastPathComponent
        let ext      = url.pathExtension

        guard let modelURL = Bundle.main.url(forResource: baseName, withExtension: ext) else {
            throw LLMError.modelNotFound(filename)
        }

        do {
            let config = MLModelConfiguration()
            // .all lets Core ML use the Neural Engine, which is required for grouped
            // palettization (4-bit per_grouped_channel) — the NE is the only runtime
            // that can decompress these weights.  .cpuAndGPU crashes with an uncatchable
            // NSException on device because the palettized weight decompression op has
            // no CPU/GPU fallback.  In the simulator there is no NE so Core ML falls
            // back to CPU automatically.
            config.computeUnits = .all
            let compiledURL = try await compileAndCache(modelURL)
            llmLog.notice("MLModel.load starting — this may take 30+ min in the simulator")
            let model = try await MLModel.load(contentsOf: compiledURL, configuration: config)
            llmLog.notice("MLModel.load complete")
            predictor = CoreMLPredictor(model: model)
        } catch let e as LLMError {
            llmLog.error("LLM load failed: \(e.localizedDescription, privacy: .public)")
            throw e
        } catch {
            llmLog.error("LLM load failed: \(error.localizedDescription, privacy: .public)")
            throw LLMError.loadFailed(error)
        }

        guard let tokURL = Bundle.main.url(forResource: "tokenizer", withExtension: "json") else {
            throw LLMError.modelNotFound("tokenizer.json")
        }
        do {
            llmLog.notice("Loading tokenizer")
            tokenizer = try BPETokenizer(url: tokURL)
            llmLog.notice("Tokenizer loaded")
        } catch {
            llmLog.error("Tokenizer load failed: \(error.localizedDescription, privacy: .public)")
            throw LLMError.loadFailed(error)
        }
        registerMemoryWarningHandler()
    }

    /// Returns a compiled .mlmodelc URL ready for MLModel.load.
    /// If the bundle already contains a pre-compiled .mlmodelc (CI builds), it is
    /// returned directly — no on-device compilation, no memory spike.
    /// Falls back to compiling a .mlpackage and caching the result for dev builds.
    private func compileAndCache(_ sourceURL: URL) async throws -> URL {
        if sourceURL.pathExtension == "mlmodelc" { return sourceURL }
        let fm = FileManager.default
        let cacheDir = fm.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CompiledModels", isDirectory: true)
        let cachedURL = cacheDir.appendingPathComponent(sourceURL.lastPathComponent + "c")
        if fm.fileExists(atPath: cachedURL.path) { return cachedURL }
        try fm.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        let tmpURL = try await MLModel.compileModel(at: sourceURL)
        try? fm.removeItem(at: cachedURL)
        try fm.moveItem(at: tmpURL, to: cachedURL)
        return cachedURL
    }

    // MARK: - Generation

    /// Streams token strings as the model generates them.
    /// Any in-flight generation is automatically cancelled before the new stream begins —
    /// concurrent generations never run simultaneously.
    func generate(prompt: String) -> AsyncStream<String> {
        currentGeneration?.cancel()
        currentGeneration = nil
        isMemoryPressured = false

        let (stream, continuation) = AsyncStream<String>.makeStream()

        currentGeneration = Task { [self] in
            await self.runGeneration(prompt: prompt, continuation: continuation)
        }

        return stream
    }

    // MARK: - Private

    private func runGeneration(
        prompt: String,
        continuation: AsyncStream<String>.Continuation
    ) async {
        defer { continuation.finish() }

        guard let predictor, let tokenizer else {
            continuation.yield("[Error: LLM not loaded — call load() before generate()]")
            return
        }

        let promptTokens = tokenizer.encode(ModelConfig.chatPrompt(userQuery: prompt))
        let eosID  = tokenizer.eosTokenID
        let maxNew = ModelConfig.maxOutputTokens

        llmLog.notice("Generation starting — prompt tokens: \(promptTokens.count, privacy: .public), maxNew: \(maxNew, privacy: .public)")

        // ── Prefill: process full prompt in one forward pass ──────────────────────
        // Returns logits for the last prompt token + initialises the KV cache so
        // every subsequent decode step only processes one new token.
        guard let (prefillLogits, kvCache) = try? await predictor.predict(
            inputIDs: promptTokens,
            pastKV:   nil
        ) else {
            continuation.yield("[Error: LLM prefill failed]")
            return
        }

        if Task.isCancelled { return }

        // Greedy pick of first generated token from prefill logits.
        var nextToken = argmax(prefillLogits)
        if isStopToken(nextToken, eosID: eosID) {
            llmLog.notice("Generation stopped — EOS on prefill")
            return
        }
        var tokenCount = 1
        let firstText  = tokenizer.decode(tokenID: nextToken)
        if !firstText.isEmpty { continuation.yield(firstText) }

        // ── Decode: one token per step using accumulated KV cache ─────────────────
        var currentKV: MLMultiArray = kvCache

        for _ in 1..<maxNew {
            // Guard: sequence length cap (RangeDim upper bound)
            let totalSoFar = promptTokens.count + tokenCount
            if totalSoFar >= ModelConfig.modelMaxSequenceLength {
                llmLog.notice("Generation stopped — sequence length limit (\(ModelConfig.modelMaxSequenceLength, privacy: .public)) reached after \(tokenCount, privacy: .public) tokens")
                break
            }

            // Yield actor so a concurrent generate() call can cancel this task.
            await Task.yield()
            if Task.isCancelled {
                llmLog.notice("Generation cancelled after \(tokenCount, privacy: .public) tokens")
                break
            }
            if isMemoryPressured {
                llmLog.notice("Generation stopped — memory pressure after \(tokenCount, privacy: .public) tokens")
                break
            }

            guard let (logits, newKV) = try? await predictor.predict(
                inputIDs: [nextToken],
                pastKV:   currentKV
            ) else {
                llmLog.error("Generation stopped — predictor failed after \(tokenCount, privacy: .public) tokens")
                break
            }

            if Task.isCancelled {
                llmLog.notice("Generation cancelled after \(tokenCount, privacy: .public) tokens")
                break
            }

            currentKV  = newKV
            nextToken  = argmax(logits)
            tokenCount += 1

            if isStopToken(nextToken, eosID: eosID) {
                llmLog.notice("Generation stopped — EOS token \(nextToken, privacy: .public) after \(tokenCount, privacy: .public) tokens")
                break
            }

            let text = tokenizer.decode(tokenID: nextToken)
            if !text.isEmpty { continuation.yield(text) }
        }
        llmLog.notice("Generation complete — \(tokenCount, privacy: .public) tokens yielded")
    }

    private func argmax(_ logits: [Float]) -> Int32 {
        var maxVal = Float(-Float.infinity)
        var maxIdx: Int32 = 0
        for (i, v) in logits.enumerated() {
            if v > maxVal { maxVal = v; maxIdx = Int32(i) }
        }
        return maxIdx
    }

    private func isStopToken(_ token: Int32, eosID: Int32) -> Bool {
        // <|im_end|> = 151645, <|endoftext|> = 151643, <|im_start|> = 151644
        token == eosID || token == 151643 || token == 151644
    }

    private func registerMemoryWarningHandler() {
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: nil
        ) { [weak self] _ in
            guard let self else { return }
            Task { await self.onMemoryWarning() }
        }
    }

    private func onMemoryWarning() {
        isMemoryPressured = true
        print("[LLMService] Memory warning received — generation will pause at next step")
    }
}
