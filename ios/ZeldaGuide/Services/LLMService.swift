// LLMService.swift
// Loads the Core ML Qwen2.5 LLM and generates streaming text output via AsyncStream<String>.
// All model filenames and generation parameters come from ModelConfig — never hardcoded.

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

/// Runs a single forward pass of the language model for one token.
protocol LLMPredictor: Sendable {
    /// Returns logits for the given single token ID, advancing the internal KV cache state.
    func predict(inputIDs: [Int32]) async throws -> [Float]
    /// Resets the KV cache state. Call at the start of each new generation.
    func reset()
}

// MARK: - Production implementations

/// Wraps a loaded `MLModel` to produce last-token logits using a fixed-size KV cache.
///
/// The model interface (from convert_llm.py) expects:
///   input_ids      [1, 1]                    — single new token
///   attention_mask [1, MAX_KV_LEN+1]         — 1 for valid past slots + current token
///   past_kv        [L, 2, H, MAX_KV_LEN, D]  — circular KV buffer
///   position_ids   [1, 1]                    — absolute position of the new token
/// and returns:
///   logits  [1, 1, vocab]
///   new_kv  [L, 2, H, 1, D]                 — KV for the new token (written into buffer)
///
/// All shapes are fixed — no RangeDim — which prevents the Jetsam OOM that killed R26-R31.
final class CoreMLPredictor: LLMPredictor, @unchecked Sendable {
    private let model: MLModel
    private let maxKVLen:   Int
    private let numLayers:  Int
    private let numKVHeads: Int
    private let headDim:    Int

    // Fixed-size circular KV buffer [L, 2, H, maxKVLen, D], Float16.
    private let kvBuffer: MLMultiArray
    // Pre-allocated per-call arrays to avoid repeated allocation on the hot path.
    private let maskArr:  MLMultiArray   // [1, maxKVLen+1], Int32
    private let inputArr: MLMultiArray   // [1, 1], Int32
    private let posArr:   MLMultiArray   // [1, 1], Int32

    private var writePos: Int = 0

    init(model: MLModel) throws {
        self.model      = model
        self.maxKVLen   = ModelConfig.llmMaxKVLen
        self.numLayers  = ModelConfig.llmNumLayers
        self.numKVHeads = ModelConfig.llmNumKVHeads
        self.headDim    = ModelConfig.llmHeadDim

        kvBuffer = try MLMultiArray(
            shape: [NSNumber(value: numLayers), 2,
                    NSNumber(value: numKVHeads), NSNumber(value: maxKVLen),
                    NSNumber(value: headDim)],
            dataType: .float16
        )
        maskArr  = try MLMultiArray(shape: [1, NSNumber(value: maxKVLen + 1)], dataType: .int32)
        inputArr = try MLMultiArray(shape: [1, 1], dataType: .int32)
        posArr   = try MLMultiArray(shape: [1, 1], dataType: .int32)

        // Zero-initialise the KV buffer.
        memset(kvBuffer.dataPointer, 0, kvBuffer.count * MemoryLayout<Float16>.size)
    }

    func reset() {
        writePos = 0
        memset(kvBuffer.dataPointer, 0, kvBuffer.count * MemoryLayout<Float16>.size)
    }

    func predict(inputIDs: [Int32]) async throws -> [Float] {
        guard let tokenID = inputIDs.last else {
            throw LLMError.loadFailed(NSError(
                domain: "LLMService", code: -1,
                userInfo: [NSLocalizedDescriptionKey: "predict() called with empty inputIDs"]))
        }

        let slot      = writePos % maxKVLen
        let validPast = min(writePos, maxKVLen)

        // Build attention mask: 1 for valid past slots, 0 for empty slots, 1 for current token.
        let maskBase = maskArr.dataPointer.assumingMemoryBound(to: Int32.self)
        memset(maskBase, 0, (maxKVLen + 1) * MemoryLayout<Int32>.size)
        for i in 0..<validPast { maskBase[i] = 1 }
        maskBase[maxKVLen] = 1  // current token

        inputArr[0] = NSNumber(value: tokenID)
        posArr[0]   = NSNumber(value: writePos)

        let features = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids":      MLFeatureValue(multiArray: inputArr),
            "attention_mask": MLFeatureValue(multiArray: maskArr),
            "past_kv":        MLFeatureValue(multiArray: kvBuffer),
            "position_ids":   MLFeatureValue(multiArray: posArr),
        ])
        let result = try await model.prediction(from: features)

        // Write new_kv [L, 2, H, 1, D] into the circular buffer at the current slot.
        guard let newKV = result.featureValue(for: "new_kv")?.multiArrayValue else {
            throw LLMError.loadFailed(NSError(
                domain: "LLMService", code: -2,
                userInfo: [NSLocalizedDescriptionKey: "Model output missing 'new_kv' key"]))
        }
        let srcBase = newKV.dataPointer.assumingMemoryBound(to: Float16.self)
        let dstBase = kvBuffer.dataPointer.assumingMemoryBound(to: Float16.self)
        let H = numKVHeads
        let D = headDim
        let M = maxKVLen
        let L = numLayers
        // new_kv  strides: [2*H*D,   H*D,   D, D, 1] for [L, 2, H, 1, D]
        // kvBuffer strides: [2*H*M*D, H*M*D, M*D, D, 1] for [L, 2, H, M, D]
        for l in 0..<L {
            for h2 in 0..<2 {
                for hh in 0..<H {
                    let srcOff = (l * 2 * H + h2 * H + hh) * D
                    let dstOff = (l * 2 * H * M + h2 * H * M + hh * M + slot) * D
                    memcpy(dstBase + dstOff, srcBase + srcOff, D * MemoryLayout<Float16>.size)
                }
            }
        }
        writePos += 1

        // Extract logits [1, 1, vocabSize] → flat [vocabSize].
        guard let mla = result.featureValue(for: "logits")?.multiArrayValue else {
            throw LLMError.loadFailed(NSError(
                domain: "LLMService", code: -3,
                userInfo: [NSLocalizedDescriptionKey: "Model output missing 'logits' key"]))
        }
        let vocabSize  = mla.shape[2].intValue
        let lastOffset = (mla.shape[1].intValue - 1) * vocabSize
        return (0..<vocabSize).map { mla[lastOffset + $0].floatValue }
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
            // cpuAndGPU matches the model's compilation target (ct.ComputeUnit.CPU_AND_GPU
            // in convert_llm.py). Fixed-shape KV cache means no RangeDim compilation spike,
            // so MLModel.load() no longer OOM-kills on device as it did in R26-R31.
            // NE will be re-enabled once NE-specific conversion time is under control.
            config.computeUnits = .cpuAndGPU
            let compiledURL = try await compileAndCache(modelURL)
            llmLog.notice("MLModel.load starting — this may take 30+ min in the simulator")
            let model = try await MLModel.load(contentsOf: compiledURL, configuration: config)
            llmLog.notice("MLModel.load complete")
            predictor = try CoreMLPredictor(model: model)
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

        predictor.reset()

        let promptTokens = tokenizer.encode(ModelConfig.chatPrompt(userQuery: prompt))
        let eosID  = tokenizer.eosTokenID
        let maxNew = ModelConfig.maxOutputTokens
        var tokenCount = 0

        llmLog.notice("Generation starting — prompt tokens: \(promptTokens.count, privacy: .public), maxNew: \(maxNew, privacy: .public)")

        // Prefill: run each prompt token through the model to populate the KV cache.
        // The last prefill call's logits seed the first decode step.
        var prefillLogits: [Float]? = nil
        for token in promptTokens {
            await Task.yield()
            if Task.isCancelled {
                llmLog.notice("Generation cancelled during prefill")
                return
            }
            prefillLogits = try? await predictor.predict(inputIDs: [token])
        }

        guard let firstLogits = prefillLogits else {
            llmLog.error("Generation stopped — prefill produced no output")
            return
        }

        // Decode: generate up to maxNew tokens, one per forward pass.
        var currentLogits = firstLogits
        for _ in 0..<maxNew {
            await Task.yield()
            if Task.isCancelled {
                llmLog.notice("Generation cancelled after \(tokenCount, privacy: .public) tokens")
                break
            }
            if isMemoryPressured {
                llmLog.notice("Generation stopped — memory pressure after \(tokenCount, privacy: .public) tokens")
                break
            }

            // Greedy decoding: pick the token with the highest logit.
            var maxLogit = Float(-Float.infinity)
            var nextToken: Int32 = 0
            for (i, v) in currentLogits.enumerated() {
                if v > maxLogit { maxLogit = v; nextToken = Int32(i) }
            }

            // Stop on <|im_end|> (151645), <|endoftext|> (151643), or <|im_start|> (151644).
            if nextToken == eosID || nextToken == 151643 || nextToken == 151644 {
                llmLog.notice("Generation stopped — EOS token \(nextToken, privacy: .public) after \(tokenCount, privacy: .public) tokens")
                break
            }

            tokenCount += 1
            let text = tokenizer.decode(tokenID: nextToken)
            if !text.isEmpty {
                continuation.yield(text)
            }

            guard let nextLogits = try? await predictor.predict(inputIDs: [nextToken]) else {
                llmLog.error("Generation stopped — predictor failed after \(tokenCount, privacy: .public) tokens")
                break
            }
            currentLogits = nextLogits
        }

        llmLog.notice("Generation complete — \(tokenCount, privacy: .public) tokens yielded")
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
