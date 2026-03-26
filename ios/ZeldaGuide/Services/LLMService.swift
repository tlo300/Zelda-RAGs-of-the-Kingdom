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

/// Runs a single forward pass of the language model.
protocol LLMPredictor: Sendable {
    /// Returns the last-token logit vector given the full accumulated token sequence.
    func predict(inputIDs: [Int32]) async throws -> [Float]
}

// MARK: - Production implementations

/// Wraps a loaded `MLModel` to produce last-token logits.
struct CoreMLPredictor: LLMPredictor {
    private let model: MLModel

    init(model: MLModel) { self.model = model }

    func predict(inputIDs: [Int32]) async throws -> [Float] {
        let seqLen = inputIDs.count
        let inputArr = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .int32)
        let maskArr  = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .int32)
        for i in 0..<seqLen {
            inputArr[i] = NSNumber(value: inputIDs[i])
            maskArr[i]  = 1
        }
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids":      MLFeatureValue(multiArray: inputArr),
            "attention_mask": MLFeatureValue(multiArray: maskArr),
        ])
        let result = try await model.prediction(from: features)
        guard let mla = result.featureValue(for: "logits")?.multiArrayValue else {
            throw LLMError.loadFailed(NSError(
                domain: "LLMService", code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Model output missing 'logits' key"]))
        }
        // Shape: [1, returnedSeqLen, vocabSize].
        // Some CoreML compilations return the full sequence; others return only the last
        // token ([1, 1, vocabSize]). Use mla.shape[1] (actual returned length) — NOT
        // the input seqLen — to avoid an out-of-bounds MLMultiArray access.
        let returnedSeqLen = mla.shape[1].intValue
        let vocabSize      = mla.shape[2].intValue
        let lastOffset     = (returnedSeqLen - 1) * vocabSize
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
            // cpuAndGPU avoids the NE runtime (which can raise uncatchable NSExceptions
            // on some device/iOS 18 combinations with grouped palettization models) while
            // also avoiding a hang that .cpuOnly exhibits in the iOS simulator with the
            // 2048-context model. In the simulator there is no GPU so this falls back to
            // CPU-only execution; on device it uses CPU+GPU.
            config.computeUnits = .cpuAndGPU
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

        var inputTokens = tokenizer.encode(ModelConfig.chatPrompt(userQuery: prompt))
        let eosID  = tokenizer.eosTokenID
        let maxNew = ModelConfig.maxOutputTokens
        var tokenCount = 0

        llmLog.notice("Generation starting — prompt tokens: \(inputTokens.count, privacy: .public), maxNew: \(maxNew, privacy: .public)")

        for _ in 0..<maxNew {
            // Stop before exceeding the model's RangeDim upper bound — passing more tokens
            // than the compiled max_context raises an uncatchable NSException.
            if inputTokens.count >= ModelConfig.modelMaxSequenceLength {
                llmLog.notice("Generation stopped — sequence length limit (\(ModelConfig.modelMaxSequenceLength, privacy: .public)) reached after \(tokenCount, privacy: .public) tokens")
                break
            }

            // Yield the actor so that a concurrent generate() call can cancel this task.
            await Task.yield()
            if Task.isCancelled {
                llmLog.notice("Generation cancelled after \(tokenCount, privacy: .public) tokens")
                break
            }

            if isMemoryPressured {
                llmLog.notice("Generation stopped — memory pressure after \(tokenCount, privacy: .public) tokens")
                break
            }

            guard let logits = try? await predictor.predict(inputIDs: inputTokens) else {
                llmLog.error("Generation stopped — predictor failed after \(tokenCount, privacy: .public) tokens")
                break
            }

            if Task.isCancelled {
                llmLog.notice("Generation cancelled after \(tokenCount, privacy: .public) tokens")
                break
            }

            // Greedy decoding: pick the token with the highest logit.
            var maxLogit = Float(-Float.infinity)
            var nextToken: Int32 = 0
            for (i, v) in logits.enumerated() {
                if v > maxLogit { maxLogit = v; nextToken = Int32(i) }
            }

            // Stop on <|im_end|> (151645), <|endoftext|> (151643), or <|im_start|> (151644).
            // Including im_start prevents the model from generating a fake new user/system turn
            // if it fails to emit im_end first.
            if nextToken == eosID || nextToken == 151643 || nextToken == 151644 {
                llmLog.notice("Generation stopped — EOS token \(nextToken, privacy: .public) after \(tokenCount, privacy: .public) tokens")
                break
            }
            inputTokens.append(nextToken)
            tokenCount += 1

            let text = tokenizer.decode(tokenID: nextToken)
            if !text.isEmpty {
                continuation.yield(text)
            }
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
