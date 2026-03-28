// SearchEngine.swift
// Embeds a query using the on-device MiniLM Core ML model and retrieves relevant
// knowledge chunks from knowledge_base.db. No LLM step — results are returned
// directly so the UI can display them instantly.

import CoreML
import Foundation

// MARK: - Errors

enum SearchEngineError: Error, LocalizedError {
    case embeddingModelNotFound(String)
    case embeddingLoadFailed(Error)
    case invalidEmbedding

    var errorDescription: String? {
        switch self {
        case .embeddingModelNotFound(let name):
            return "Embedding model '\(name)' not found in the app bundle."
        case .embeddingLoadFailed(let error):
            return "Failed to load embedding model: \(error.localizedDescription)"
        case .invalidEmbedding:
            return "Query embedding contains NaN — try a different query."
        }
    }
}

// MARK: - On-device embedding (all-MiniLM-L6-v2)

private actor CoreMLEmbeddingService {

    private var model: MLModel?
    private var tokenizer: WordPieceTokenizer?

    private func ensureLoaded() async throws {
        guard model == nil else { return }
        let filename = ModelConfig.embeddingModelFilename
        let url = URL(fileURLWithPath: filename)
        let baseName = url.deletingPathExtension().lastPathComponent
        let ext = url.pathExtension
        guard let modelURL = Bundle.main.url(forResource: baseName, withExtension: ext) else {
            throw SearchEngineError.embeddingModelNotFound(filename)
        }
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly
            let compiledURL = try await compileAndCache(modelURL)
            model = try await MLModel.load(contentsOf: compiledURL, configuration: config)
            tokenizer = try WordPieceTokenizer()
        } catch {
            throw SearchEngineError.embeddingLoadFailed(error)
        }
    }

    /// Returns a compiled .mlmodelc URL ready for MLModel.load.
    /// Pre-compiled .mlmodelc bundles (CI builds) are returned directly.
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

    func embed(_ text: String) async throws -> [Float] {
        try await ensureLoaded()
        guard let model, let tokenizer else {
            throw SearchEngineError.embeddingLoadFailed(NSError(
                domain: "CoreMLEmbeddingService", code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Model unavailable after load"]))
        }
        let fixedLen = 128
        let (ids, mask) = tokenizer.encode(text, maxLength: fixedLen)
        let inputIDs = try MLMultiArray(shape: [1, NSNumber(value: fixedLen)], dataType: .int32)
        let attentionMask = try MLMultiArray(shape: [1, NSNumber(value: fixedLen)], dataType: .int32)
        for i in 0..<fixedLen {
            inputIDs[i] = NSNumber(value: ids[i])
            attentionMask[i] = NSNumber(value: mask[i])
        }
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids":      MLFeatureValue(multiArray: inputIDs),
            "attention_mask": MLFeatureValue(multiArray: attentionMask),
        ])
        let result = try await model.prediction(from: features)
        guard let mla = result.featureValue(for: "embedding")?.multiArrayValue
                     ?? result.featureValue(for: "sentence_embedding")?.multiArrayValue
                     ?? result.featureValue(for: "embeddings")?.multiArrayValue else {
            throw SearchEngineError.invalidEmbedding
        }
        let dims = ModelConfig.embeddingDimensions
        let embedding = (0..<dims).map { mla[$0].floatValue }
        guard !embedding.contains(where: { $0.isNaN }) else {
            throw SearchEngineError.invalidEmbedding
        }
        return embedding
    }
}

// MARK: - SearchEngine

actor SearchEngine {

    private let embedder = CoreMLEmbeddingService()
    private let vectorSearch: VectorSearchService

    init() throws {
        vectorSearch = try VectorSearchService()
    }

    /// Embeds `query` and returns the top-K most relevant knowledge chunks.
    func search(query: String) async throws -> [KnowledgeChunk] {
        let embedding = try await embedder.embed(query)
        return try await vectorSearch.search(
            queryText: query,
            queryEmbedding: embedding,
            topK: ModelConfig.ragTopK
        )
    }
}
