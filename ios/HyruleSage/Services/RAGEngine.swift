// RAGEngine.swift
// Wires VectorSearchService and LLMService into a complete RAG pipeline.
// answer(question:) is the single public entry point: it embeds the query on-device,
// retrieves the top-K context chunks, assembles a grounded prompt, and streams
// the LLM response token by token. sourceChunks exposes the retrieved chunks for
// the attribution UI while the stream is in progress.

import CoreML
import Foundation

// MARK: - Embedding errors

enum EmbeddingError: Error, LocalizedError {
    case modelNotFound(String)
    case loadFailed(Error)
    case invalidOutput(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "Embedding model '\(name)' not found in the app bundle."
        case .loadFailed(let error):
            return "Failed to load embedding model: \(error.localizedDescription)"
        case .invalidOutput(let reason):
            return "Embedding model produced invalid output: \(reason)"
        }
    }
}

// MARK: - EmbeddingService protocol (injectable for testing)

protocol EmbeddingService: Sendable {
    func embed(_ text: String) async throws -> [Float]
}

// MARK: - Core ML production embedder

actor CoreMLEmbeddingService: EmbeddingService {

    private var model: MLModel?
    private var tokenizer: WordPieceTokenizer?

    private func ensureLoaded() async throws {
        guard model == nil else { return }
        let filename = ModelConfig.embeddingModelFilename
        let url      = URL(fileURLWithPath: filename)
        let baseName = url.deletingPathExtension().lastPathComponent
        let ext      = url.pathExtension
        guard let modelURL = Bundle.main.url(forResource: baseName, withExtension: ext) else {
            throw EmbeddingError.modelNotFound(filename)
        }
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly
            let compiledURL = try await compileAndCache(modelURL)
            model = try await MLModel.load(contentsOf: compiledURL, configuration: config)
            tokenizer = try WordPieceTokenizer()
        } catch {
            throw EmbeddingError.loadFailed(error)
        }
    }

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

    static func buildInputArrays(for text: String, using tokenizer: WordPieceTokenizer) throws -> (inputIDs: MLMultiArray, attentionMask: MLMultiArray) {
        let fixedLen = 128
        let (ids, mask) = tokenizer.encode(text, maxLength: fixedLen)
        let inputIDs      = try MLMultiArray(shape: [1, NSNumber(value: fixedLen)], dataType: .int32)
        let attentionMask = try MLMultiArray(shape: [1, NSNumber(value: fixedLen)], dataType: .int32)
        for i in 0..<fixedLen {
            inputIDs[i]      = NSNumber(value: ids[i])
            attentionMask[i] = NSNumber(value: mask[i])
        }
        return (inputIDs, attentionMask)
    }

    func embed(_ text: String) async throws -> [Float] {
        try await ensureLoaded()
        guard let model else {
            throw EmbeddingError.loadFailed(NSError(
                domain: "CoreMLEmbeddingService", code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Model unavailable after load"]))
        }
        guard let tokenizer else {
            throw EmbeddingError.loadFailed(NSError(
                domain: "CoreMLEmbeddingService", code: -2,
                userInfo: [NSLocalizedDescriptionKey: "WordPiece tokenizer unavailable after load"]))
        }
        let (inputIDs, attentionMask) = try CoreMLEmbeddingService.buildInputArrays(for: text, using: tokenizer)

        let features = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids":      MLFeatureValue(multiArray: inputIDs),
            "attention_mask": MLFeatureValue(multiArray: attentionMask),
        ])
        let result = try await model.prediction(from: features)

        guard let mla = result.featureValue(for: "embedding")?.multiArrayValue
                     ?? result.featureValue(for: "sentence_embedding")?.multiArrayValue
                     ?? result.featureValue(for: "embeddings")?.multiArrayValue else {
            throw EmbeddingError.invalidOutput(
                "Model output missing expected key ('embedding', 'sentence_embedding', or 'embeddings')")
        }

        let dims      = ModelConfig.embeddingDimensions
        let embedding = (0..<dims).map { mla[$0].floatValue }

        guard !embedding.contains(where: { $0.isNaN }) else {
            throw EmbeddingError.invalidOutput(
                "Embedding vector contains NaN — possible model or tokenisation error")
        }

        return embedding
    }
}

// MARK: - RAGEngine

actor RAGEngine {

    private let embedder: any EmbeddingService
    private let vectorSearch: VectorSearchService
    private let llm: LLMService

    private(set) var sourceChunks: [KnowledgeChunk] = []

    init() throws {
        embedder     = CoreMLEmbeddingService()
        vectorSearch = try VectorSearchService()
        llm          = LLMService()
    }

    init(embedder: any EmbeddingService,
         vectorSearch: VectorSearchService,
         llm: LLMService) {
        self.embedder     = embedder
        self.vectorSearch = vectorSearch
        self.llm          = llm
    }

    func prepare() async throws {
        try await llm.load()
    }

    func answer(question: String) -> AsyncStream<String> {
        let (stream, continuation) = AsyncStream<String>.makeStream()
        Task { [self] in
            await self.runRAG(question: question, continuation: continuation)
        }
        return stream
    }

    // MARK: - Private

    private func runRAG(
        question: String,
        continuation: AsyncStream<String>.Continuation
    ) async {
        defer { continuation.finish() }

        let queryEmbedding: [Float]
        do {
            queryEmbedding = try await embedder.embed(question)
        } catch {
            let desc = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            continuation.yield("[Error: \(desc)]")
            return
        }

        guard !queryEmbedding.contains(where: { $0.isNaN || $0.isInfinite }) else {
            continuation.yield("[Error: Query embedding contains NaN/Inf — cannot perform vector search]")
            return
        }

        let chunks: [KnowledgeChunk]
        do {
            chunks = try await vectorSearch.search(
                queryText: question,
                queryEmbedding: queryEmbedding,
                topK: ModelConfig.ragTopK
            )
        } catch {
            continuation.yield("[Error: Vector search failed — \(error.localizedDescription)]")
            return
        }

        sourceChunks = chunks

        let userMessage = assembleUserMessage(question: question, chunks: chunks)
        for await token in await llm.generate(prompt: userMessage) {
            continuation.yield(token)
        }
    }

    private func assembleUserMessage(question: String, chunks: [KnowledgeChunk]) -> String {
        let contextBlock: String
        if chunks.isEmpty {
            contextBlock = "(No relevant information found in the knowledge base.)"
        } else {
            let charsPerChunk = (ModelConfig.maxContextTokens * 4) / max(chunks.count, 1)
            contextBlock = chunks.enumerated().map { _, chunk in
                let text = String(chunk.chunkText.prefix(charsPerChunk))
                return "<<\(chunk.source) | \(chunk.pageTitle)>>\n\(text)"
            }.joined(separator: "\n\n")
        }
        return "Context:\n\(contextBlock)\n\nQuestion: \(question)"
    }
}
