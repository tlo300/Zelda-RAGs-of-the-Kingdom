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

/// Converts a text string to a 384-dimension sentence embedding.
protocol EmbeddingService: Sendable {
    func embed(_ text: String) async throws -> [Float]
}

// MARK: - Core ML production embedder

/// Wraps the bundled MiniLMEmbedder.mlpackage and produces 384-dim sentence embeddings.
/// Loads the Core ML model lazily on the first embed() call.
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

    /// Returns a compiled .mlmodelc URL ready for MLModel.load.
    /// Pre-compiled .mlmodelc bundles (CI builds) are returned directly.
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

    /// Builds the `[1, 128]` MLMultiArray pair expected by the MiniLM Core ML model.
    /// The model was traced with max_length=128 — inputs must always have exactly that shape.
    /// Uses BERT WordPiece tokenization: [CLS] + subwords + [SEP] + zero-padding.
    /// Attention mask is 1 for real tokens and 0 for padding positions.
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

    /// Embeds `text` using the Core ML all-MiniLM-L6-v2 model.
    /// Throws `EmbeddingError.invalidOutput` if the vector contains NaN values.
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

        // coremltools names the output based on the ct.TensorType name passed at conversion time.
        // Our convert_embeddings.py uses "embedding"; fall back to legacy names for older builds.
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

/// The single entry point for the RAG pipeline.
///
/// answer(question:) orchestrates four steps:
///   1. Embed the question on-device with the MiniLM Core ML model.
///   2. Retrieve the top-K chunks from knowledge_base.db via VectorSearchService
///      (with automatic FTS5 fallback when vector search returns nothing).
///   3. Assemble a grounded user message that includes the retrieved context.
///   4. Stream LLM output via LLMService.
///
/// Error handling:
/// - Embedding failure or NaN vector → single "[Error: …]" token, stream ends immediately.
/// - Zero chunks retrieved → LLM is still called with an explicit no-context prompt so it
///   states it does not have the information rather than silently producing an empty stream.
actor RAGEngine {

    private let embedder: any EmbeddingService
    private let vectorSearch: VectorSearchService
    private let llm: LLMService

    /// The context chunks retrieved for the most recent answer() call.
    /// Populated before the first LLM token is yielded, so the attribution UI can read
    /// this property while the stream is still in progress.
    private(set) var sourceChunks: [KnowledgeChunk] = []

    // MARK: - Initialisation

    /// Production init — opens the bundled knowledge base and wires up all services.
    /// Throws `VectorSearchError.databaseNotFound` if knowledge_base.db is absent.
    init() throws {
        embedder     = CoreMLEmbeddingService()
        vectorSearch = try VectorSearchService()
        llm          = LLMService()
    }

    /// Test init — injects all three dependencies so tests run without real Core ML models.
    init(embedder: any EmbeddingService,
         vectorSearch: VectorSearchService,
         llm: LLMService) {
        self.embedder     = embedder
        self.vectorSearch = vectorSearch
        self.llm          = llm
    }

    // MARK: - Public API

    /// Loads the LLM into memory. Must be called once before the first answer() call.
    /// Safe to call multiple times — the underlying service no-ops if already loaded.
    func prepare() async throws {
        try await llm.load()
    }

    /// Answers a question via the full RAG pipeline, streaming LLM output token by token.
    /// Errors are surfaced as a single "[Error: …]" token rather than thrown exceptions,
    /// so the caller can always treat the return value as a uniform token stream.
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

        // Step 1: Generate query embedding.
        let queryEmbedding: [Float]
        do {
            queryEmbedding = try await embedder.embed(question)
        } catch {
            let desc = (error as? LocalizedError)?.errorDescription
                    ?? error.localizedDescription
            continuation.yield("[Error: \(desc)]")
            return
        }

        // Step 2: Validate — belt-and-suspenders guard in case the service returns NaN
        // without throwing (e.g. a test double or a future implementation change).
        guard !queryEmbedding.contains(where: { $0.isNaN }) else {
            continuation.yield(
                "[Error: Query embedding contains NaN — cannot perform vector search]")
            return
        }

        // Step 3: Retrieve top-K chunks (FTS5 fallback is built into VectorSearchService).
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

        // Step 4: Expose source chunks for the attribution UI before streaming begins.
        sourceChunks = chunks

        // Step 5: Assemble grounded prompt and stream LLM output.
        let userMessage = assembleUserMessage(question: question, chunks: chunks)
        for await token in await llm.generate(prompt: userMessage) {
            continuation.yield(token)
        }
    }

    /// Builds the user-turn message passed to `LLMService.generate(prompt:)`.
    /// When chunks is empty the message instructs the model that no context was found,
    /// so it will say it does not have the information rather than hallucinating.
    private func assembleUserMessage(question: String, chunks: [KnowledgeChunk]) -> String {
        let contextBlock: String
        if chunks.isEmpty {
            contextBlock = "(No relevant information found in the knowledge base.)"
        } else {
            // Limit each chunk to 500 characters (~130 BPE tokens) so the full assembled
            // prompt stays within the 512-token model limit. Remove this cap after
            // re-running convert-model with max_context = 2048 (issue #83).
            contextBlock = chunks.enumerated().map { index, chunk in
                let text = String(chunk.chunkText.prefix(500))
                return "[\(index + 1)] (source: \(chunk.source) | \(chunk.pageTitle))\n\(text)"
            }.joined(separator: "\n\n")
        }
        return "Context:\n\(contextBlock)\n\nQuestion: \(question)"
    }
}
