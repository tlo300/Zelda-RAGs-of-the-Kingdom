// ModelConfig.swift
// THE single source of truth for the LLM variant.
// To switch from 1B to 3B: change modelVariant to "3B", re-run the convert-model CI job,
// and replace QwenModel.mlpackage in Resources/. No other files need to change.

import Foundation

enum ModelVariant: String {
    case qwen1B = "1B"
    case qwen3B = "3B"
}

struct ModelConfig {

    // MARK: - Change this to switch model variants
    static let activeVariant: ModelVariant = .qwen1B

    // MARK: - Derived config (do not edit below this line)

    static var modelFilename: String {
        switch activeVariant {
        case .qwen1B: return "QwenModel-1B.mlmodelc"
        case .qwen3B: return "QwenModel-3B.mlmodelc"
        }
    }

    /// Maximum number of past tokens stored in the fixed-size KV cache.
    /// Must match MAX_KV_LEN in model/convert_llm.py — change both together via convert-model CI.
    static var llmMaxKVLen: Int { 512 }

    /// Maximum tokens for the RAG context block (chunks only, excluding system prompt and question).
    /// Must fit within llmMaxKVLen minus overhead (~80 tokens for system prompt + ChatML).
    /// 380 tokens × 4 chars ÷ 5 chunks ≈ 304 chars per chunk.
    static var maxContextTokens: Int { 380 }

    /// Generation-loop iteration cap. The context guard may stop generation earlier
    /// if inputTokens.count reaches modelMaxSequenceLength first.
    static var maxOutputTokens: Int { 512 }

    /// KV-cache dimensions — must match the converted model (set by convert_llm.py).
    /// Qwen2.5-1.5B: 28 layers, 2 GQA KV heads, head_dim = hidden_size / num_attention_heads = 128.
    /// These are used by CoreMLPredictor to allocate the initial empty KV-cache array.
    static var llmNumLayers: Int { 28 }
    static var llmNumKVHeads: Int { 2 }
    static var llmHeadDim: Int { 128 }

    /// Number of RAG chunks to retrieve.
    static var ragTopK: Int { 5 }

    /// Minimum cosine similarity for vector search results to be used.
    /// If the best result is below this threshold the pipeline falls back to FTS5 keyword search,
    /// which handles cases where the on-device embedder produces mismatched vectors.
    /// Similarity is derived from L2 distance d via: s = 1 − d²/2 (unit-vector identity).
    static var minVectorSimilarity: Float { 0.3 }

    static var embeddingDimensions: Int { 384 }
    static var embeddingModelFilename: String { "MiniLMEmbedder.mlmodelc" }
    static var knowledgeBaseFilename: String { "knowledge_base.db" }

    static var systemPrompt: String {
        return """
        You are a helpful guide for The Legend of Zelda: Tears of the Kingdom. \
        Answer questions using only the context provided below. \
        If the answer is not in the context, say you do not have that information. \
        Be concise and specific.
        """
    }

    /// Wraps a user query in the Qwen2.5 ChatML format required by the Instruct model.
    /// Without this wrapper the model degenerates into repetition loops.
    /// <|im_start|> = token 151644, <|im_end|> = token 151645 in the Qwen2.5 vocabulary.
    static func chatPrompt(userQuery: String) -> String {
        "<|im_start|>system\n\(systemPrompt)<|im_end|>\n" +
        "<|im_start|>user\n\(userQuery)<|im_end|>\n" +
        "<|im_start|>assistant\n"
    }
}
