// ModelConfig.swift
// THE single source of truth for the Llama model variant used in HyruleSage.
// To switch from 1B to 3B: change activeVariant to .llamaLarge, re-run the
// convert-model-llama CI job with MODEL_VARIANT=3B, and replace LlamaModel.mlpackage
// in Resources/. No other files need to change.

import Foundation

enum ModelVariant: String {
    case llamaSmall = "1B"   // Llama 3.2 1B Instruct  (~600 MB at 4-bit)
    case llamaLarge = "3B"   // Llama 3.2 3B Instruct  (~1.5 GB at 4-bit)
}

struct ModelConfig {

    // MARK: - Change this to switch model variants
    static let activeVariant: ModelVariant = .llamaSmall

    // MARK: - Derived config (do not edit below this line)

    static var modelFilename: String {
        switch activeVariant {
        case .llamaSmall: return "LlamaModel-1B.mlmodelc"
        case .llamaLarge: return "LlamaModel-3B.mlmodelc"
        }
    }

    /// Maximum number of past tokens stored in the fixed-size KV cache.
    /// Must match MAX_KV_LEN in model/convert_llm_llama.py — change both together via CI.
    static var llmMaxKVLen: Int { 512 }

    /// Maximum tokens for the initial prompt (system + RAG context + user question).
    /// Must stay well below llmMaxKVLen (512) once overhead tokens are added.
    /// 380 context + ~50 format/system overhead = ~430 tokens, fits in the 512-slot KV cache.
    static var maxContextTokens: Int { 380 }

    /// Generation-loop iteration cap.
    static var maxOutputTokens: Int { 512 }

    /// KV-cache dimensions — must match the converted model (set by convert_llm_llama.py).
    /// Llama 3.2 1B: 16 layers, 8 GQA KV heads, head_dim = 2048 / 32 = 64.
    /// Llama 3.2 3B: 28 layers, 8 GQA KV heads, head_dim = 3072 / 24 = 128.
    static var llmNumLayers: Int {
        switch activeVariant {
        case .llamaSmall: return 16
        case .llamaLarge: return 28
        }
    }
    static var llmNumKVHeads: Int { 8 }
    static var llmHeadDim: Int {
        switch activeVariant {
        case .llamaSmall: return 64
        case .llamaLarge: return 128
        }
    }

    /// Number of RAG chunks to retrieve.
    static var ragTopK: Int { 5 }

    /// Minimum cosine similarity for vector search results to be used.
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

    /// Wraps a user query in the Llama 3.2 Instruct chat format.
    /// <|eot_id|> (token 128009) signals end-of-turn; the model stops generating at this token.
    /// <|begin_of_text|> is the BOS token prepended automatically by the tokenizer.
    static func chatPrompt(userQuery: String) -> String {
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" +
        "\(systemPrompt)<|eot_id|>" +
        "<|start_header_id|>user<|end_header_id|>\n\n" +
        "\(userQuery)<|eot_id|>" +
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    }
}
