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

    /// Total token limit the Core ML model was compiled for.
    /// Must match max_context in model/convert_llm.py — change both together via convert-model CI.
    /// Set to 2048 to match convert-model run 23591992777.
    static var modelMaxSequenceLength: Int { 2048 }

    /// Maximum tokens for the initial prompt (system + RAG context + user question).
    /// Sized to leave headroom for the generated answer within modelMaxSequenceLength.
    static var maxContextTokens: Int { 1500 }

    /// Generation-loop iteration cap. The context guard may stop generation earlier
    /// if inputTokens.count reaches modelMaxSequenceLength first.
    static var maxOutputTokens: Int { 512 }

    /// Number of RAG chunks to retrieve.
    static var ragTopK: Int { 5 }
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
