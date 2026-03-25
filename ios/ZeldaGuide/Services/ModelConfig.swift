// ModelConfig.swift
// THE single source of truth for the LLM variant.
// To switch from 1B to 3B: change modelVariant to "3B", re-run the convert-model CI job,
// and replace LlamaModel.mlpackage in Resources/. No other files need to change.

import Foundation

enum ModelVariant: String {
    case llama1B = "1B"
    case llama3B = "3B"
}

struct ModelConfig {

    // MARK: - Change this to switch model variants
    static let activeVariant: ModelVariant = .llama1B

    // MARK: - Derived config (do not edit below this line)

    static var modelFilename: String {
        switch activeVariant {
        case .llama1B: return "LlamaModel-1B.mlpackage"
        case .llama3B: return "LlamaModel-3B.mlpackage"
        }
    }

    static var maxContextTokens: Int {
        switch activeVariant {
        case .llama1B: return 4096
        case .llama3B: return 8192
        }
    }

    static var maxOutputTokens: Int { 512 }
    static var ragTopK: Int { 5 }
    static var embeddingDimensions: Int { 384 }
    static var embeddingModelFilename: String { "MiniLMEmbedder.mlpackage" }
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
