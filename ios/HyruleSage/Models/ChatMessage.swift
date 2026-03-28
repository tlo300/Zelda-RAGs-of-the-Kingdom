// ChatMessage.swift
// Data model for a single message in the chat session.
// Sessions are in-memory only — not persisted between app launches.

import Foundation

struct ChatMessage: Identifiable {
    enum Role { case user, assistant }

    let id: UUID
    let role: Role
    var text: String
    var isStreaming: Bool
    var sources: [KnowledgeChunk]

    init(role: Role, text: String = "", isStreaming: Bool = false, sources: [KnowledgeChunk] = []) {
        self.id = UUID()
        self.role = role
        self.text = text
        self.isStreaming = isStreaming
        self.sources = sources
    }
}
