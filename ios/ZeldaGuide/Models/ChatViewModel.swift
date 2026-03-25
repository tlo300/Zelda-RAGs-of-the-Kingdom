// ChatViewModel.swift
// MainActor ObservableObject that drives ChatView.
// Owns the RAGEngine, manages the message list, and handles streaming cancellation.

import Foundation

@MainActor
final class ChatViewModel: ObservableObject {

    @Published var messages: [ChatMessage] = []
    @Published var inputText: String = ""
    @Published private(set) var isGenerating: Bool = false
    @Published private(set) var isLoading: Bool = false

    private var ragEngine: RAGEngine?
    private var streamTask: Task<Void, Never>?

    /// Production init — creates a real RAGEngine backed by the bundled knowledge base.
    /// If the database is missing, a permanent error message is shown in the chat.
    init() {
        do {
            ragEngine = try RAGEngine()
        } catch {
            messages.append(ChatMessage(
                role: .assistant,
                text: "[Guide unavailable: \(error.localizedDescription)]"
            ))
        }
    }

    /// Test / preview init — injects a pre-built RAGEngine.
    init(ragEngine: RAGEngine) {
        self.ragEngine = ragEngine
    }

    // MARK: - Actions

    func send() {
        let question = inputText.trimmingCharacters(in: .whitespaces)
        guard !question.isEmpty, !isGenerating, let engine = ragEngine else { return }
        inputText = ""

        messages.append(ChatMessage(role: .user, text: question))
        let assistant = ChatMessage(role: .assistant, isStreaming: true)
        messages.append(assistant)
        let assistantID = assistant.id

        isGenerating = true
        isLoading = true

        streamTask = Task {
            let stream = await engine.answer(question: question)
            var firstToken = true
            for await token in stream {
                guard !Task.isCancelled else { break }
                if firstToken {
                    isLoading = false
                    firstToken = false
                    let chunks = await engine.sourceChunks
                    if let idx = messages.firstIndex(where: { $0.id == assistantID }) {
                        messages[idx].sources = chunks
                    }
                }
                if let idx = messages.firstIndex(where: { $0.id == assistantID }) {
                    messages[idx].text += token
                }
            }
            finalise(assistantID: assistantID)
        }
    }

    func cancelGeneration() {
        streamTask?.cancel()
        streamTask = nil
        if let idx = messages.indices.last, messages[idx].isStreaming {
            messages[idx].isStreaming = false
        }
        isGenerating = false
        isLoading = false
    }

    // MARK: - Private

    private func finalise(assistantID: UUID) {
        if let idx = messages.firstIndex(where: { $0.id == assistantID }) {
            messages[idx].isStreaming = false
            if messages[idx].text.isEmpty {
                messages[idx].text = "(No answer generated)"
            }
        }
        isGenerating = false
        isLoading = false
    }
}
