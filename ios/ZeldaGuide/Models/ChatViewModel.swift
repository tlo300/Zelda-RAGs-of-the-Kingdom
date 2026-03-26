// ChatViewModel.swift
// MainActor ObservableObject that drives ChatView.
// Owns the RAGEngine, manages the message list, and handles streaming cancellation.

import Foundation
import os

private let ciLog = Logger(subsystem: "com.tlo300.ZeldaGuide", category: "CI")

@MainActor
final class ChatViewModel: ObservableObject {

    @Published var messages: [ChatMessage] = []
    @Published var inputText: String = ""
    @Published private(set) var isGenerating: Bool = false
    @Published private(set) var isLoading: Bool = false
    @Published private(set) var isReady: Bool = false

    private var ragEngine: RAGEngine?
    private var streamTask: Task<Void, Never>?

    /// Production init — creates a real RAGEngine backed by the bundled knowledge base.
    /// If the database is missing, a permanent error message is shown in the chat.
    init() {
        do {
            let engine = try RAGEngine()
            ragEngine = engine
            Task {
                do {
                    try await engine.prepare()
                    isReady = true
                    if ProcessInfo.processInfo.arguments.contains("--autoquery") {
                        await runCIAutoQuery(engine: engine)
                    }
                } catch {
                    if ProcessInfo.processInfo.arguments.contains("--autoquery") {
                        ciLog.notice("[CI-AUTOQUERY] FAIL: engine.prepare() threw — \(error.localizedDescription, privacy: .public)")
                    }
                    messages.append(ChatMessage(
                        role: .assistant,
                        text: "[LLM unavailable: \(error.localizedDescription)]"
                    ))
                }
            }
        } catch {
            messages.append(ChatMessage(
                role: .assistant,
                text: "[Guide unavailable: \(error.localizedDescription)]"
            ))
        }
    }

    /// Test / preview init — injects a pre-built RAGEngine (already loaded).
    init(ragEngine: RAGEngine) {
        self.ragEngine = ragEngine
        self.isReady = true
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

    /// CI-only: submits a fixed test question and prints a pass/fail marker for debug-simulator.yml.
    /// Invoked automatically when the app is launched with `--autoquery`.
    private func runCIAutoQuery(engine: RAGEngine) async {
        let question = "What is a Korok Seed?"
        ciLog.notice("[CI-AUTOQUERY] Engine ready — submitting test query: \(question, privacy: .public)")
        let stream = await engine.answer(question: question)
        var answer = ""
        for await token in stream {
            answer += token
        }
        if answer.isEmpty {
            ciLog.notice("[CI-AUTOQUERY] FAIL: got empty answer")
        } else if answer.hasPrefix("[Error:") {
            ciLog.notice("[CI-AUTOQUERY] FAIL: \(answer, privacy: .public)")
        } else {
            ciLog.notice("[CI-AUTOQUERY] PASS: \(answer.prefix(200), privacy: .public)")
        }
    }
}
