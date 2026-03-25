// ChatView.swift
// Main chat screen. Displays the message history in a scrollable list, streams
// assistant answers token by token, and provides an input bar with Send / Stop controls.
// Dark mode is handled entirely by system colors and materials.

import SwiftUI

struct ChatView: View {
    @StateObject private var viewModel = ChatViewModel()
    @FocusState private var inputFocused: Bool

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                messageList
                Divider()
                inputBar
            }
            .navigationTitle("Zelda TotK Guide")
            .navigationBarTitleDisplayMode(.inline)
        }
        .tint(.green)
    }

    // MARK: - Message list

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 10) {
                    if viewModel.messages.isEmpty {
                        emptyState
                    } else {
                        ForEach(viewModel.messages) { message in
                            MessageBubbleView(message: message)
                                .padding(.horizontal, 12)
                                .id(message.id)
                        }
                    }
                }
                .padding(.vertical, 12)
            }
            .onChange(of: viewModel.messages.count) {
                scrollToBottom(proxy: proxy)
            }
            .onChange(of: viewModel.messages.last?.text) {
                scrollToBottom(proxy: proxy)
            }
        }
    }

    private func scrollToBottom(proxy: ScrollViewProxy) {
        guard let last = viewModel.messages.last else { return }
        withAnimation(.easeOut(duration: 0.2)) {
            proxy.scrollTo(last.id, anchor: .bottom)
        }
    }

    // MARK: - Empty state

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "leaf.fill")
                .font(.system(size: 48))
                .foregroundStyle(.green)
            Text("Ask about shrines, weapons,\nquests, or game mechanics.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 80)
    }

    // MARK: - Input bar

    private var inputBar: some View {
        HStack(alignment: .bottom, spacing: 10) {
            TextField("Ask a question…", text: $viewModel.inputText, axis: .vertical)
                .lineLimit(1...4)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(Color(.secondarySystemBackground))
                .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
                .focused($inputFocused)
                .disabled(viewModel.isGenerating)
                .onSubmit { viewModel.send() }

            if viewModel.isGenerating {
                stopButton
            } else {
                sendButton
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(.regularMaterial)
    }

    private var sendButton: some View {
        let empty = viewModel.inputText.trimmingCharacters(in: .whitespaces).isEmpty
        return Button(action: viewModel.send) {
            Image(systemName: "arrow.up.circle.fill")
                .font(.system(size: 34))
                .foregroundStyle(empty ? Color.secondary : Color.accentColor)
        }
        .disabled(empty)
    }

    private var stopButton: some View {
        Button(action: viewModel.cancelGeneration) {
            Image(systemName: "stop.circle.fill")
                .font(.system(size: 34))
                .foregroundStyle(.red)
        }
    }
}

#Preview {
    ChatView()
}
