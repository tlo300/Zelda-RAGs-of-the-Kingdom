// MessageBubbleView.swift
// Single chat message bubble. User messages appear on the right in Hylian green;
// assistant messages appear on the left with the secondary system background.
// Shows an animated typing indicator while the assistant response is loading.

import SwiftUI

struct MessageBubbleView: View {
    let message: ChatMessage

    private var isUser: Bool { message.role == .user }

    var body: some View {
        HStack(alignment: .bottom, spacing: 0) {
            if isUser { Spacer(minLength: 48) }
            VStack(alignment: isUser ? .trailing : .leading, spacing: 6) {
                if !message.text.isEmpty {
                    Text(message.text)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(isUser ? Color.accentColor : Color(.secondarySystemBackground))
                        .foregroundStyle(isUser ? Color.white : Color.primary)
                        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                        .textSelection(.enabled)
                }
                if message.isStreaming && message.text.isEmpty {
                    TypingIndicatorView()
                }
                if !isUser && !message.sources.isEmpty && !message.isStreaming {
                    SourceAttributionView(sources: message.sources)
                }
            }
            if !isUser { Spacer(minLength: 48) }
        }
    }
}

// MARK: - Typing indicator

/// Three-dot animated indicator shown while the assistant is retrieving context.
struct TypingIndicatorView: View {
    @State private var dotOpacities: [Double] = [1.0, 0.3, 0.3]

    var body: some View {
        HStack(spacing: 5) {
            ForEach(0..<3, id: \.self) { i in
                Circle()
                    .fill(Color.secondary)
                    .frame(width: 8, height: 8)
                    .opacity(dotOpacities[i])
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .task { await animateDots() }
    }

    private func animateDots() async {
        var step = 0
        while !Task.isCancelled {
            withAnimation(.easeInOut(duration: 0.3)) {
                dotOpacities = (0..<3).map { $0 == step % 3 ? 1.0 : 0.3 }
            }
            step += 1
            try? await Task.sleep(for: .milliseconds(400))
        }
    }
}
