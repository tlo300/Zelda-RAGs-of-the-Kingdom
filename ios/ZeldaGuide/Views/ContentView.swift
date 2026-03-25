// ContentView.swift
// Placeholder home screen shown while RAG services are being implemented.
// Replace with ChatView once LLMService and RAGEngine are complete (issues #13, #14).

import SwiftUI

struct ContentView: View {
    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "books.vertical.fill")
                .font(.system(size: 72))
                .foregroundStyle(.green)

            Text("Zelda TotK Guide")
                .font(.largeTitle)
                .fontWeight(.bold)

            Text("AI-powered offline guide")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            Spacer()
                .frame(height: 8)

            Text("RAG services coming soon")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
