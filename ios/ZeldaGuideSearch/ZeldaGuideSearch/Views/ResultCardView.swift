// ResultCardView.swift
// Displays a single knowledge chunk as an expandable card.
// Shows page title, source badge, and chunk text (initially truncated to 4 lines).

import SwiftUI

struct ResultCardView: View {
    let chunk: KnowledgeChunk
    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top) {
                Text(chunk.pageTitle)
                    .font(.headline)
                    .lineLimit(2)
                Spacer()
                Text(sourceLabel)
                    .font(.caption2)
                    .fontWeight(.medium)
                    .padding(.horizontal, 7)
                    .padding(.vertical, 3)
                    .background(sourceColor.opacity(0.15))
                    .foregroundStyle(sourceColor)
                    .clipShape(Capsule())
            }

            ParagraphText(text: chunk.chunkText, isExpanded: isExpanded)

            Button {
                withAnimation(.easeInOut(duration: 0.2)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack(spacing: 4) {
                    Text(isExpanded ? "Show less" : "Show more")
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                }
                .font(.caption)
                .foregroundStyle(.green)
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var sourceLabel: String {
        let s = chunk.source.lowercased()
        if s.contains("compendium") || s.contains("hyrule") { return "Compendium" }
        if s.contains("zeldawiki") || s.contains("zelda.wiki") { return "Zelda Wiki" }
        return "Wiki"
    }

    private var sourceColor: Color {
        let s = chunk.source.lowercased()
        if s.contains("compendium") || s.contains("hyrule") { return .orange }
        return .blue
    }
}

// Renders chunk text respecting paragraph breaks (\n\n).
// Collapsed mode caps the first paragraph at 4 lines; expanded shows all paragraphs.
private struct ParagraphText: View {
    let text: String
    let isExpanded: Bool

    private var paragraphs: [String] {
        text.components(separatedBy: "\n\n").filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if isExpanded {
                ForEach(Array(paragraphs.enumerated()), id: \.offset) { _, para in
                    Text(para)
                        .font(.body)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
            } else {
                Text(paragraphs.first ?? text)
                    .font(.body)
                    .lineLimit(4)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            }
        }
    }
}
