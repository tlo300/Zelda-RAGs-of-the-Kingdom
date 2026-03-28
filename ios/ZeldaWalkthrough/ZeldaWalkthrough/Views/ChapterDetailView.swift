// ChapterDetailView.swift
// Lists all sections within a walkthrough chapter.

import SwiftUI

struct ChapterDetailView: View {
    let chapter: WalkthroughChapter

    var body: some View {
        List {
            ForEach(Array(chapter.sections.enumerated()), id: \.element.id) { index, section in
                NavigationLink(destination: SectionDetailView(chapter: chapter, section: section)) {
                    SectionRow(index: index + 1, section: section)
                }
            }
        }
        .navigationTitle(chapter.title)
        .navigationBarTitleDisplayMode(.inline)
    }
}

private struct SectionRow: View {
    let index: Int
    let section: WalkthroughSection

    var body: some View {
        HStack(spacing: 12) {
            Text("\(index)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 24, alignment: .trailing)
            Text(section.title)
                .font(.body)
            Spacer()
            if section.content.isEmpty {
                Image(systemName: "clock")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }
}

#Preview {
    NavigationStack {
        ChapterDetailView(chapter: WalkthroughData.chapters[1])
    }
}
