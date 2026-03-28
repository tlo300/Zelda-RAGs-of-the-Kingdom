// TableOfContentsView.swift
// Root view — displays all walkthrough chapters grouped by category.

import SwiftUI

struct TableOfContentsView: View {
    var body: some View {
        NavigationStack {
            List {
                ForEach(ChapterCategory.allCases, id: \.self) { category in
                    let chapters = WalkthroughData.chapters(for: category)
                    if !chapters.isEmpty {
                        Section(category.rawValue) {
                            ForEach(chapters) { chapter in
                                NavigationLink(destination: ChapterDetailView(chapter: chapter)) {
                                    ChapterRow(chapter: chapter)
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("TotK Walkthrough")
            .navigationBarTitleDisplayMode(.large)
        }
    }
}

private struct ChapterRow: View {
    let chapter: WalkthroughChapter

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: chapter.icon)
                .font(.title3)
                .foregroundStyle(.green)
                .frame(width: 28)
            VStack(alignment: .leading, spacing: 2) {
                Text(chapter.title)
                    .font(.body)
                Text("\(chapter.sections.count) sections")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }
}

#Preview {
    TableOfContentsView()
}
