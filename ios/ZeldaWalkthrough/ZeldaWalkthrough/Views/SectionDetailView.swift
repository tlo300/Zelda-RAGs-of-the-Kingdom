// SectionDetailView.swift
// Displays the content of a single walkthrough section. Content is blank pending population.

import SwiftUI

struct SectionDetailView: View {
    let chapter: WalkthroughChapter
    let section: WalkthroughSection

    private var sectionIndex: Int? {
        chapter.sections.firstIndex(where: { $0.id == section.id })
    }

    private var previousSection: WalkthroughSection? {
        guard let i = sectionIndex, i > 0 else { return nil }
        return chapter.sections[i - 1]
    }

    private var nextSection: WalkthroughSection? {
        guard let i = sectionIndex, i < chapter.sections.count - 1 else { return nil }
        return chapter.sections[i + 1]
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                if section.content.isEmpty {
                    ContentUnavailableView(
                        "Content Coming Soon",
                        systemImage: "clock",
                        description: Text("This section hasn't been written yet.")
                    )
                    .padding(.top, 40)
                } else {
                    Text(section.content)
                        .font(.body)
                        .textSelection(.enabled)
                        .padding(.horizontal)
                }

                navigationFooter
            }
            .padding(.bottom, 32)
        }
        .navigationTitle(section.title)
        .navigationBarTitleDisplayMode(.inline)
    }

    @ViewBuilder
    private var navigationFooter: some View {
        if previousSection != nil || nextSection != nil {
            Divider()
                .padding(.horizontal)

            HStack {
                if let prev = previousSection {
                    NavigationLink(destination: SectionDetailView(chapter: chapter, section: prev)) {
                        Label(prev.title, systemImage: "chevron.left")
                            .font(.caption)
                            .lineLimit(1)
                    }
                }
                Spacer()
                if let next = nextSection {
                    NavigationLink(destination: SectionDetailView(chapter: chapter, section: next)) {
                        Label(next.title, systemImage: "chevron.right")
                            .labelStyle(.titleAndIcon)
                            .font(.caption)
                            .lineLimit(1)
                    }
                }
            }
            .padding(.horizontal)
        }
    }
}

#Preview {
    let chapter = WalkthroughData.chapters[1]
    NavigationStack {
        SectionDetailView(chapter: chapter, section: chapter.sections[0])
    }
}
