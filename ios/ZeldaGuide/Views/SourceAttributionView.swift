// SourceAttributionView.swift
// Disclosure group shown below each assistant answer, listing up to three knowledge
// chunks used to generate the response. Wiki sources open in SFSafariViewController
// when tapped. Compendium sources display a plain label with no tap target.
// A GFDL attribution line is always visible below the disclosure group.

import SafariServices
import SwiftUI

struct SourceAttributionView: View {
    let sources: [KnowledgeChunk]

    @State private var safariItem: IdentifiableURL?

    private var displaySources: [KnowledgeChunk] { Array(sources.prefix(3)) }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            DisclosureGroup("Sources") {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(displaySources, id: \.id) { chunk in
                        sourceRow(chunk)
                    }
                }
                .padding(.top, 4)
            }
            .font(.caption)

            Text("Content from Zelda Dungeon Wiki and Zelda Wiki, licenced under GFDL")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .sheet(item: $safariItem) { item in
            SafariView(url: item.url)
        }
    }

    // MARK: - Private

    @ViewBuilder
    private func sourceRow(_ chunk: KnowledgeChunk) -> some View {
        if let url = wikiURL(for: chunk) {
            Button {
                safariItem = IdentifiableURL(url)
            } label: {
                sourceLabel(chunk)
                    .foregroundStyle(.tint)
            }
            .buttonStyle(.plain)
        } else {
            sourceLabel(chunk)
                .foregroundStyle(.secondary)
        }
    }

    private func sourceLabel(_ chunk: KnowledgeChunk) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(chunk.pageTitle)
                .font(.caption)
                .fontWeight(.medium)
            Text(displayName(for: chunk.source))
                .font(.caption2)
        }
    }

    private func wikiURL(for chunk: KnowledgeChunk) -> URL? {
        let base: String
        switch chunk.source {
        case "zelda_wiki":
            base = "https://zeldawiki.wiki/wiki/"
        case "zelda_dungeon":
            base = "https://www.zeldadungeon.net/wiki/"
        default:
            // compendium/* and any unknown source have no canonical URL
            return nil
        }
        let slug = chunk.pageTitle
            .addingPercentEncoding(withAllowedCharacters: .urlPathAllowed)?
            .replacingOccurrences(of: "%20", with: "_") ?? chunk.pageTitle
        return URL(string: base + slug)
    }

    private func displayName(for source: String) -> String {
        if source.hasPrefix("compendium") { return "Hyrule Compendium" }
        switch source {
        case "zelda_wiki":    return "Zelda Wiki"
        case "zelda_dungeon": return "Zelda Dungeon Wiki"
        default:              return source
        }
    }
}

// MARK: - SFSafariViewController wrapper

private struct SafariView: UIViewControllerRepresentable {
    let url: URL

    func makeUIViewController(context: Context) -> SFSafariViewController {
        SFSafariViewController(url: url)
    }

    func updateUIViewController(_ uiViewController: SFSafariViewController, context: Context) {}
}

// MARK: - URL sheet helper

private struct IdentifiableURL: Identifiable {
    let id: String
    let url: URL
    init(_ url: URL) { self.id = url.absoluteString; self.url = url }
}
