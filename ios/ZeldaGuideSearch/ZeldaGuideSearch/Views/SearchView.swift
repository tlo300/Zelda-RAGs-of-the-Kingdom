// SearchView.swift
// Main UI for ZeldaGuideSearch. Provides a search bar and displays the top-K
// retrieved knowledge chunks as expandable cards — no LLM, instant results.

import SwiftUI

struct SearchView: View {
    @StateObject private var viewModel = SearchViewModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                searchBar
                Divider()
                resultsArea
            }
            .navigationTitle("TotK Guide")
            .navigationBarTitleDisplayMode(.inline)
        }
        .tint(.green)
    }

    // MARK: - Search bar

    private var searchBar: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)

            TextField("Dungeons, shrines, items, bosses…", text: $viewModel.query)
                .textFieldStyle(.plain)
                .submitLabel(.search)
                .onSubmit { Task { await viewModel.search() } }
                .autocorrectionDisabled()

            if viewModel.isSearching {
                ProgressView()
                    .tint(.green)
            } else if !viewModel.query.isEmpty {
                Button {
                    viewModel.query = ""
                    viewModel.results = []
                    viewModel.hasSearched = false
                    viewModel.errorMessage = nil
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }

                Button {
                    Task { await viewModel.search() }
                } label: {
                    Text("Search")
                        .fontWeight(.semibold)
                        .foregroundStyle(.green)
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    // MARK: - Results area

    @ViewBuilder
    private var resultsArea: some View {
        if let error = viewModel.errorMessage {
            ContentUnavailableView(
                "Search Error",
                systemImage: "exclamationmark.triangle",
                description: Text(error)
            )
        } else if viewModel.hasSearched && !viewModel.isSearching && viewModel.results.isEmpty {
            ContentUnavailableView(
                "No Results",
                systemImage: "magnifyingglass",
                description: Text("Try different words or a more specific question.")
            )
        } else if !viewModel.hasSearched {
            emptyState
        } else {
            ScrollView {
                LazyVStack(spacing: 12) {
                    ForEach(viewModel.results, id: \.id) { chunk in
                        ResultCardView(chunk: chunk)
                    }
                }
                .padding()
            }
        }
    }

    // MARK: - Empty state

    private var emptyState: some View {
        VStack(spacing: 20) {
            Image(systemName: "leaf.fill")
                .font(.system(size: 56))
                .foregroundStyle(.green)
            VStack(spacing: 8) {
                Text("Ask anything about TotK")
                    .font(.title3)
                    .fontWeight(.semibold)
                Text("Dungeons · Shrines · Bosses\nItems · Quests · Abilities")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
