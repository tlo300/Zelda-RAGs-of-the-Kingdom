// SearchViewModel.swift
// Drives SearchView — owns the SearchEngine, manages query state, and
// exposes results or errors to the UI.

import Foundation

@MainActor
final class SearchViewModel: ObservableObject {

    @Published var query = ""
    @Published var results: [KnowledgeChunk] = []
    @Published var isSearching = false
    @Published var errorMessage: String? = nil
    @Published var hasSearched = false

    private var engine: SearchEngine?
    private let initError: String?

    init() {
        do {
            engine = try SearchEngine()
            initError = nil
        } catch {
            engine = nil
            initError = error.localizedDescription
        }
    }

    func search() async {
        let trimmed = query.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }

        if let initError {
            errorMessage = initError
            hasSearched = true
            return
        }

        isSearching = true
        errorMessage = nil
        hasSearched = true

        do {
            results = try await engine!.search(query: trimmed)
        } catch {
            errorMessage = error.localizedDescription
            results = []
        }

        isSearching = false
    }
}
