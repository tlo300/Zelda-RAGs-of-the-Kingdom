// ZeldaGuideSearchApp.swift
// Entry point for ZeldaGuideSearch — an offline semantic search companion for
// The Legend of Zelda: Tears of the Kingdom. No LLM; results are instant.

import SwiftUI

@main
struct ZeldaGuideSearchApp: App {
    var body: some Scene {
        WindowGroup {
            SearchView()
        }
    }
}
