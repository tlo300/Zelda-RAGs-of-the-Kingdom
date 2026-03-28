// ZeldaWalkthroughApp.swift
// Entry point for the ZeldaWalkthrough app — a structured chapter-based guide for TotK.

import SwiftUI

@main
struct ZeldaWalkthroughApp: App {
    var body: some Scene {
        WindowGroup {
            TableOfContentsView()
        }
        .tint(.green)
    }
}
