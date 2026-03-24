# Global preferences addendum - iOS and Swift projects

## Additional environment
- No Mac available locally - use GitHub Actions macOS runners for Core ML conversion and Xcode builds
- iOS target: iPhone 12 or later minimum, any iPhone for 1B model
- Distribution via AltStore sideloading with free Apple ID - no paid developer account
- All model conversion and Xcode build steps must be scripted for CI - never require manual Xcode GUI steps
- MODEL_VARIANT env var controls which model is used: "1B" (default, free) or "3B" (requires paid account)

## Swift and iOS code style
- Swift 5.9+, iOS 17+ minimum deployment target
- Use Swift concurrency (async/await, actors) throughout - no completion handlers
- SwiftUI only - no UIKit unless absolutely forced by a framework dependency
- All Swift files include a module-level comment describing purpose
- Use Swift Package Manager for all dependencies - no CocoaPods or Carthage
- Model name, context length, and generation parameters all live in ModelConfig.swift
  Never hardcode model filenames or parameters anywhere else in the codebase

## Python data pipeline style (Windows)
- All Python scripts must run on Windows with PowerShell
- Use virtual environments: python -m venv .venv then .venv\Scripts\activate
- Use pathlib.Path throughout - never os.path string concatenation
- Rate limit all HTTP requests: minimum 1.5 seconds between requests, respect crawl-delay if set
- Log progress to console for long-running scrape/embed jobs so it is clear what is happening
- Tests always use mock HTTP (responses library) - never hit live wikis in tests

## GitHub Actions CI rules
- macos-14 runner for Core ML conversion and Xcode build jobs
- ubuntu-latest for Python data pipeline jobs
- MODEL_VARIANT secret/variable controls 1B vs 3B in convert-model workflow
- Never hardcode secrets - use GitHub Actions secrets
- Cache pip and derived data between runs
- build-ipa.yml produces an unsigned .ipa artifact - AltStore handles signing
- All workflow files live in .github/workflows/
