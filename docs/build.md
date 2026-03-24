# Building and installing with AltStore

## Prerequisites
- AltStore installed on your iPhone and PC (https://altstore.io)
- AltStore Daemon running on your PC
- GitHub account with this repo

## Building the .ipa

1. Go to the GitHub Actions tab in this repo
2. Select the 'Build iOS .ipa' workflow
3. Click 'Run workflow'
4. Wait ~15 minutes for the build to complete
5. Download the .ipa artifact from the completed run

## Installing with AltStore

1. Connect your iPhone to your PC via USB (first time only)
2. Open AltStore on your iPhone
3. Tap the + button and select the downloaded .ipa file
4. The app will install and sign with your free Apple ID

## Auto-renewal

AltStore Daemon on your PC will automatically re-sign the app every 7 days
when your iPhone is on the same Wi-Fi network as your PC.
You do not need to manually reinstall unless you delete the app.

## Switching to the 3B model

See docs/upgrade-to-3b.md when you have a paid Apple Developer account.
