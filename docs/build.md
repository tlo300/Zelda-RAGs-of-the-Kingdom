# Building and installing with SideStore

## Prerequisites
- SideStore installed on your iPhone (https://sidestore.io)
- A free Apple ID
- GitHub account with this repo

SideStore re-signs apps over the internet via a built-in WireGuard VPN — no PC or home Wi-Fi required.
This means the app stays valid while travelling.

## Building the .ipa
1. Go to the GitHub Actions tab in this repo
2. Select the 'Build iOS .ipa' workflow
3. Click 'Run workflow'
4. Wait ~15 minutes for the build to complete
5. Download the .ipa artifact from the completed run

## Installing with SideStore
1. Open SideStore on your iPhone
2. Tap the + button and select the downloaded .ipa file
3. The app will install and sign with your free Apple ID

## Auto-renewal
SideStore automatically re-signs the app every 7 days using its built-in WireGuard VPN.
No PC or home Wi-Fi needed — renewal works anywhere you have an internet connection.

## Switching to the 3B model
See docs/upgrade-to-3b.md for steps to switch to the 3B model.
