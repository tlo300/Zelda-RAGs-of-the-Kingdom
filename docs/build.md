# Building and installing with SideStore

## Overview

This app is distributed as an unsigned `.ipa` built by GitHub Actions.
SideStore signs it with your free Apple ID and auto-renews every 7 days over the internet —
no PC, no home Wi-Fi, no paid Apple Developer account required.

---

## Step 1 — Set up SideStore on your iPhone

1. On your iPhone, go to [sidestore.io](https://sidestore.io) and follow the installation guide.
   SideStore itself is installed via a web-based profile — you only need to do this once.
2. Open **Settings → General → VPN & Device Management** and trust the SideStore certificate.
3. Open SideStore and complete the pairing wizard.
   SideStore needs a WireGuard VPN profile to contact Apple's signing servers without a PC.
   Follow the in-app prompts to install it.
4. Log in with your free Apple ID when prompted.

> **3-app limit**: A free Apple ID can have at most 3 sideloaded apps active at a time.
> SideStore itself counts as one of those 3.

---

## Step 2 — Build the .ipa on GitHub Actions

1. Go to **Actions → Build iOS .ipa** in this repo.
2. Click **Run workflow**.
3. Select the model variant:
   - `1B` (default) — ~750 MB app, works on any iPhone with iOS 18+
   - `3B` — ~1.8 GB app, better answer quality (see [upgrade-to-3b.md](upgrade-to-3b.md))
4. Click **Run workflow** and wait ~15 minutes for the build to finish.
5. Open the completed run and download the `ZeldaGuide.ipa` artifact.
   The run summary shows the model variant and `.ipa` size.

---

## Step 3 — Transfer the .ipa to your iPhone

**Option A — Files app (simplest)**

1. Upload `ZeldaGuide.ipa` to iCloud Drive, Dropbox, or any cloud storage accessible from your iPhone.
2. On your iPhone, open the **Files** app and locate the `.ipa`.

**Option B — AirDrop**

1. On a Mac, AirDrop the `.ipa` directly to your iPhone.

---

## Step 4 — Sideload with SideStore

1. Open **SideStore** on your iPhone.
2. Tap the **+** button (bottom-right).
3. Navigate to the `.ipa` file using the Files picker and tap it.
4. SideStore signs and installs the app (~30 seconds).
5. Go to **Settings → General → VPN & Device Management** and trust the new certificate if prompted.

---

## Auto-renewal

SideStore automatically re-signs all sideloaded apps every 7 days using its built-in WireGuard VPN.
The phone just needs an internet connection on the renewal day — no PC or home Wi-Fi required.

If you miss a renewal and the app expires, open SideStore and tap **Refresh All**.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Maximum number of apps reached" | Free Apple ID limit is 3 apps (including SideStore). Remove one in SideStore before installing. |
| App crashes on launch | iOS 18+ is required. Check Settings → General → About for your iOS version. |
| SideStore can't connect | Open SideStore → Settings and confirm the WireGuard VPN is enabled. |
| Build fails in CI | Check the Actions log. The workflow uses macos-14; Xcode version mismatches are the most common cause. |

---

## Switching to the 3B model

See [upgrade-to-3b.md](upgrade-to-3b.md) for the exact steps.
