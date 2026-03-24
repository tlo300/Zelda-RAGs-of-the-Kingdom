# model/

Scripts for converting ML models to Core ML format for on-device iOS inference.

## Scripts

| Script | Output | When to run |
|--------|--------|-------------|
| `convert_embeddings.py` | `MiniLMEmbedder.mlpackage` | Once, or when embedding model changes |
| `convert_llm.py` | `LlamaModel-1B.mlpackage` or `LlamaModel-3B.mlpackage` | Once, or when switching variants |

Both scripts run via GitHub Actions on a `macos-14` runner — no local Mac required.

---

## Getting the LLM artifact (LlamaModel.mlpackage)

### Prerequisites

1. You need a HuggingFace account that has **accepted Meta's Llama 3.2 licence**:
   - Go to `meta-llama/Llama-3.2-1B-Instruct` on HuggingFace and click "Access repository"
   - Generate a HuggingFace token with `read` scope
2. Add the token as a GitHub Actions secret named **`HF_TOKEN`** in the repo settings
   (Settings → Secrets and variables → Actions → New repository secret)

### Trigger the conversion

1. Go to the repo on GitHub → **Actions** → **Convert Core ML model**
2. Click **Run workflow**
3. Select the model variant: `1B` (default, ~600 MB) or `3B` (~1.8 GB)
4. Click **Run workflow** to start the job

The job takes approximately 30–60 minutes (model download + conversion + smoke test).

### Download the artifact

1. After the job completes, open the workflow run
2. Under **Artifacts**, download **`LlamaModel-1B.mlpackage`** (or 3B)
3. Unzip the downloaded archive — it contains the `.mlpackage` bundle

### Place the artifact in the app

Copy the `.mlpackage` into the iOS Resources directory:

```
ios/ZeldaGuide/Resources/LlamaModel-1B.mlpackage
```

The filename must exactly match the value of `ModelConfig.modelFilename` in
`ios/ZeldaGuide/Services/ModelConfig.swift`. The conversion script enforces this
at CI time — if the names diverge, the job fails before uploading the artifact.

> **Note:** The `.mlpackage` is not committed to git (too large). You need to
> download it from the Actions artifact and place it locally before building the `.ipa`.

---

## Switching to 3B

See [docs/upgrade-to-3b.md](../docs/upgrade-to-3b.md) for the full upgrade path.
The short version: change `activeVariant` in `ModelConfig.swift` to `.llama3B`,
re-run the conversion workflow with `MODEL_VARIANT=3B`, and replace the artifact.

---

## iOS deployment target note

The LLM `.mlpackage` uses Core ML stateful conversion (KV-cache via `StateType`),
which requires **iOS 18** or later. iPhone 12 and later all support iOS 18, so this
is not a device restriction — only an OS version requirement.

The embedding model (`MiniLMEmbedder.mlpackage`) targets iOS 17 and is unaffected.
