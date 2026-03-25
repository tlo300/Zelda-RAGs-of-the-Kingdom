# Upgrading from Qwen2.5 1B to 3B model

Switch to the 3B model when you want better answer quality for complex questions.
The upgrade requires no code changes beyond a single constant in ModelConfig.swift.

---

## Steps

### 1. Update ModelConfig.swift

In [ios/ZeldaGuide/Services/ModelConfig.swift](../ios/ZeldaGuide/Services/ModelConfig.swift):

```swift
// Change:
static let activeVariant: ModelVariant = .qwen1B
// To:
static let activeVariant: ModelVariant = .qwen3B
```

### 2. Convert the 3B model

1. Go to **Actions → Convert Core ML model → Run workflow**.
2. Set `model_variant` to `3B`.
3. Click **Run workflow** and wait ~45 minutes.
4. Download the `QwenModel-3B.mlpackage` artifact from the completed run.

### 3. Replace the model bundle

Replace the existing model in the app resources:

```
ios/ZeldaGuide/Resources/QwenModel-1B.mlpackage  →  QwenModel-3B.mlpackage
```

Rename the downloaded artifact to `QwenModel-3B.mlpackage` and copy it into that folder.
Delete (or move aside) `QwenModel-1B.mlpackage`.

### 4. Build and sideload

1. Go to **Actions → Build iOS .ipa → Run workflow**.
2. Select `model_variant: 3B` so the run summary records it.
3. Download `ZeldaGuide.ipa` and sideload via SideStore as usual (see [build.md](build.md)).

---

## Notes

| | 1B | 3B |
|-|----|----|
| Model | Qwen2.5 1B Instruct | Qwen2.5 3B Instruct |
| App size | ~750 MB | ~1.8 GB |
| Speed (A15) | ~15 tok/s | ~8 tok/s |
| Speed (A17 Pro) | ~25 tok/s | ~20 tok/s |
| Distribution | SideStore (free Apple ID) | SideStore (free Apple ID) |

No other code changes are needed. ModelConfig.swift handles all variant-specific differences:
model filename, context length, and generation parameters.
