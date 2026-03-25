# Upgrading from 1B to 3B model

When you want better answer quality for complex questions:

## Steps

1. Edit ios/ZeldaGuide/Services/ModelConfig.swift
   Change: static let activeVariant: ModelVariant = .llama1B
   To:     static let activeVariant: ModelVariant = .llama3B

2. Go to GitHub Actions > Convert Core ML model > Run workflow
   Set MODEL_VARIANT input to: 3B

3. Wait for the conversion job to complete (~45 minutes)

4. Download the LlamaModel-3B.mlpackage artifact

5. Replace ios/ZeldaGuide/Resources/LlamaModel-1B.mlpackage with LlamaModel-3B.mlpackage

6. Rebuild the .ipa via GitHub Actions > Build iOS .ipa

7. Sideload the new .ipa (or use Xcode direct install with your Developer account)

## Notes

- The 3B model requires ~1.8GB of storage vs ~600MB for 1B
- All other code stays the same - ModelConfig.swift handles all differences
- Sideload with SideStore the same way as the 1B build
