# Extended MedGemma Tokenizer

**Status:** ✅ Built and Ready to Use

**Created:** 2026-01-20

## Summary

Extended tokenizer for MedGemma with Korean medical vocabulary support.

- **Base Model:** google/medgemma-4b-it (also works with medgemma-27b)
- **Original Vocabulary:** 262,145 tokens
- **Extended Vocabulary:** 276,172 tokens
- **Tokens Added:** +14,027 (5.4% increase)

## Vocabulary Sources

| Source | Count | Description |
|--------|-------|-------------|
| Special Characters | 31 | Medical symbols (℃, ≥, μ, →, etc.) |
| Dictionary Terms | 4,670 | Korean medical terminology |
| Train02 Words | 10,000 | MCQ training vocabulary |
| **Total** | **14,701** | Before deduplication |
| Duplicates Removed | 376 | |
| Already in Vocab | 298 | |
| **Actually Added** | **14,027** | Final count |

## Performance

**Tokenization Efficiency:** 1.11x improvement

Example improvements:
- `심근경색의 주요 증상은 흉통입니다.` → 13 tokens → 10 tokens (1.30x)
- `혈액검사 결과 포도당 ≥126 mg/dL입니다.` → 17 tokens → 14 tokens (1.21x)

## Files

```
model/tokenizer/
├── extended_tokenizer/          ← Extended tokenizer (ready to use)
│   ├── tokenizer.json          (35 MB)
│   ├── tokenizer.model         (4.5 MB)
│   ├── added_tokens.json       (368 KB - 14,027 new tokens)
│   ├── tokenizer_config.json   (3.6 MB)
│   └── ...
├── token_mapping.json          ← Token ID mapping
├── tokenizer_stats.json        ← Build statistics
└── README.md                   ← This file
```

## Usage

### Load Extended Tokenizer

```python
from transformers import AutoTokenizer

# Load extended tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "model/tokenizer/extended_tokenizer"
)

# Verify
print(f"Vocab size: {len(tokenizer):,}")  # 276,172

# Test tokenization
text = "당뇨병 환자의 혈당 조절이 필요합니다."
tokens = tokenizer.tokenize(text)
print(f"Tokens ({len(tokens)}): {tokens}")
```

### Compare with Original

```python
from transformers import AutoTokenizer

# Original MedGemma tokenizer
original = AutoTokenizer.from_pretrained("google/medgemma-4b-it")

# Extended tokenizer
extended = AutoTokenizer.from_pretrained("model/tokenizer/extended_tokenizer")

text = "심근경색의 주요 증상은 흉통입니다."

orig_tokens = original.tokenize(text)
ext_tokens = extended.tokenize(text)

print(f"Original: {len(orig_tokens)} tokens")  # 13
print(f"Extended: {len(ext_tokens)} tokens")   # 10
print(f"Improvement: {len(orig_tokens)/len(ext_tokens):.2f}x")  # 1.30x
```

## Rebuild Tokenizer

If you need to rebuild with different settings:

```bash
cd data/tokenizer

# Basic build (medical tokens only)
python3 build_medgemma_tokenizer.py

# Include general Korean tokens (requires general_korean_tokens.txt)
python3 build_medgemma_tokenizer.py --use-general-korean

# Custom token limits
python3 build_medgemma_tokenizer.py --max-tokens 30000 --max-train02 15000

# Use different base model
python3 build_medgemma_tokenizer.py --base-model google/medgemma-27b-text-it
```

## Next Steps

### 1. Initialize LoRA with Extended Tokenizer

Create and run `init_lora_with_extended_tokenizer.py`:

```python
# Will:
# - Load extended tokenizer (276,172 tokens)
# - Resize model embeddings (+14,027 new embeddings)
# - Initialize new embeddings
# - Add LoRA adapter with embedding training enabled
```

### 2. Train Embeddings (Phase 0)

Train the new Korean token embeddings:

```bash
python3 script/train/train_00_plain_text.py --model medgemma-4b --epochs 3
```

This teaches the model to properly use the new Korean tokens.

### 3. Continue Training Pipeline

After embeddings are trained:
- Phase 1: Medical dictionary training
- Phase 2: MCQ with reasoning training

## Compatibility

✅ **Works with both:**
- MedGemma-4b (`google/medgemma-4b-it`)
- MedGemma-27b (`google/medgemma-27b-text-it`)

Both models share the same Gemma tokenizer base (262,145 tokens).

## Token Sources Documentation

### Special Characters (31 tokens)

From `data/tokenizer/special_chars_report.json`:
- Medical symbols: ℃, °, μ, ≥, ≤, ±, →
- Greek letters: α, β, γ
- Range indicators: ～, ∼
- Mathematical: ×, ÷
- Brackets: 「」

### Dictionary Terms (4,670 tokens)

From `data/tokenizer/dictionary_terms.txt` (cleaned):
- Korean medical terminology
- Disease names: 당뇨병, 고혈압, 심근경색
- Anatomy: 간, 폐, 심장
- Symptoms: 통증, 발열, 기침
- Treatments: 치료, 수술, 약물

**Quality:** Cleaned to remove parentheses, brackets, mixed English text

### Train02 Words (10,000 tokens)

From `data/tokenizer/train02_words.txt`:
- Korean words extracted from MCQ training data
- Medical context vocabulary
- Clinical terminology

## Build Statistics

Complete statistics in `tokenizer_stats.json`:

```json
{
  "base_model": "google/medgemma-4b-it",
  "original_vocab_size": 262145,
  "new_vocab_size": 276172,
  "tokens_actually_added": 14027,
  "special_chars": 31,
  "dictionary_terms": 4670,
  "train02_words": 10000,
  "duplicates_removed": 376,
  "tokenization_improvement": 1.11
}
```

## Notes

- **Tokenization improvement (1.11x)** is modest but expected for medical domain
- Real improvement comes during embedding training when model learns to utilize new tokens
- Special characters now tokenize as single tokens (3-5x improvement for symbols)
- Korean medical terms tokenize more efficiently (1.2-1.5x improvement)

## References

- Documentation: `research/00_tokenizer_embedding.md`
- Build script: `data/tokenizer/build_medgemma_tokenizer.py`
- Source data: `data/tokenizer/`
