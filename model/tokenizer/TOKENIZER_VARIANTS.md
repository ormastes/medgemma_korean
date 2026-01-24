# MedGemma Korean Tokenizer Variants

**Created:** 2026-01-20
**Approach:** BPE (Byte Pair Encoding) + Dictionary Terms
**Base Model:** google/medgemma-4b-it (262,145 tokens)

## Overview

Three progressive tokenizer variants with increasing Korean medical vocabulary:

| Variant | Tokens Added | New Vocab Size | Components | Use Case |
|---------|--------------|----------------|------------|----------|
| **medgemma_dedicated** | 4,657 | 266,802 | Special chars + Dictionary | Medical terminology only |
| **medgemma_ded_medical** | 10,681 | 272,826 | Dedicated + Medical BPE | Medical domain focus |
| **medgemma_ded_med_normal** | 20,576 | 282,721 | Dedicated + Medical + General BPE | Maximum coverage |

---

## Variant 1: medgemma_dedicated

**Purpose:** Core medical terminology without general language expansion

### Components
- **Special Characters (31):** Medical symbols (℃, ≥, μ, →, etc.)
- **Dictionary Terms (4,670):** Cleaned Korean medical terms

### Statistics
```
Original vocab:      262,145
New vocab:           266,802
Tokens added:        4,657
Increase:            1.8%
Tokenization:        1.13x improvement
```

### Token Sources
1. `data/tokenizer/special_chars_report.json` → 31 chars
2. `data/tokenizer/dictionary_terms.txt` → 4,670 terms (cleaned)

### Sample Terms
```
당뇨병 (diabetes)
고혈압 (hypertension)
심근경색 (myocardial infarction)
간경변 (liver cirrhosis)
뇌졸중 (stroke)
```

### Use Case
- **Minimal footprint:** Small vocabulary increase
- **Medical focus:** Only medical terminology
- **Faster training:** Fewer embeddings to learn
- **Best for:** Dedicated medical applications, limited resources

---

## Variant 2: medgemma_ded_medical

**Purpose:** Medical domain with learned subword vocabulary via BPE

### Components
- **Dedicated (4,701):** Special chars + Dictionary terms
- **Medical BPE (8,502):** Subwords learned from medical corpus

### Statistics
```
Original vocab:      262,145
New vocab:           272,826
Tokens added:        10,681
Increase:            4.1%
Tokenization:        1.18x improvement
```

### Token Sources
1. Dedicated tokens (4,701)
2. `data/tokenizer/bpe_medical_10k_vocab.txt` → 8,502 BPE tokens

### Medical BPE Corpus
- **Sources:** MCQ data (1,890 samples) + Medical dict (4,049 entries)
- **Unique lines:** 12,027
- **Corpus size:** 1.0 MB
- **BPE training:** vocab_size=10,000, min_frequency=3

### BPE Token Examples
```
병 (disease suffix):  90 BPE variants
증 (symptom suffix): 139 BPE variants
염 (inflammation):   103 BPE variants
당뇨 (diabetes):       7 BPE variants
고혈압 (hypertension): 5 BPE variants
심근 (myocardium):     3 BPE variants
```

### Use Case
- **Balanced:** Medical coverage + efficient tokenization
- **BPE learned:** Optimal subword segmentation for Korean medical text
- **Recommended for:** Most medical applications
- **Best for:** Medical QA, clinical notes, medical documentation

---

## Variant 3: medgemma_ded_med_normal

**Purpose:** Full Korean language support (medical + general)

### Components
- **Dedicated (4,701):** Special chars + Dictionary
- **Medical BPE (8,502):** Medical subwords
- **General BPE (14,309):** General Korean subwords

### Statistics
```
Original vocab:      262,145
New vocab:           282,721
Tokens added:        20,576
Increase:            7.8%
Tokenization:        1.18x improvement
```

### Token Sources
1. Dedicated tokens (4,701)
2. Medical BPE tokens (8,502)
3. `data/tokenizer/bpe_general_20k_vocab.txt` → 14,309 BPE tokens

### General BPE Corpus
- **Source:** Plain Korean text (100,000 samples)
- **Unique lines:** 99,990
- **Corpus size:** 85.3 MB
- **BPE training:** vocab_size=20,000, min_frequency=3

### General BPE Token Examples
```
Common morphemes:
  하다 (to do):   20 BPE variants
  되다 (to become): 1 BPE variant
  있다 (to exist):  7 BPE variants
  이다 (to be):    38 BPE variants
```

### Use Case
- **Maximum coverage:** Medical + general Korean
- **Versatile:** Handles diverse Korean text
- **Higher cost:** More embeddings to train
- **Best for:** Mixed content, general medical chatbot, Korean fluency

---

## Tokenization Comparison

Test sentences showing token count reduction:

| Sentence | Original | Dedicated | Medical | Full |
|----------|----------|-----------|---------|------|
| 당뇨병 환자의 혈당 조절이 필요합니다. | 13 | 12 | 11 | 11 |
| 고혈압 치료를 위해 약물을 복용해야 합니다. | 13 | 13 | 12 | 12 |
| 심근경색의 주요 증상은 흉통입니다. | 13 | 10 | 10 | 10 |
| 체온은 36.5℃이며 혈압은 120/80 mmHg입니다. | 23 | 22 | 20 | 20 |
| 혈액검사 결과 포도당 ≥126 mg/dL입니다. | 17 | 13 | 14 | 14 |

**Average Improvement:**
- Dedicated: 1.13x
- Medical: 1.18x
- Full: 1.18x

---

## File Locations

```
model/tokenizer/
├── medgemma_dedicated/              Variant 1 (4,657 tokens)
│   ├── tokenizer.json
│   ├── tokenizer.model
│   ├── added_tokens.json
│   └── ...
├── medgemma_ded_medical/            Variant 2 (10,681 tokens)
│   ├── tokenizer.json
│   ├── tokenizer.model
│   ├── added_tokens.json
│   └── ...
├── medgemma_ded_med_normal/         Variant 3 (20,576 tokens)
│   ├── tokenizer.json
│   ├── tokenizer.model
│   ├── added_tokens.json
│   └── ...
├── medgemma_dedicated_stats.json
├── medgemma_ded_medical_stats.json
└── medgemma_ded_med_normal_stats.json
```

---

## Usage

### Load Tokenizer

```python
from transformers import AutoTokenizer

# Variant 1: Dedicated
tokenizer = AutoTokenizer.from_pretrained(
    "model/tokenizer/medgemma_dedicated"
)

# Variant 2: Medical (Recommended)
tokenizer = AutoTokenizer.from_pretrained(
    "model/tokenizer/medgemma_ded_medical"
)

# Variant 3: Full
tokenizer = AutoTokenizer.from_pretrained(
    "model/tokenizer/medgemma_ded_med_normal"
)
```

### Test Tokenization

```python
text = "당뇨병 환자의 혈당 조절이 필요합니다."

tokens = tokenizer.tokenize(text)
print(f"Tokens ({len(tokens)}): {tokens}")

# Dedicated: 12 tokens
# Medical:   11 tokens
# Full:      11 tokens
# Original:  13 tokens
```

---

## Building Process

### Step 1: Prepare Medical Corpus
```bash
cd data/tokenizer
python3 train_bpe_medical.py --vocab-size 10000
```

Output:
- `medical_corpus.txt` (1.0 MB, 12,027 lines)
- `bpe_medical_10k.json` (BPE model)
- `bpe_medical_10k_vocab.txt` (8,502 Korean tokens)

### Step 2: Prepare General Corpus
```bash
python3 train_bpe_general.py --vocab-size 20000 --max-lines 100000
```

Output:
- `general_corpus.txt` (85.3 MB, 99,990 lines)
- `bpe_general_20k.json` (BPE model)
- `bpe_general_20k_vocab.txt` (14,309 Korean tokens)

### Step 3: Build Tokenizer Variants
```bash
python3 build_tokenizer_variants.py --variant all
```

Output:
- 3 tokenizer directories
- 3 stats JSON files

### Rebuild Single Variant
```bash
# Build only one variant
python3 build_tokenizer_variants.py --variant dedicated
python3 build_tokenizer_variants.py --variant medical
python3 build_tokenizer_variants.py --variant full
```

---

## Recommendation

### For Medical Applications
✅ **Use `medgemma_ded_medical`**

**Reasons:**
- Best balance between coverage and efficiency
- Medical BPE learns optimal Korean medical subwords
- 1.18x tokenization improvement
- 10,681 tokens is manageable for embedding training

### For General Korean + Medical
✅ **Use `medgemma_ded_med_normal`**

**Reasons:**
- Full Korean language coverage
- Handles both medical and general text
- Best for chatbots, mixed-domain applications

### For Minimal Deployment
✅ **Use `medgemma_dedicated`**

**Reasons:**
- Smallest vocabulary increase (4,657 tokens)
- Fastest embedding training
- Core medical terms covered

---

## Next Steps

1. **Choose variant** based on use case
2. **Initialize LoRA** with chosen tokenizer
3. **Train embeddings** (Phase 0) to teach model new Korean tokens

```bash
# Example: Using medical variant
python3 init_lora_with_extended_tokenizer.py \
    --model medgemma-4b \
    --tokenizer model/tokenizer/medgemma_ded_medical

python3 script/train/train_00_plain_text.py \
    --model medgemma-4b \
    --tokenizer model/tokenizer/medgemma_ded_medical \
    --epochs 3
```

---

## Technical Details

### BPE Training Parameters

**Medical BPE:**
- Vocab size: 10,000
- Min frequency: 3
- Algorithm: Byte Pair Encoding
- Pre-tokenizer: Whitespace
- Learned: 8,502 Korean tokens

**General BPE:**
- Vocab size: 20,000
- Min frequency: 3
- Algorithm: Byte Pair Encoding
- Pre-tokenizer: Whitespace
- Learned: 14,309 Korean tokens

### Deduplication

When combining token sources, deduplication occurs:
- Dedicated: 4,701 candidates → 4,657 added (44 already existed)
- Medical: 12,277 candidates → 10,681 added (1,596 already existed)
- Full: 23,027 candidates → 20,576 added (2,451 already existed)

---

## References

- Documentation: `research/00_tokenizer_embedding.md`
- BPE medical script: `data/tokenizer/train_bpe_medical.py`
- BPE general script: `data/tokenizer/train_bpe_general.py`
- Builder script: `data/tokenizer/build_tokenizer_variants.py`
