# MedGemma Korean - Data & Model Preparation

This directory contains scripts to download, prepare, and refine all data and models required for training.

## Quick Start

```bash
# 1. Download all data (automatic)
python script/prepare/data/download_all.py

# 2. Download manual data (see MANUAL_DATA.md)
# - Medical dictionaries from StarDict

# 3. Download and prepare model
python script/prepare/model/prepare_all.py --model medgemma-4b

# 4. Refine all data
python script/prepare/refine/refine_all.py

# Ready for training!
python script/train/train_00_plain_text.py
```

---

## Directory Structure

```
script/prepare/
├── README.md                    # This file
├── __init__.py
│
├── data/                        # Data download scripts
│   ├── download_all.py          # Master download script
│   ├── download_korean_text.py  # Korean plain text (Wiki, C4)
│   ├── download_kormedmcqa.py   # KorMedMCQA dataset
│   ├── download_translation.py  # Korean-English pairs
│   ├── validate_manual_data.py  # Validate manual downloads
│   └── MANUAL_DATA.md           # Manual download instructions
│
├── model/                       # Model download/preparation
│   ├── prepare_all.py           # Master model preparation
│   ├── download_base_model.py   # Download from HuggingFace
│   ├── prepare_tokenizer.py     # Extend Korean tokenizer
│   ├── add_lora.py              # Add LoRA adapter
│   └── MANUAL_MODEL.md          # Model access instructions
│
└── refine/                      # Data refinement scripts
    ├── refine_all.py            # Master refinement script
    ├── refine_plain_text.py     # Clean wiki text
    ├── refine_kormedmcqa.py     # Transform MCQ format
    ├── refine_medical_dict.py   # Merge dictionaries
    ├── refine_translation.py    # Filter translation pairs
    └── enhance_with_llm.py      # LLM-based enhancement
```

---

## Data Requirements

### Automatic Downloads

| Dataset | Script | Output | Size |
|---------|--------|--------|------|
| Namu Wiki | `download_korean_text.py` | `data/01_raw/00_korean/namu_wiki/` | ~18GB |
| Wikipedia Korean | `download_korean_text.py` | `data/01_raw/00_korean/wikipedia-korean/` | ~3GB |
| C4 Korean | `download_korean_text.py` | `data/01_raw/00_korean/c4_korean/` | ~3GB |
| KorMedMCQA | `download_kormedmcqa.py` | `data/01_raw/02_kor_med_test/` | ~5MB |
| OPUS Tatoeba | `download_translation.py` | `data/01_raw/03_korean_english/` | ~100MB |

### Manual Downloads

| File | Location | Source |
|------|----------|--------|
| `korean_medical_dict.jsonl` | `data/01_raw/01_medical_dict/` | StarDict conversion |
| `korean_animal_medical_dict.jsonl` | `data/01_raw/01_medical_dict/` | StarDict conversion |
| `bilingual_medical_dict.json` | `data/01_raw/01_medical_dict/` | StarDict conversion |
| `bilingual_medical_dict_ko_en.json` | `data/01_raw/01_medical_dict/` | StarDict conversion |
| `bilingual_medical_dict_categorized.json` | `data/01_raw/01_medical_dict/` | StarDict conversion |

See `data/MANUAL_DATA.md` for detailed download instructions.

---

## Model Requirements

### Base Models

| Model | HuggingFace ID | Size | VRAM |
|-------|----------------|------|------|
| MedGemma 4B | `google/medgemma-4b-it` | 8.5GB | ~12GB (8-bit) |
| MedGemma 27B | `google/medgemma-27b-text-it` | 55GB | ~38GB (8-bit) |

**Prerequisites:**
1. HuggingFace account
2. Accept model license on HuggingFace
3. `huggingface-cli login`

See `model/MANUAL_MODEL.md` for detailed instructions.

---

## Refinement Pipeline

### Step 1: Plain Text (train_00)

```
data/01_raw/00_korean/         →  refine_plain_text.py  →  data/02_refined/00_plain_text/train.jsonl
  ├── namu_wiki/                                            Format: {"text": "..."}
  ├── wikipedia-korean/
  └── c4_korean/
```

**Processing:**
- Clean wiki markup
- Filter by Korean ratio (≥10%)
- Filter adult content
- Minimum length: 100 chars

### Step 2: KorMedMCQA (train_02)

```
data/01_raw/02_kor_med_test/   →  refine_kormedmcqa.py  →  data/02_refined/02_kor_med_test/
  ├── train.jsonl                                           ├── train.jsonl (1890 samples)
  └── test.jsonl                                            └── test.jsonl (604 samples)

                                                        →  data/02_refined/02_char_dict.json (special chars)
```

**Processing:**
- Transform answer format (1-5 → A-E)
- Extract special characters
- Validate format

**Output Format:**
```json
{"question": "...", "A": "...", "B": "...", "C": "...", "D": "...", "E": "...", "answer": "A"}
```

### Step 3: Medical Dictionary (train_01)

```
data/01_raw/01_medical_dict/   →  refine_medical_dict.py  →  data/02_refined/01_medical_dict.json
  ├── korean_medical_dict.jsonl                               [{"term": "고혈압", "definition": "Hypertension"}]
  ├── bilingual_medical_dict.json
  └── ...
```

**Processing:**
- Merge 5 dictionary sources
- Deduplicate by term
- Standardize format

### Step 4: Translation (train_01)

```
data/01_raw/03_korean_english/  →  refine_translation.py  →  data/02_refined/01_english_korean/
  └── opus_tatoeba_ko_en/                                     ├── en_to_ko.jsonl
                                                              └── ko_to_en.jsonl
```

**Processing:**
- Filter by length (5-500 chars)
- Validate Korean content
- Create bidirectional pairs

### Step 5: LLM Enhancement (Optional)

```
data/02_refined/02_kor_med_test/train.jsonl  →  enhance_with_llm.py  →  train_with_reasoning.jsonl
data/02_refined/01_medical_dict.json         →  enhance_with_llm.py  →  01_medical_dict_explained.json
```

**Processing:**
- Generate reasoning chains for MCQ
- Add Korean explanations to dictionary
- Requires: DeepSeek-7B-Chat on GPU

---

## Output Files Summary

After running all preparation scripts:

```
data/
├── 01_raw/                              # Raw downloaded data
│   ├── 00_korean/                       # Plain text sources
│   ├── 01_medical_dict/                 # Medical dictionaries (MANUAL)
│   ├── 02_kor_med_test/                 # KorMedMCQA
│   └── 03_korean_english/               # Translation pairs
│
└── 02_refined/                          # Refined training data
    ├── 00_plain_text/
    │   └── train.jsonl                  # ~1.5GB plain Korean text
    ├── 01_english_korean/
    │   ├── en_to_ko.jsonl               # English→Korean pairs
    │   └── ko_to_en.jsonl               # Korean→English pairs
    ├── 01_medical_dict.json             # 4000+ medical terms
    ├── 02_char_dict.json                # Special characters
    └── 02_kor_med_test/
        ├── train.jsonl                  # 1890 MCQ samples
        └── test.jsonl                   # 604 MCQ samples (eval)

model/
├── raw/                                 # Base models
│   ├── medgemma-4b/
│   └── medgemma-27b/
├── tokenizer/                           # Extended Korean tokenizer
│   ├── tokenizer.json
│   ├── new_tokens.txt                   # Added Korean tokens
│   └── tokenizer_info.json
└── raw_lora_added/                      # Models with LoRA
    ├── medgemma-4b/
    └── medgemma-27b/
```

---

## Training Scripts → Prepare Scripts Mapping

| Training Script | Required Preparation | Prepare Script |
|-----------------|---------------------|----------------|
| `train_00_plain_text.py` | Plain text data | `refine_plain_text.py` |
| `train_01_medical_dict.py` | Medical dictionary | `refine_medical_dict.py` |
| `train_01_mixed.py` | Dictionary + Translation | `refine_medical_dict.py`, `refine_translation.py` |
| `train_02_kor_med_test.py` | KorMedMCQA | `refine_kormedmcqa.py` |

---

## GPU Requirements

| Script | GPU | VRAM | Time |
|--------|-----|------|------|
| `download_*.py` | None | - | ~1 hour |
| `refine_*.py` | None | - | ~30 min |
| `prepare_tokenizer.py` | None | - | ~10 min |
| `add_lora.py` | Required | ~15GB | ~5 min |
| `enhance_with_llm.py` | Required | ~20GB | ~4 hours |

---

## Troubleshooting

### Download fails with 401/403
```bash
huggingface-cli login
# Then accept license on model page
```

### Out of memory during refinement
```bash
# Process in chunks
python script/prepare/refine/refine_plain_text.py --max-samples 100000
```

### Missing manual data
```bash
python script/prepare/data/validate_manual_data.py
# Follow instructions in MANUAL_DATA.md
```

### LLM enhancement fails
```bash
# Check GPU memory
python -c "import torch; print(torch.cuda.memory_summary())"

# Use different device
python script/prepare/refine/enhance_with_llm.py --device cuda:0
```

---

## Complete Pipeline

```bash
# Step 1: Download data
python script/prepare/data/download_all.py

# Step 2: Validate manual downloads
python script/prepare/data/validate_manual_data.py

# Step 3: Prepare model
python script/prepare/model/prepare_all.py --model medgemma-4b

# Step 4: Refine data
python script/prepare/refine/refine_all.py

# Step 5: (Optional) LLM enhancement
python script/prepare/refine/enhance_with_llm.py --task all --device cuda:1

# Ready for training!
```

---

## Version History

- 2024-12: Initial creation of prepare pipeline
- Migrated extraction scripts from `data/01_raw/` to `script/prepare/refine/`
