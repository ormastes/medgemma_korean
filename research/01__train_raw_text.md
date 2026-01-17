# 1.1 Train Raw Korean Texts (Continued Pretraining)

## Overview

Continued pretraining on Korean plain text using cross-entropy loss with sequentially packed tokens. This phase teaches the model Korean language patterns before medical-specific training.

## Data Sources & Conversion

### Input Sources (4 types)

| Source | Location | Original Format | Converted To | Base | Multiplier | Final |
|--------|----------|-----------------|--------------|------|------------|-------|
| Korean Plain Text | `data/01_raw/00_korean/` | Dataset (arrow) | Plain text | ~100K | 1x | ~100K |
| KorMedMCQA | `data/01_raw/02_kor_med_test/train.jsonl` | JSON (MCQ) | Q&A text | 1,890 | **5x** | ~9,450 |
| Medical Dictionary | `data/02_refined/01_medical_dict.json` | JSON (term-def) | Definition text | 4,071 | **5x** | ~20,355 |
| Character Dictionary | `data/02_refined/02_char_dict.json` | JSON (symbol-def) | Explanation text | 89 | **5x** | ~445 |

**Total Base:** ~106K samples
**After Selective Multiplication:** ~130K samples (medical data 5x, randomly mixed)

### Format Conversions

#### 1. Korean Plain Text (Namu Wiki, Wikipedia, C4)

**Original:**
```json
{
  "namespace": 0,
  "title": "당뇨병",
  "text": "[목차] '''당뇨병'''은 [[혈당]] 조절 장애로 인한 [[대사 질환]]이다...",
  "contributors": ["user1", "user2"],
  "id": "123"
}
```

**Converted:**
```
당뇨병은 혈당 조절 장애로 인한 대사 질환이다...
```

**Cleaning:**
- Remove wiki markup: `[[ ]]`, `{{ }}`, `'''`, etc.
- Remove navigation: `[목차]`, `[각주]`
- Remove HTML tags: `<ref>`, `</ref>`
- Filter: min 100 chars, ≥10% Korean ratio

---

#### 2. KorMedMCQA (Medical MCQ)

**Original:**
```json
{
  "question": "항문압 측정 검사에서 항문 압력이 증가하는 경우는?",
  "A": "직장질루(rectovaginal fistula)",
  "B": "항문열창(anal fissure)",
  "C": "대변실금(fecal incontinence)",
  "D": "대변메막힘(fecal impaction)",
  "E": "직장탈출증(rectal prolapse)",
  "answer": 2
}
```

**Converted:**
```
항문압 측정 검사에서 항문 압력이 증가하는 경우는?
선택지: 직장질루, 항문열창, 대변실금, 대변메막힘, 직장탈출증
정답: 항문열창

항문열창(anal fissure)은 의학적으로 중요한 개념이다.
```

**Benefits:**
- Exposes model to Korean medical questions
- Links Korean ↔ English medical terms
- Natural Q&A format for language modeling

---

#### 3. Medical Dictionary

**Original:**
```json
{
  "term": "고혈압",
  "definition": "Hypertension. 혈압이 정상 범위보다 높은 상태"
}
```

**Converted:**
```
고혈압(Hypertension)은 혈압이 정상 범위보다 높은 상태를 의미한다.
```

**Benefits:**
- Teaches Korean-English medical term mapping
- Natural definition sentences
- Core medical vocabulary

---

#### 4. Character Dictionary (Medical Symbols)

**Original:**
```json
{
  "term": "↑",
  "definition": "increased, elevated, or upward trend"
}
```

**Converted:**
```
의학 기호 ↑는 increased, elevated, or upward trend를 나타낸다.
```

**Benefits:**
- Teaches medical symbols (↑, ↓, ±, etc.)
- Common in medical literature
- Helps with arrow notation understanding

---

## Selective Medical Data Multiplication Strategy

### Why Only Medical Data 5x?

1. **Medical vocabulary exposure**: Only ~6K medical samples vs ~100K+ general Korean
2. **Prevents drowning**: Without multiplication, medical terms would be severely underrepresented
3. **Balanced learning**: Ensures model learns both Korean fluency AND medical terminology
4. **Random mixing**: All samples randomly mixed to prevent sequential bias

### Distribution After Selective Multiplication

From 106K base samples → 130K mixed samples (medical data 5x):

| Source | Base | Multiplier | Final | Ratio |
|--------|------|------------|-------|-------|
| Korean Plain | 100,000 | 1x | 100,000 | 77% |
| MCQ | 1,890 | **5x** | 9,450 | 7% |
| Medical Dict | 4,071 | **5x** | 20,355 | 16% |
| Char Dict | 89 | **5x** | 445 | <1% |

**Note:** Only medical data (MCQ, dictionaries) is multiplied. General Korean text is not multiplied to avoid excessive file size and redundancy.

---

## How Much Training is Enough?

### Token Requirements for Continued Pretraining

**General Rule:** 10-100x model parameters in tokens

| Model Size | Parameters | Ideal Token Range | Practical Range |
|------------|------------|-------------------|-----------------|
| MedGemma-4B | 4B | 40B-400B tokens | 1B-10B tokens |
| MedGemma-27B | 27B | 270B-2.7T tokens | 10B-100B tokens |

**Domain Adaptation** (our use case): 0.1%-1% of full pretraining data
- General pretraining: 1-2 trillion tokens
- Domain adaptation: 1-20 billion tokens
- Minimum effective: 100M-1B tokens

### Our Dataset Token Analysis

| Dataset Size | Samples | Avg Tokens/Sample | Total Tokens | Use Case |
|--------------|---------|-------------------|--------------|----------|
| **Quick Test** | 5,000 | ~800 | ~4M | Initial validation |
| **Small Scale** | 10,000 | ~800 | ~8M | Quick adaptation |
| **Medium Scale** | 30,000 | ~800 | ~24M | Good adaptation |
| **Large Scale** | 50,000 | ~800 | ~40M | Strong adaptation |
| **Full Dataset** | 130,000 | ~800 | **~104M** | **Maximum adaptation** |

### Recommended Training Stages

#### Stage 1: Quick Validation (5K-10K samples)
**Purpose:** Verify training setup and tokenizer integration

```bash
python script/train/train_00_plain_text.py \
    --model medgemma-4b \
    --tokenizer-path model/tokenizer/medgemma_ded_med_normal \
    --max-samples 5000 \
    --epochs 1 \
    --device cuda:0
```

**Expected:**
- Training time: 1-2 hours
- Token count: ~4M tokens
- Perplexity: 4.0-5.0 (baseline improvement)
- KorMedMCQA: 20-25% (slight change expected)

---

#### Stage 2: Initial Adaptation (30K-50K samples)
**Purpose:** Learn Korean medical vocabulary and patterns

```bash
python script/train/train_00_plain_text.py \
    --model medgemma-4b \
    --tokenizer-path model/tokenizer/medgemma_ded_med_normal \
    --max-samples 30000 \
    --epochs 2 \
    --device cuda:0
```

**Expected:**
- Training time: 6-10 hours
- Token count: ~48M tokens (2 epochs × 24M)
- Perplexity: 3.0-4.0
- KorMedMCQA: 25-35% (vocabulary learning)

---

#### Stage 3: Full Adaptation (100K-130K samples)
**Purpose:** Maximum Korean medical language proficiency

```bash
python script/train/train_00_plain_text.py \
    --model medgemma-4b \
    --tokenizer-path model/tokenizer/medgemma_ded_med_normal \
    --max-samples 130000 \
    --epochs 1-2 \
    --device cuda:0
```

**Expected:**
- Training time: 12-24 hours
- Token count: 104M-208M tokens
- Perplexity: <3.0
- KorMedMCQA: 30-40% (baseline ready for SFT)

---

### Literature References

**RedWhale (Korean Medical LLM):**
- Dataset: 10B tokens for medical CPT
- Result: Strong Korean medical performance
- Paper: https://arxiv.org/html/2408.11294v1

**Llama 3 Pretraining:**
- General: 15T tokens
- Code: 3T tokens
- Multilingual: 5% of total

**Domain Adaptation Studies:**
- BioBERT: 4.5B tokens (PubMed + PMC)
- SciBERT: 3.17B tokens (scientific papers)
- ClinicalBERT: 2B tokens (clinical notes)

**Key Insight:** 100M-500M tokens is sufficient for effective domain adaptation when starting from a strong base model like MedGemma.

### Practical Recommendations

**For Quick Experiments (1-2 hours):**
- Use 5K-10K samples
- 1 epoch
- Validates tokenizer and setup

**For Production (6-12 hours):**
- Use 30K-50K samples
- 2-3 epochs
- Balances time and quality

**For Maximum Quality (12-24 hours):**
- Use full 130K samples
- 1-2 epochs
- Best foundation for medical SFT

**Note:** After CPT, supervised fine-tuning (train_01, train_02) will significantly improve task performance. CPT goal is vocabulary and fluency, not task performance.

---

## Training Process

### 1. Data Preparation

```bash
# Run conversion script
python script/create_raw_text_dataset.py

# Output: data/01_raw/00_korean/korean/raw_text_mixed.jsonl
# Format: {"text": "plain text content"}
```

### 2. Training

```bash
# Train with plain text
python script/train/train_00_plain_text.py \
    --model medgemma-4b \
    --epochs 1 \
    --max-samples 130000
```

**Training config:**
- Loss: Cross-entropy (next-token prediction)
- Packing: Sequential packing with EOS between docs
- Batch size: 2 (with gradient checkpointing)
- Gradient accumulation: 16 steps
- Learning rate: 2e-5

### 3. Validation

**Metrics:**
1. **Validation loss** (held-out Korean text)
2. **Validation perplexity**: `ppl = exp(loss)`
3. **KorMedMCQA accuracy** (periodic checks)

**Stop conditions:**
- Val loss stops improving (<0.01 improvement)
- Patience: 5-10 evaluations without improvement
- Target perplexity: <3.0

---

## Expected Results

| Metric | Target | Notes |
|--------|--------|-------|
| Validation PPL | <3.0 | Good Korean fluency |
| KorMedMCQA Accuracy | 30-40% | Baseline (no reasoning) |
| Training time | 4-6 hours | medgemma-4b, 530K samples |

**Why low MCQ accuracy?**
- This is CPT (continued pretraining), not SFT
- Model learns Korean + medical terms, not reasoning
- MCQ accuracy improves in later stages (train_01, train_02)

---

## Output

```
model/00_trained/medgemma-4b/
├── adapter_config.json      # LoRA config
├── adapter_model.safetensors # Trained weights
├── tokenizer/               # Extended tokenizer
└── training_info.json       # Metrics
```

**Key info:**
- Trainable params: 1.59B (26.78%) if embeddings trained
- Training includes Korean embeddings for extended vocab
- Uses rsLoRA for stable training

---

## Next Steps

After train_00 (raw text CPT):

1. **train_01**: Medical dictionary SFT
2. **train_02**: MCQ with reasoning SFT
3. **Loop training**: Alternate 01 ↔ 02 until 90% accuracy

---

## References

- RedWhale paper: https://arxiv.org/html/2408.11294v1
- Unsloth CPT guide: https://unsloth.ai/blog/contpretraining
- HuggingFace LM tutorial: https://huggingface.co/docs/transformers/tasks/language_modeling
