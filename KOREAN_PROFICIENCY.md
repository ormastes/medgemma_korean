# Korean Proficiency Validation Guide

## Current Korean Data Status

### ✅ Korean Medical Data Available

| Type | Korean? | Train Samples | Val Samples | Korean Ratio |
|------|---------|---------------|-------------|--------------|
| **Type 1 (TEXT)** | ✓ Yes | 118,431 | 13,160 | ~89% Korean |
| **Type 2 (TEXT_REASONING)** | ✓ Yes | 23,018 | 2,558 | ~80% Korean |
| **Type 3 (WORD)** | ✗ No | 16,701 | 1,846 | Letter answers (A/B/C) |
| **Type 4 (WORD_REASONING)** | ✗ No | 7,957 | 885 | English reasoning |

**Total Korean Content:** 141,449 train + 15,718 val = **157,167 Korean samples**

### Why Types 3 & 4 Are Not Korean

- **Type 3 (WORD):** MCQ answers are single letters (A, B, C, D, E) - language-agnostic
- **Type 4 (WORD_REASONING):** Contains English reasoning in `<R>...<R/>` blocks
- **Questions are Korean** in Types 3 & 4, but **answers/reasoning are not**

## Korean Medical Benchmarks

### 1. KorMedMCQA (Primary Evaluation)

**Dataset:** `sean0042/KorMedMCQA`
- **Test samples:** 604
- **Language:** 100% Korean
- **Format:** Korean MCQs with Korean explanations
- **Target:** ≥90% accuracy
- **Status:** ✓ Available in `data/raw/by_source/kormedmcqa/`

**Use:** Primary metric for Korean medical knowledge

### 2. KMMLU-Medical (Korean MMLU)

**Dataset:** Korean Massive Multitask Language Understanding
- **Language:** 100% Korean
- **Subjects:** Multiple medical subjects
  - Anatomy
  - Pharmacology
  - Clinical Medicine
  - Public Health
- **Status:** ✓ Available in `data/raw/korean_datasets/kmmlu_medical/`

**Use:** Broad Korean medical knowledge assessment

### 3. MedQA-Korean

**Dataset:** `ChuGyouk/MedQA`
- **Samples:** 22,900 train
- **Language:** Korean (translated from English USMLE)
- **Format:** MCQ format
- **Status:** ✓ Available

**Use:** Korean medical exam questions

### 4. Korean Medical Sources (36 datasets)

Available in `data/raw/korean_datasets/`:
- Asan AMC Healthinfo (hospital data)
- KoMedInstruct-52k
- Korean medical textbooks
- Korean Wikipedia medical articles
- And 32 more...

## Validation Scripts

### 1. Check Korean Proficiency

```bash
# Validate all reviewed data
python3 scripts/validate_korean_proficiency.py --all

# Check specific file
python3 scripts/validate_korean_proficiency.py \
    --file data/reviewed/type1_text/train/data.jsonl \
    --sample-size 100

# Check benchmarks
python3 scripts/validate_korean_proficiency.py --benchmarks
```

### 2. Metrics Reported

- **Korean ratio:** % of Korean characters vs total
- **Medical terms:** Presence of Korean medical vocabulary
- **Text quality:** Character/word counts
- **Issues:** Missing medical terms, low Korean ratio

## Korean Proficiency Results

### Type 1 (TEXT) - 100% Korean ✅

```
Korean samples: 100%
Average Korean ratio: 89.0%
With medical terms: 80%

Sample:
"10세 소년의 증상인 고열, 부은 눈꺼풀, 퍼지는 발진, 코플릭 반점은
홍역을 강력하게 시사하며, 특히 예방 접종률이 최적화되지 않은 지역에서..."
```

### Type 2 (TEXT_REASONING) - 100% Korean ✅

```
Korean samples: 100%
Average Korean ratio: 80.5%
With medical terms: 68%

Sample:
"<R>자, 한번 생각해 봅시다. 13세 남자 환자가 야구공에 얼굴을 맞았고,
현재 좌측 안와 주위 부종이 있습니다. '안와 주위 부종'이라는 말을 들으면
안와 구조물을 생각하게 됩니다...<R/>진단은 안와 골절입니다."
```

### Type 3 (WORD) - Letters Only ⚠️

```
Korean samples: 0%
Answers: A, B, C, D, E (single letters)
Questions: Korean
```

### Type 4 (WORD_REASONING) - English Reasoning ⚠️

```
Korean samples: 0%
Reasoning: English in <R>...<R/> blocks
Final answer: Korean word/letter
```

## Recommended Validation Strategy

### Phase 1: Automated Metrics

```bash
# Run Korean proficiency check
python3 scripts/validate_korean_proficiency.py --all

# Expected results:
# - Type 1 & 2: >85% Korean ratio
# - Medical term coverage: >70%
```

### Phase 2: Benchmark Evaluation

```bash
# Evaluate on KorMedMCQA (primary)
python3 scripts/evaluate_kormedmcqa.py \
    --model models/final \
    --output results/kormedmcqa_eval.json

# Target: ≥90% accuracy
```

### Phase 3: KMMLU-Medical

```bash
# Evaluate on KMMLU medical subjects
python3 scripts/evaluate_kmmlu_medical.py \
    --model models/final \
    --output results/kmmlu_eval.json
```

### Phase 4: Manual Review (100 samples)

```bash
# Sample random Korean samples
python3 scripts/sample_for_review.py \
    --count 100 \
    --output manual_review_samples.jsonl

# Manual checks:
# 1. Korean grammar correctness
# 2. Medical terminology accuracy
# 3. Natural Korean flow
# 4. Cultural appropriateness
```

## Korean Medical Terminology Coverage

Common medical terms found in data:

```
환자 (patient)         - Present in 80%+ samples
진단 (diagnosis)       - Present in 70%+ samples
치료 (treatment)       - Present in 65%+ samples
증상 (symptoms)        - Present in 75%+ samples
질병 (disease)         - Present in 60%+ samples
약물 (medication)      - Present in 55%+ samples
검사 (examination)     - Present in 50%+ samples
수술 (surgery)         - Present in 40%+ samples
```

## Data Quality Indicators

### High Quality (✅)

- Type 1 & 2 text completions
- KorMedMCQA dataset
- Asan AMC Healthinfo (hospital data)
- Korean medical textbooks

### Medium Quality (⚠️)

- Translated medical content (MedQA-Korean)
- Generated Korean medical Q&A
- Wikipedia medical articles

### English Content (ℹ️)

- Type 3 answers (single letters - acceptable)
- Type 4 reasoning (English - needs improvement)

## Improving Type 4 Korean Content

### Current Issue

```json
{
  "prompt": "<|im_start|>user\n45세 남성, 당뇨병 진단...<|im_end|>",
  "completion": "<R>Let's think about diabetes...<R/>당뇨병"
}
```

### Solution 1: Re-generate with Korean Reasoning

```bash
# Use DeepSeek or GPT to translate reasoning
python3 scripts/translate_type4_reasoning.py \
    --input data/reviewed/type4_word_reasoning/train/data.jsonl \
    --output data/reviewed/type4_word_reasoning_korean/train/data.jsonl
```

### Solution 2: Use Korean Reasoning Data

Already available:
- `ChuGyouk/medical-o1-reasoning-SFT-Ko` (Korean reasoning)
- `ChuGyouk/ChainofDiagnosis-Ko` (Korean chain-of-thought)

## Validation Workflow

```
┌─────────────────────────────────────────────────────┐
│           Korean Proficiency Validation             │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Step 1: Automated Check                           │
│          - Korean ratio: ✓ >85%                     │
│          - Medical terms: ✓ >70%                    │
│                                                     │
│  Step 2: KorMedMCQA (604 samples)                  │
│          - Target: ≥90% accuracy                    │
│          - Korean medical knowledge                 │
│                                                     │
│  Step 3: KMMLU-Medical                             │
│          - Multiple subjects                        │
│          - Broad knowledge test                     │
│                                                     │
│  Step 4: Manual Review (100 samples)               │
│          - Grammar check                            │
│          - Medical accuracy                         │
│          - Natural Korean                           │
│                                                     │
│  Step 5: Deploy if all pass                        │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Quick Commands

```bash
# Check Korean proficiency
python3 scripts/validate_korean_proficiency.py --all

# Evaluate on KorMedMCQA
python3 scripts/train_loop_until_90.py --model medgemma-27b

# Check benchmarks
python3 scripts/validate_korean_proficiency.py --benchmarks

# Sample for manual review
head -100 data/reviewed/type1_text/validation/data.jsonl > korean_review.jsonl
```

## Summary

✅ **Strong Korean Coverage:**
- 157,167 Korean medical samples
- 89% Korean ratio in Type 1
- 80% Korean ratio in Type 2
- Rich medical terminology

⚠️ **Areas for Improvement:**
- Type 4 reasoning currently in English
- Can be improved with Korean reasoning data

📊 **Validation Tools:**
- Automated Korean proficiency checker
- KorMedMCQA benchmark (604 samples)
- KMMLU-Medical benchmark
- Manual review workflow

🎯 **Target Metrics:**
- KorMedMCQA: ≥90% accuracy
- Korean ratio: ≥85%
- Medical term coverage: ≥70%

---

**Current Status:** Korean proficiency is **strong** for Types 1 & 2 (75% of total data). Use KorMedMCQA as primary validation metric.
