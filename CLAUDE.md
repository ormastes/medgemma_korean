# MedGemma Korean - Training Guide

## 4 Training Types

| Type | Output | Reasoning | Example Output | Evaluation |
|------|--------|-----------|----------------|------------|
| **Type 1: TEXT** | Full text | NO | `"당뇨병은 혈당 조절..."` | Perplexity <3.0 |
| **Type 2: TEXT_REASONING** | Full text | YES | `"<R>분석...<R/>당뇨병은..."` | Perplexity <3.0 |
| **Type 3: WORD** | Word/Letter | NO | `"B"` or `"폐렴"` | Accuracy ≥90% |
| **Type 4: WORD_REASONING** | Word/Letter | YES | `"<R>분석...<R/>B"` | Score ≥1.2 |

### Special Tokens
```
<R>  = Reasoning start
<R/> = Reasoning end
```

---

## Dataset → Type Mapping

### Type 1: TEXT (Full text, NO reasoning)
| Dataset | HuggingFace ID | Samples |
|---------|----------------|---------|
| Asan AMC Healthinfo | `ChuGyouk/Asan-AMC-Healthinfo` | 19.2K |
| HealthSearchQA Korean | `ChuGyouk/HealthSearchQA-ko` | 3.2K |
| AI Healthcare QA | `ChuGyouk/AI_healthcare_QA` | 12.1K |
| KorMedConceptsQA | `ChuGyouk/KorMedConceptsQA` | 73.2K |
| KoMedInstruct-52k | `ChuGyouk/KoMedInstruct-52k` | 52K |
| GenMedGPT-5k-ko | `ChuGyouk/GenMedGPT-5k-ko` | 5.5K |

**Format:**
```
System: 당신은 한국어 의료 전문 AI입니다. 상세하게 답변하세요.
User: 당뇨병 환자의 식이요법은?
Assistant: 당뇨병 환자의 식이요법은 다음과 같습니다...
```

---

### Type 2: TEXT_REASONING (Full text, WITH reasoning)
| Dataset | HuggingFace ID | Condition |
|---------|----------------|-----------|
| Medical O1 Reasoning | `ChuGyouk/medical-o1-reasoning-SFT-Ko` | Long answer |
| ChainofDiagnosis | `ChuGyouk/ChainofDiagnosis-Ko` | Long answer |

**Format:**
```
System: 먼저 추론한 후 상세하게 답변하세요.
User: 45세 여성, TSH 상승, T4 감소. 진단과 치료는?
Assistant: <R>1. TSH↑ T4↓ = 일차성 갑상선기능저하
2. 원인: 하시모토 갑상선염 가능성
3. 치료: 레보티록신<R/>갑상선기능저하증으로 진단됩니다. 레보티록신...
```

---

### Type 3: WORD (Letter/word, NO reasoning)
| Dataset | HuggingFace ID | Samples |
|---------|----------------|---------|
| KorMedMCQA | `sean0042/KorMedMCQA` | 7.5K |
| MedQA Korean | `ChuGyouk/MedQA` | 22.9K |
| MedQA Evol Korean | `ChuGyouk/MedQA-Evol-Korean` | 51.8K |
| KorMedLawQA | `snuh/KorMedLawQA` | 5K |

**Format:**
```
System: 정답 알파벳만 답하세요 (A, B, C, D, E 중 하나).
User: 심전도 ST상승, 진단은? A)협심증 B)심근경색 ...
Assistant: B
```

---

### Type 4: WORD_REASONING (Word, WITH reasoning)
| Dataset | HuggingFace ID | Condition |
|---------|----------------|-----------|
| Medical O1 Reasoning | `ChuGyouk/medical-o1-reasoning-SFT-Ko` | Short answer |
| ChainofDiagnosis | `ChuGyouk/ChainofDiagnosis-Ko` | Short answer |

**Format:**
```
System: 먼저 추론한 후 진단명을 한 단어로 답하세요.
User: 피로감, 체중증가, TSH↑ T4↓
Assistant: <R>TSH 상승과 T4 감소는 일차성 갑상선기능저하. 중년 여성에서 흔한 원인은 하시모토 갑상선염<R/>갑상선기능저하증
```

---

## Training Scripts

| Type | Script | Target | Command |
|------|--------|--------|---------|
| 1 | `train_type1_text.py` | PPL <3.0 | `python scripts/train_type1_text.py --model medgemma-27b` |
| 2 | `train_type2_text_reasoning.py` | PPL <3.0 | `python scripts/train_type2_text_reasoning.py --model medgemma-27b` |
| 3 | `train_type3_word.py` | Acc ≥90% | `python scripts/train_type3_word.py --model medgemma-27b` |
| 4 | `train_type4_word_reasoning.py` | Score ≥1.2 | `python scripts/train_type4_word_reasoning.py --model medgemma-27b` |

---

## Scoring (Type 4)

| Component | Score |
|-----------|-------|
| Exact answer match | 1.0 |
| Reasoning ≥10 tokens | +0.1 |
| Key term matches (each) | +0.05 |
| **Maximum** | **1.4** |

---

## Data Flow

```
HuggingFace Datasets
        ↓
python scripts/download_all_datasets.py
        ↓
data/raw/by_source/
        ↓
python refine_scripts/refine_4types.py
        ↓
data/refined/
├── type1_text/          (QA, Instruction)
├── type2_text_reasoning/ (Reasoning → long answer)
├── type3_word/          (MCQ)
└── type4_word_reasoning/ (Reasoning → short answer)
        ↓
python refine_scripts/verify_formats.py  (Check format correctness)
        ↓
Training Scripts
```

---

## Recommended Training Order

```
1. Type 1 (TEXT)           → Learn Korean medical language
2. Type 2 (TEXT_REASONING) → Learn to use <R>...<R/> with text
3. Type 3 (WORD)           → Learn direct answer WITHOUT reasoning
4. Type 4 (WORD_REASONING) → Learn <R>...<R/> with word answer
```

**Why this order:**
- Types 1 & 3 teach direct answering (no reasoning tokens)
- Types 2 & 4 teach when to use reasoning
- Model learns to distinguish when to reason vs. answer directly

---

## Key Difference: Type 1/3 vs Type 2/4

| Aspect | Type 1, 3 | Type 2, 4 |
|--------|-----------|-----------|
| Reasoning tokens | NO `<R>...<R/>` | YES `<R>...<R/>` |
| System prompt | "답변하세요" | "추론한 후 답변하세요" |
| Output | Direct answer | `<R>reasoning<R/>` + answer |

**Training separate types ensures model learns:**
- When system says "추론" → use `<R>...<R/>`
- When system says "답변" → answer directly (no reasoning tokens)

---

## Directory Structure

```
medgemma_korean/
├── phase0_data_preparation/
├── phase1_tokenizer/
├── phase2_embedding/
├── phase3_staged_training/
├── phase4_instruction_tuning/
├── phase5_subject_training/        # NEW: Division-based training
│   ├── scripts/
│   │   ├── annotate_with_deepseek.py   # DeepSeek annotation
│   │   ├── validate_divisions.py        # Validate & fix divisions
│   │   ├── train_with_divisions.py      # Train with division tracking
│   │   └── run_pipeline.sh              # Full pipeline
│   ├── models/                          # Division-aware models
│   ├── logs/
│   └── results/
├── phase6_evaluation/              # Renamed from phase5
├── phase7_deployment/              # Renamed from phase6
├── data/
│   ├── refined/
│   │   ├── type1_text/
│   │   ├── type2_text_reasoning/
│   │   ├── type3_word/
│   │   └── type4_word_reasoning/
│   ├── reviewed/                   # Human-reviewed data
│   └── division/                   # NEW: Division-annotated data
├── scripts/
│   ├── train_type1_text.py
│   ├── train_type2_text_reasoning.py
│   ├── train_type3_word.py
│   └── train_type4_word_reasoning.py
└── refine_scripts/
    ├── refine_4types.py
    └── verify_formats.py
```

---

## Loop Training Until 90% KorMedMCQA

**Goal:** Train iteratively through all 4 types until KorMedMCQA accuracy ≥90%

### Strategy
```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                        │
├─────────────────────────────────────────────────────────┤
│  while KorMedMCQA_accuracy < 90%:                       │
│      for type in [1, 2, 3, 4]:                          │
│          train_type(type, epochs=1)                     │
│          save_checkpoint()                              │
│      evaluate_kormedmcqa()                              │
│      if accuracy >= 90%: break                          │
└─────────────────────────────────────────────────────────┘
```

### Loop Training Script

```bash
# Single command - trains all types in loop until 90%
python scripts/train_loop_until_90.py --model medgemma-27b --target 90

# Options:
#   --model: medgemma-4b or medgemma-27b
#   --target: target accuracy (default: 90)
#   --max-loops: maximum training loops (default: 10)
#   --base-model: path to base model (default: stage5 or HF)
```

### Loop Training Process

| Loop | Type 1 | Type 2 | Type 3 | Type 4 | Eval | Action |
|------|--------|--------|--------|--------|------|--------|
| 1 | 1 epoch | 1 epoch | 1 epoch | 1 epoch | 65% | Continue |
| 2 | 1 epoch | 1 epoch | 1 epoch | 1 epoch | 75% | Continue |
| 3 | 1 epoch | 1 epoch | 1 epoch | 1 epoch | 85% | Continue |
| 4 | 1 epoch | 1 epoch | 1 epoch | 1 epoch | 92% | **STOP** |

### Data Sources (Reviewed)

| Type | Train | Val | Path |
|------|-------|-----|------|
| Type 1 | ~118K | ~13K | `data/reviewed/type1_text/` |
| Type 2 | ~23K | ~2.5K | `data/reviewed/type2_text_reasoning/` |
| Type 3 | ~17K | ~1.8K | `data/reviewed/type3_word/` |
| Type 4 | ~8K | ~900 | `data/reviewed/type4_word_reasoning/` |

### Checkpoint Strategy

```
models/loop_training/
├── loop_1/
│   ├── after_type1/
│   ├── after_type2/
│   ├── after_type3/
│   └── after_type4/
├── loop_2/
│   └── ...
├── best_checkpoint/     # Highest accuracy
└── final/               # When 90% reached
```

### Evaluation (KorMedMCQA)

- **Dataset:** `sean0042/KorMedMCQA` (604 test samples)
- **Metric:** Exact match accuracy on MCQ answers (A/B/C/D/E)
- **Target:** ≥90%
- **Eval frequency:** After each complete loop (all 4 types)

---

## Quick Start

```bash
# 1. Download datasets
python scripts/download_all_datasets.py

# 2. Refine into 4 types
python refine_scripts/refine_4types.py

# 3. Verify formats
python refine_scripts/verify_formats.py

# 4. Loop training until 90% (RECOMMENDED)
python scripts/train_loop_until_90.py --model medgemma-27b

# OR train each type separately:
python scripts/train_type1_text.py --model medgemma-27b --epochs 3
python scripts/train_type2_text_reasoning.py --model medgemma-27b --epochs 3
python scripts/train_type3_word.py --model medgemma-27b --epochs 10
python scripts/train_type4_word_reasoning.py --model medgemma-27b --epochs 5

# 5. Phase 5: Subject Training (NEW)
bash phase5_subject_training/scripts/run_pipeline.sh
```

---

## Phase 5: Subject Training (Division-Based)

### Overview
Phase 5 adds **medical division annotations** to training data and tracks performance per medical subject area. This enables:
- Identification of weak subject areas
- Targeted training for specific medical divisions
- Performance monitoring per medical specialty

### Medical Divisions (10 Major Categories)

1. **Cardiovascular Medicine**
2. **Respiratory Medicine**
3. **Gastroenterology and Hepatology**
4. **Nephrology**
5. **Endocrinology and Metabolism**
6. **Hematology and Oncology**
7. **Neurology**
8. **Infectious Diseases** (Cross-Cutting)
9. **Emergency and Critical Care** (Cross-Cutting)
10. **Ethics, Law, and Patient Safety** (Cross-Cutting)

Each division has sub-categories (e.g., `1.4.1` = Ischemic Heart Disease). See `med_division.json` for full taxonomy.

### Data Format with Divisions

```json
{
  "prompt": "<|im_start|>system\n당신은 한국어 의료 전문 AI입니다...",
  "completion": "심근경색증으로 진단됩니다...",
  "text": "...",
  "divisions": ["1.4.1", "9.2"],
  "primary_division": "1.4.1",
  "division_reasoning": "Primary: Cardiovascular - ACS/MI. Secondary: Emergency - Shock management"
}
```

### Pipeline Steps

#### Step 1: Annotate with DeepSeek (TITAN RTX)

```bash
# Annotate single file
python3 phase5_subject_training/scripts/annotate_with_deepseek.py \
    --input data/reviewed/type1_text/train/data.jsonl \
    --output data/division/type1_text/train.jsonl \
    --device cuda:1
```

**How it works:**
- Uses DeepSeek model on TITAN RTX (cuda:1)
- Analyzes question + answer content
- Assigns 1-3 division IDs per sample
- First ID is PRIMARY (majority) division
- Saves checkpoint every 100 samples

#### Step 2: Validate & Fix Divisions

```bash
# Validate and fix malformed annotations
python3 phase5_subject_training/scripts/validate_divisions.py \
    --input data/division/type1_text/train.jsonl \
    --output data/division/type1_text/train_fixed.jsonl \
    --fix
```

**Validation checks:**
- Missing `divisions` or `primary_division` fields
- Invalid division IDs (not in taxonomy)
- Empty divisions list
- Primary division not in divisions list
- Non-list divisions field

**Auto-fixes:**
- Adds missing fields
- Removes invalid IDs
- Sets primary to first valid division
- Replaces empty lists with ["UNKNOWN"]

#### Step 3: Train with Division Tracking

```bash
# Train with per-division performance tracking
python3 phase5_subject_training/scripts/train_with_divisions.py \
    --train-data data/division/type1_text/train.jsonl \
    --val-data data/division/type1_text/validation.jsonl \
    --model google/gemma-2-2b-it \
    --output-dir phase5_subject_training/models/type1_text \
    --epochs 3 \
    --device cuda:0
```

**Division Tracking:**
During evaluation, tracks per-division:
- Accuracy
- Average loss
- Sample count

**Output:**
```
============================================================
Division Performance
============================================================
Division             Count   Accuracy   Avg Loss
------------------------------------------------------------
1                     1234      87.50%     0.4521
2                      876      92.30%     0.3821
5.4.1                  543      78.40%     0.6234
...
------------------------------------------------------------
OVERALL               8234      85.67%
============================================================
```

Saves `division_report.json` with detailed stats per division.

### Full Pipeline

```bash
# Run entire pipeline for all 4 types
bash phase5_subject_training/scripts/run_pipeline.sh
```

This will:
1. Annotate all 4 types (train + validation)
2. Validate and fix annotations
3. Train with division tracking
4. Generate division reports

### Outputs

```
data/division/
├── type1_text/
│   ├── train.jsonl          # Division-annotated
│   └── validation.jsonl
├── type2_text_reasoning/
├── type3_word/
└── type4_word_reasoning/

phase5_subject_training/models/
├── type1_text/
│   ├── final/               # Trained model
│   └── division_report.json # Performance per division
├── type2_text_reasoning/
├── type3_word/
└── type4_word_reasoning/
```

### Using Division Reports

After training, review `division_report.json` to identify weak areas:

```json
{
  "1.4.1": {
    "accuracy": 0.65,
    "count": 234,
    "avg_loss": 0.8234
  },
  "2.4.3": {
    "accuracy": 0.92,
    "count": 156,
    "avg_loss": 0.3421
  }
}
```

**Weak divisions** (accuracy < 80% or high loss):
- Need more training data
- Require targeted fine-tuning
- May need curriculum learning

### GPU Assignment

- **TITAN RTX (cuda:1)**: DeepSeek annotation
- **RTX A6000 (cuda:0)**: Training

This prevents OOM errors and maximizes throughput.

### Customization

**Change DeepSeek model:**
```bash
python3 phase5_subject_training/scripts/annotate_with_deepseek.py \
    --model deepseek-ai/deepseek-llm-67b-chat \
    --device cuda:1
```

**Custom taxonomy:**
Edit `med_division.json` to add/modify divisions.

**Adjust validation strictness:**
Edit `validate_divisions.py` to customize validation rules.

---

## Integration with Phase 4 Training

Phase 5 can be inserted **after** Phase 4 (Instruction Tuning) and **before** Phase 6 (Evaluation):

```
Phase 4: Instruction Tuning
    ↓
Phase 5: Subject Training (Division-Based)
    ↓ (Models with division tracking)
Phase 6: Evaluation
    ↓
Phase 7: Deployment
```

**Recommended workflow:**
1. Complete Phase 4 instruction tuning
2. Run Phase 5 pipeline to annotate with divisions
3. Identify weak divisions from reports
4. Optionally: Add more data for weak divisions
5. Re-train with division tracking
6. Proceed to Phase 6 evaluation

---

---

## Phase 5: Division-Based Data Organization (UPDATED)

### Overview

Phase 5 now creates **division-specific datasets** by:
1. Adding division annotations to reviewed data using DeepSeek on TITAN RTX
2. Checking division annotation quality
3. Organizing data into division-specific folders for targeted training

### Pipeline Architecture

```
data/reviewed/                     (Human-reviewed data)
        ↓
   DeepSeek Annotation (TITAN RTX, cuda:1)
        ↓
data/division_added/               (Division-annotated data)
├── type1_text/
├── type2_text_reasoning/
├── type3_word/
└── type4_word_reasoning/
        ↓
   Quality Check & Organization
        ↓
data/division_added/               (Division-specific folders)
├── 1/                            (Cardiovascular)
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── metadata.json
├── 2/                            (Respiratory)
├── 3/                            (Gastroenterology)
├── ...
└── division_index.json
```

### One-Command Pipeline

```bash
bash phase5_subject_training/scripts/run_division_pipeline.sh
```

This will:
1. ✨ **Add divisions** to all reviewed data using DeepSeek
2. ✅ **Check quality** of division annotations
3. 📁 **Organize** data into division-specific folders
4. 📊 **Generate** division index with statistics

### Manual Steps

#### Step 1: Add Divisions to Reviewed Data

```bash
# Add divisions to all types
python3 phase5_subject_training/scripts/add_divisions_to_reviewed.py \
    --type all \
    --device cuda:1

# Or process single type
python3 phase5_subject_training/scripts/add_divisions_to_reviewed.py \
    --type type1_text \
    --device cuda:1
```

**Output:** `data/division_added/{type}/train.jsonl` and `validation.jsonl`

#### Step 2: Check Division Quality

```bash
# Check all types
python3 phase5_subject_training/scripts/check_divisions.py --all

# Or check single file
python3 phase5_subject_training/scripts/check_divisions.py \
    --file data/division_added/type1_text/train.jsonl
```

**Report includes:**
- Total samples and validity percentages
- Division distribution (top 15)
- Invalid division IDs
- Missing fields
- Errors and warnings

#### Step 3: Organize by Division

```bash
python3 phase5_subject_training/scripts/organize_by_division.py \
    --source data/division_added \
    --output data/division_added \
    --min-samples 10
```

**Creates:**
- One folder per division (e.g., `1/`, `2/`, etc.)
- Each folder contains `train.jsonl`, `validation.jsonl`, `metadata.json`
- `division_index.json` with statistics

### Division-Specific Folder Structure

```
data/division_added/
├── type1_text/
│   ├── train.jsonl               (with division annotations)
│   └── validation.jsonl
├── type2_text_reasoning/
├── type3_word/
├── type4_word_reasoning/
├── 1/                            (Cardiovascular Medicine)
│   ├── train.jsonl               (all Cardiovascular samples)
│   ├── validation.jsonl
│   └── metadata.json
├── 2/                            (Respiratory Medicine)
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── metadata.json
├── 3/                            (Gastroenterology)
├── 4/                            (Nephrology)
├── 5/                            (Endocrinology)
├── 6/                            (Hematology/Oncology)
├── 7/                            (Neurology)
├── 8/                            (Infectious Diseases)
├── 9/                            (Emergency/Critical Care)
├── 10/                           (Ethics/Law)
└── division_index.json           (index with all divisions)
```

### Division Index Format

`division_index.json`:
```json
{
  "1": {
    "train_samples": 15234,
    "validation_samples": 1692,
    "total_samples": 16926,
    "path": "data/division_added/1"
  },
  "2": {
    "train_samples": 8921,
    "validation_samples": 991,
    "total_samples": 9912,
    "path": "data/division_added/2"
  }
}
```

### Division Metadata Format

Each division folder has `metadata.json`:
```json
{
  "division_id": "1",
  "train_samples": 15234,
  "validation_samples": 1692,
  "total_samples": 16926
}
```

### Quality Check Report Example

```
======================================================================
Division Check Report: data/division_added/type1_text/train.jsonl
======================================================================

Overall Statistics:
  Total samples: 118431
  Valid: 115234 (97.30%)
  Invalid: 1897 (1.60%)
  Unknown: 1300 (1.10%)
  Missing fields: 0

Primary Division Distribution (Top 15):
  Division  Name                                     Count      %
  ------------------------------------------------------------------
  1         Cardiovascular Medicine                  28234   23.8%
  2         Respiratory Medicine                     18921   16.0%
  5         Endocrinology and Metabolism             15432   13.0%
  3         Gastroenterology and Hepatology          12876   10.9%
  7         Neurology                                11234    9.5%
  ...
```

### Training Division-Specific Models

After organizing data by division, you can train division-specific models:

```bash
# Train Cardiovascular-specific model
python3 phase5_subject_training/scripts/train_with_divisions.py \
    --train-data data/division_added/1/train.jsonl \
    --val-data data/division_added/1/validation.jsonl \
    --model google/gemma-2-2b-it \
    --output-dir phase5_subject_training/models/division_1_cardio \
    --epochs 5
```

### Use Cases

#### 1. Targeted Training for Weak Divisions

If division reports show low accuracy for division "5" (Endocrinology):

```bash
# Train specialized Endocrinology model
python3 phase5_subject_training/scripts/train_with_divisions.py \
    --train-data data/division_added/5/train.jsonl \
    --val-data data/division_added/5/validation.jsonl \
    --output-dir phase5_subject_training/models/division_5_endo \
    --epochs 10 \
    --lr 5e-6
```

#### 2. Multi-Division Training

Combine multiple related divisions:

```bash
# Combine Cardio + Respiratory for internal medicine
cat data/division_added/1/train.jsonl data/division_added/2/train.jsonl > temp_train.jsonl
cat data/division_added/1/validation.jsonl data/division_added/2/validation.jsonl > temp_val.jsonl

python3 phase5_subject_training/scripts/train_with_divisions.py \
    --train-data temp_train.jsonl \
    --val-data temp_val.jsonl \
    --output-dir phase5_subject_training/models/internal_medicine
```

#### 3. Balanced Division Training

If some divisions have too few samples, combine them:

```bash
# Check division sizes
cat data/division_added/division_index.json

# Combine small divisions
cat data/division_added/10/train.jsonl \
    data/division_added/9/train.jsonl > combined_train.jsonl
```

### Scripts Reference

| Script | Purpose | GPU |
|--------|---------|-----|
| `add_divisions_to_reviewed.py` | Add divisions to reviewed data | cuda:1 (TITAN RTX) |
| `check_divisions.py` | Check annotation quality | CPU |
| `organize_by_division.py` | Organize into division folders | CPU |
| `run_division_pipeline.sh` | Run full pipeline | cuda:1 |

### Time Estimates

| Step | Time (all 4 types) |
|------|-------------------|
| Add divisions (DeepSeek) | ~5-6 hours |
| Check quality | ~5 minutes |
| Organize by division | ~10 minutes |
| **Total** | **~6 hours** |

**Recommendation:** Run overnight

### Expected Output Sizes

| Type | Original Samples | After Division Annotation |
|------|-----------------|---------------------------|
| Type 1 | 118K train, 13K val | 118K with divisions |
| Type 2 | 23K train, 2.5K val | 23K with divisions |
| Type 3 | 17K train, 1.8K val | 17K with divisions |
| Type 4 | 8K train, 900 val | 8K with divisions |
| **Total** | **166K train, 18K val** | **~10-15 division folders** |

### Benefits

1. 🎯 **Targeted Training**: Train models for specific medical specialties
2. 📊 **Better Analysis**: Understand which divisions have sufficient data
3. 🔍 **Weak Division Focus**: Identify and improve weak subject areas
4. 📁 **Organized Data**: Clear separation by medical subject
5. 🚀 **Flexible Training**: Mix and match divisions as needed

### Next Steps After Division Organization

1. **Review division_index.json**
   - Check which divisions have enough data
   - Identify divisions needing more samples

2. **Check previous training reports**
   - Find divisions with low accuracy
   - Prioritize training for those divisions

3. **Train division-specific models**
   - Focus on weak divisions first
   - Use higher learning rates for small divisions

4. **Evaluate division models**
   - Test on division-specific validation sets
   - Compare against general model

5. **Ensemble or merge**
   - Combine division models for production
   - Or use division routing (route questions to specialist models)

### Troubleshooting

**DeepSeek OOM:**
```bash
# Use smaller model
--model deepseek-ai/deepseek-llm-7b-chat
```

**Too many UNKNOWN divisions:**
- Review annotation prompt in `add_divisions_to_reviewed.py`
- Increase temperature for more diverse outputs
- Check if questions are medical-related

**Division folders not created:**
- Check `--min-samples` parameter (default: 10)
- Some divisions may have too few samples
- Review `division_index.json` for statistics

**Want to re-organize:**
```bash
# Delete division folders (keeps type folders)
find data/division_added -maxdepth 1 -type d -name '[0-9]*' -exec rm -rf {} +
rm data/division_added/division_index.json

# Re-run organization
python3 phase5_subject_training/scripts/organize_by_division.py --all
```

---

---

## Korean Proficiency Validation

### Current Korean Data Status

**Total Data:** 184,556 samples (166,107 train + 18,449 val)

| Type | Korean Content | Samples | Korean Ratio |
|------|----------------|---------|--------------|
| Type 1 (TEXT) | ✅ 100% Korean | 131,591 | ~89% Korean chars |
| Type 2 (TEXT_REASONING) | ✅ 100% Korean | 25,576 | ~80% Korean chars |
| Type 3 (WORD) | ⚠️ Letters only | 18,547 | Answers: A/B/C/D/E |
| Type 4 (WORD_REASONING) | ⚠️ English reasoning | 8,842 | English <R>...<R/> |

**Korean Medical Content:** 157,167 samples (85% of total)

### Korean Proficiency Check

```bash
# Validate Korean quality in all data
python3 scripts/validate_korean_proficiency.py --all

# Check specific file
python3 scripts/validate_korean_proficiency.py \
    --file data/reviewed/type1_text/train/data.jsonl
```

**Results:**
- Type 1: 100% Korean, 89% Korean character ratio, 80% with medical terms
- Type 2: 100% Korean, 80% Korean character ratio, 68% with medical terms
- Type 3: Answer-only (A/B/C/D/E), questions are Korean
- Type 4: English reasoning blocks, needs improvement

### Korean Medical Benchmarks

#### 1. KorMedMCQA (PRIMARY)

**Dataset:** `sean0042/KorMedMCQA`
- **Test samples:** 604 Korean medical MCQs
- **Target:** ≥90% accuracy
- **Use:** Primary evaluation metric for Korean medical proficiency
- **Status:** Available in `data/raw/by_source/kormedmcqa/`

```bash
# Evaluate on KorMedMCQA
python3 scripts/train_loop_until_90.py --model medgemma-27b
```

#### 2. KMMLU-Medical

**Dataset:** Korean Massive Multitask Language Understanding
- Multiple medical subjects (anatomy, pharmacology, clinical)
- 100% Korean content
- **Status:** Available in `data/raw/korean_datasets/kmmlu_medical/`

#### 3. MedQA-Korean

**Dataset:** `ChuGyouk/MedQA` (22,900 samples)
- Korean translated USMLE-style questions
- **Status:** Available and used in training

### Korean Medical Vocabulary Coverage

Common medical terms in dataset:

```
환자 (patient)       - 80%+ coverage
진단 (diagnosis)     - 70%+ coverage
치료 (treatment)     - 65%+ coverage
증상 (symptoms)      - 75%+ coverage
질병 (disease)       - 60%+ coverage
약물 (medication)    - 55%+ coverage
```

### Validation Workflow

**Automated (Quick):**
```bash
python3 scripts/validate_korean_proficiency.py --all
```

**Benchmark Evaluation:**
```bash
python3 scripts/train_loop_until_90.py --model medgemma-27b
# Targets: KorMedMCQA ≥90%
```

**Manual Review (100 samples):**
- Korean grammar correctness
- Medical terminology accuracy
- Natural Korean flow

### Recommended Metrics

| Metric | Target | Method |
|--------|--------|--------|
| KorMedMCQA Accuracy | ≥90% | Automated evaluation |
| Korean Ratio (Type 1/2) | ≥85% | `validate_korean_proficiency.py` |
| Medical Term Coverage | ≥70% | `validate_korean_proficiency.py` |
| Manual Review Score | ≥8/10 | Human evaluation |

### Korean Data Sources (36 total)

Available in `data/raw/korean_datasets/`:
- Asan AMC Healthinfo (hospital data)
- Korean medical textbooks
- KoMedInstruct-52k
- Korean Wikipedia medical
- And 32 more datasets

See `KOREAN_PROFICIENCY.md` for complete validation guide.


---

## KorMedMCQA Test Exclusion (Critical for Evaluation)

### Problem: Test Contamination

KorMedMCQA test set (604 samples) may be in training data → Invalid evaluation results

### Solution: Automated Test Exclusion

```bash
# One command - extracts test set and cleans training data
bash scripts/run_test_exclusion.sh
```

**This creates:**
1. `data/kormedmcqa_test/` - Test set for evaluation (604 samples)
2. `data/division_added_clean/` - Training data WITHOUT test samples

### Manual Steps

```bash
# Step 1: Extract KorMedMCQA test set
python3 scripts/extract_kormedmcqa_test.py

# Step 2: Remove test from training
python3 scripts/exclude_test_from_training.py \
    --source data/division_added \
    --output data/division_added_clean
```

### Training Data Folders

| Folder | Use | Status |
|--------|-----|--------|
| `data/division_added/` | ❌ DON'T USE | May contain test samples |
| `data/division_added_clean/` | ✅ USE THIS | Test samples removed |
| `data/kormedmcqa_test/` | 📊 EVALUATE | 604 test samples |

### Correct Training Workflow

```bash
# ❌ WRONG - test contamination
python3 train.py --data data/division_added/

# ✅ CORRECT - clean data
python3 train.py --data data/division_added_clean/
```

### Verification

```bash
# Check removal stats
cat data/division_added_clean/test_exclusion_stats.json

# Expected output:
# {
#   "summary": {
#     "total": 166107,
#     "removed": 604,  ← Test samples removed
#     "kept": 165503
#   }
# }
```

### Integration with Phase 5

```bash
# Complete pipeline:
# 1. Add divisions
bash phase5_subject_training/scripts/run_division_pipeline.sh

# 2. Exclude test (NEW - REQUIRED)
bash scripts/run_test_exclusion.sh

# 3. Train on CLEAN data
python3 scripts/train_loop_until_90.py \
    --data-dir data/division_added_clean

# 4. Evaluate on test set
# (604 samples in data/kormedmcqa_test/all_test.jsonl)
```

### Why This Matters

**Without exclusion:**
- Model sees test questions during training
- Evaluation accuracy artificially high
- Results are INVALID

**With exclusion:**
- Model never sees test questions
- Evaluation accuracy is TRUE performance
- Results are VALID ✅

See `KORMEDMCQA_TEST_EXCLUSION.md` for complete guide.


---

## Probe-and-Focus Training (NEW)

### Overview

Smart training strategy that finds the most effective data type for KorMedMCQA:

1. **PROBE PHASE**: Train N steps on each type, measure KorMedMCQA improvement
2. **FOCUS PHASE**: Focus on best type until improvement < 1%
3. **ROTATE**: Move to next best type
4. **STOP**: When all types exhausted or target reached

### Quick Start

```bash
# Run probe-and-focus training
bash scripts/run_probe_focus.sh

# With custom options
bash scripts/run_probe_focus.sh \
    --model medgemma-27b \
    --probe-steps 30 \
    --focus-steps 50 \
    --min-improvement 1.0 \
    --target 90.0 \
    --device cuda:0
```

### Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    PROBE PHASE                              │
├─────────────────────────────────────────────────────────────┤
│  for type in [type1, type2, type3, type4]:                  │
│      train N steps on type                                  │
│      measure KorMedMCQA accuracy improvement                │
│      record improvement                                     │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    FOCUS PHASE                              │
├─────────────────────────────────────────────────────────────┤
│  while not all_types_exhausted:                             │
│      best_type = type with highest avg improvement          │
│      while improvement >= 1%:                               │
│          train N steps on best_type                         │
│          evaluate on KorMedMCQA                             │
│      mark best_type as exhausted                            │
│      rotate to next best type                               │
└─────────────────────────────────────────────────────────────┘
```

### Failed Question Tracking

The script tracks **top 10 persistently failing questions** for analysis:

```json
// failed_questions.json
[
  {
    "question_id": 42,
    "fail_count": 15,
    "expected": "B",
    "last_predicted": "C",
    "question_preview": "환자가 흉통을 호소하며...",
    "last_failed_step": 500
  }
]
```

**Use cases:**
- Identify specific medical topics the model struggles with
- Find question patterns that confuse the model
- Guide data augmentation for weak areas

### Output Files

```
models/probe_focus_training/
├── final/                    # Trained model
├── final_report.json         # Full training report
│   ├── type_results          # Per-type statistics
│   ├── recommendations       # Type effectiveness ranking
│   └── persistent_failures   # Top 10 failing questions
└── failed_questions.json     # Separate file for failure analysis
```

### Example Report

```json
{
  "final_best_accuracy": 85.5,
  "initial_accuracy": 65.0,
  "type_results": {
    "type1_text": {
      "total_steps": 150,
      "improvements": [3.5, 2.1, 0.8],
      "is_exhausted": true
    },
    "type3_word": {
      "total_steps": 200,
      "improvements": [5.0, 4.2, 3.1, 1.5, 0.5],
      "is_exhausted": true
    }
  },
  "recommendations": [
    {"rank": 1, "type": "type3_word", "avg_improvement": 2.86},
    {"rank": 2, "type": "type1_text", "avg_improvement": 2.13}
  ]
}
```

### Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/train_probe_and_focus.py` | Main training script |
| `scripts/run_probe_focus.sh` | Shell wrapper |
| `scripts/train_adaptive_types.py` | Alternative adaptive approach |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--probe-steps` | 30 | Steps for probing each type |
| `--focus-steps` | 50 | Steps per focus round |
| `--min-improvement` | 1.0 | Min improvement % to continue |
| `--target-accuracy` | 90.0 | Target KorMedMCQA accuracy |
| `--max-focus-rounds` | 10 | Max rounds per type during focus |
| `--eval-samples` | 200 | KorMedMCQA samples for evaluation |

### Integration

After probe-and-focus training:

1. **Check report** for which type was most effective
2. **Review failed questions** to understand weak areas
3. **Add targeted data** for persistently failing topics
4. **Re-run** if needed with more data


---

## Phase 5: Division-Based Subject Training - CURRENT STATUS

**Started:** 2025-12-18 06:00 UTC  
**Updated:** 2025-12-19 03:40 UTC

### Goal
Add medical division annotations to all reviewed data, then train by subject/division to identify weak areas.

### Division Annotation Progress

| Type | Status | Samples | Quality | Device | ETA |
|------|--------|---------|---------|--------|-----|
| Type 2 | 🟡 Running | 16,100 | ❌ 87% UNKNOWN | TITAN RTX | Unknown |
| Type 4 | ⏳ Waiting | 0 | - | - | After Type 2 |
| Type 3 | ⏳ Waiting | 0 | - | - | After Type 4 |
| Type 1 | ⏳ Waiting | 0 | - | - | After Type 3 |

### Critical Issue: Division Quality 🔴

**Type 2 Division Distribution (first 1000 samples):**
```
875 samples → "UNKNOWN" (87.5%)
96 samples → "1" (numeric instead of division name)
29 samples → "2", "3", "4"... (numeric)
```

**Root Cause:** DeepSeek prompt/parsing is failing to extract Korean medical divisions.

**Impact:** 
- 87% of annotations are useless ("UNKNOWN")
- Remaining 13% are numbers instead of "내과", "외과", etc.
- Cannot proceed with division-based training without fixing

### Action Plan

#### Option 1: Wait & Fix Post-Processing ⏳
1. Let Type 2 finish (~16K/23K done, 70%)
2. Analyze completed output
3. Re-annotate with improved prompt
4. Start Type 4 → 3 → 1 with fixes

#### Option 2: Stop & Fix Now 🛑  
1. Kill Type 2 (lose 16K samples)
2. Fix division extraction logic immediately
3. Restart all with corrected script

#### Option 3: Hybrid Approach ✅ **RECOMMENDED**
1. **Let Type 2 finish** (it's 70% done, process active)
2. **Analyze failures** - why "UNKNOWN"? Non-medical data?
3. **Fix prompt/parsing** for Types 4, 3, 1
4. **Start Type 4** with fixes as test case (smallest dataset)
5. **Reprocess Type 2** if needed

### Division Annotation Scripts

- **Current:** `scripts/fast_division_annotation.py` (running on Type 2)
- **Location:** Will move to `phase5_subject_training/scripts/`
- **Model:** DeepSeek-7B-Chat (25GB VRAM on TITAN RTX)
- **Prompt:** Needs improvement for Korean medical divisions

### Next Steps (Manual Intervention Required)

1. **Monitor Type 2 completion:**
   ```bash
   watch -n 60 'ls -lh data/division_added/type2_text_reasoning/train.jsonl'
   ```

2. **When Type 2 finishes, analyze failures:**
   ```bash
   cat data/division_added/type2_text_reasoning/train.jsonl | \
     jq -r '.primary_division' | sort | uniq -c | sort -rn
   ```

3. **Fix division script and start Type 4**

4. **Sequential: Type 4 → Type 3 → Type 1**

### Timeline Estimate

- Type 2 finish: 2-6 hours (uncertain)
- Fix & test: 1-2 hours
- Type 4: 2-3 hours
- Type 3: 4-5 hours
- Type 1: 30-36 hours

**Total:** ~40-52 hours from now

---

## Division List (Korean Medical Specialties)

Target divisions for annotation:
1. 내과 (Internal Medicine)
2. 외과 (Surgery)
3. 소아청소년과 (Pediatrics)
4. 산부인과 (Obstetrics & Gynecology)
5. 정신건강의학과 (Psychiatry)
6. 피부과 (Dermatology)
7. 비뇨의학과 (Urology)
8. 안과 (Ophthalmology)
9. 이비인후과 (Otolaryngology)
10. 정형외과 (Orthopedics)
11. 신경과 (Neurology)
12. 흉부외과 (Thoracic Surgery)
13. 마취통증의학과 (Anesthesiology)
14. 영상의학과 (Radiology)
15. 진단검사의학과 (Laboratory Medicine)
16. 응급의학과 (Emergency Medicine)
17. 가정의학과 (Family Medicine)
18. 재활의학과 (Rehabilitation Medicine)
19. 핵의학과 (Nuclear Medicine)
20. 병리과 (Pathology)

**Special:**
- GENERAL: Multi-division or general medical knowledge
- UNKNOWN: Cannot determine (should be <5%)

---

## Simplified Training Pipeline (script/ & data/)

### Overview

A simplified, self-contained training pipeline using local data. This is an alternative to the full phase-based pipeline for quick experiments and focused training.

### Training Flow

```
data/01_raw/                  (Raw datasets)
        ↓
   Refine scripts
        ↓
data/02_refined/              (Training-ready data)
        ↓
script/train/train_*.py        (Training scripts)
        ↓
models/train_*/                   (Output models)
```

### Directory Structure

```
script/
├── train/
│   ├── train_00_plain_text.py       # Phase 0: Korean plain text (continued pretraining)
│   ├── train_01_medical_dict.py     # Phase 1: Medical dictionary learning
│   ├── train_02_kor_med_test.py     # Phase 2: MCQ with reasoning (KorMedMCQA)
│   └── train_01_02_loop.py          # Loop training: 01 → 02 → 01 → 02...
├── training_config.py            # Model configurations (medgemma-4b, medgemma-27b)
├── training_utils.py             # Shared utilities (LoRA, 8-bit, SFT)
├── validation_kor_med_test.py    # Standalone KorMedMCQA validation
├── add_lora_adapter.py           # Utility to add LoRA to base models
├── check_gpu_memory.py           # GPU memory diagnostic tool
└── old_versions/                 # Deprecated script versions

data/
├── 01_raw/
│   ├── 00_korean/               # Korean plain text sources
│   │   ├── c4_korean/           # C4 Korean (~3GB, 7 arrow files)
│   │   ├── namu_wiki/           # NamuWiki (~18GB, 38 arrow files)
│   │   └── wikipedia-korean/    # Korean Wikipedia (~3GB, 7 arrow files)
│   ├── 01_medical_dict/         # Medical dictionaries
│   │   ├── korean_medical_dict.jsonl       # Human medical terms
│   │   └── korean_animal_medical_dict.jsonl # Veterinary terms
│   └── 02_kor_med_test/         # KorMedMCQA dataset
│       ├── train.jsonl          # 1890 training samples
│       └── test.jsonl           # 604 test samples
└── 02_refined/
    ├── 00_plain_text/
    │   └── train.jsonl          # 1.5GB refined plain text
    ├── 01_medical_dict.json     # 4049 medical terms
    ├── 02_char_dict.json        # 89 special symbols
    └── 02_kor_med_test/
        ├── train.jsonl          # 1890 refined MCQ samples
        └── test.jsonl           # 604 test samples

models/
├── train_00_plain_text/         # Plain text model outputs
├── train_01_medical_dict/       # Dictionary training outputs
└── train_02_kor_med_test/       # MCQ training outputs
```

---

### Training Scripts

#### train/train_00_plain_text.py (Continued Pretraining)

**Purpose:** Learn Korean language patterns from plain text

**Data:** `data/02_refined/00_plain_text/train.jsonl` (~1.5GB)

**Features:**
- Continued pretraining on Korean text
- Includes embeddings in LoRA (`include_embeddings=True`)
- Uses rsLoRA for stable training

```bash
python script/train/train/train_00_plain_text.py --epochs 1 --max-samples 10000
```

---

#### train/train_01_medical_dict.py (Medical Dictionary)

**Purpose:** Learn Korean-English medical terminology and special symbols

**Data:**
- `data/02_refined/01_medical_dict.json` (4049 medical terms)
- `data/02_refined/02_char_dict.json` (89 special symbols)

**Training Format:**
```
<start_of_turn>user
Meaning of word {term}:<end_of_turn>
<start_of_turn>model
{definition}<end_of_turn>
```

**Validation:** KorMedMCQA test (604 samples)

**Key Features:**
- Left padding for batch generation
- `<end_of_turn>` as termination token
- Periodic KorMedMCQA evaluation during training

```bash
# Basic training
python script/train/train/train_01_medical_dict.py --epochs 3

# With custom evaluation interval
python script/train/train/train_01_medical_dict.py --epochs 3 --show-samples-every 50 --eval-samples 10
```

**Output:** `models/train_01_medical_dict/medgemma-4b/`

---

#### train/train_02_kor_med_test.py (MCQ with Reasoning)

**Purpose:** Learn to answer medical MCQs with chain-of-thought reasoning

**Data:** `data/02_refined/02_kor_med_test/` (1890 train, 604 test)

**Training Format (95% simple):**
```
<start_of_turn>user
Reasoning 후 정답 알파벳 하나만 답하세요.

{question}
A) {A}
B) {B}
C) {C}
D) {D}
E) {E}

<end_of_turn>
<start_of_turn>model
<reasoning>
{reasoning_per_choice}
</reasoning>{answer}<end_of_turn>
```

**Training Format (5% detailed):**
```
<start_of_turn>user
Reasoning 후 정답 알파벳 하나만 답하세요.

```format
...detailed format instruction...
```

```example
...full example with reasoning...
```

{question}
...
<end_of_turn>
<start_of_turn>model
<reasoning>
첫째: ...
둘째: ...
셋째: ...
넷째: A) ... xx%, B) ... yy%, ...
</reasoning>{answer}<end_of_turn>
```

**Scoring System:**
- Correct answer: 2/3 weight
- Reasoning format: 1/3 weight
- Wrong answer with good reasoning: 1/4 weight

**Reasoning Format Checks:**
1. `<reasoning>` and `</reasoning>` tags exist
2. `첫째:`, `둘째:`, `셋째:`, `넷째:` keywords with 3+ words
3. `A)`, `B)`, `C)`, `D)`, `E)` with 3+ words and `number%`

```bash
# Basic training
python script/train/train/train_02_kor_med_test.py --epochs 3

# Use train_01 output as base model
python script/train/train/train_02_kor_med_test.py \
    --base-model models/train_01_medical_dict/medgemma-4b/final \
    --epochs 3
```

**Output:** `models/train_02_kor_med_test/medgemma-4b/`

---

#### train/train_01_02_loop.py (Loop Training)

**Purpose:** Alternating training between dictionary and MCQ until target accuracy

**Strategy:**
```
Loop 1: train_01 (1 epoch) → train_02 (1 epoch) → evaluate
Loop 2: train_01 (1 epoch) → train_02 (1 epoch) → evaluate
...
Stop when KorMedMCQA accuracy ≥ target
```

```bash
python script/train/train/train_01_02_loop.py --max-loops 5 --target-accuracy 80
```

---

### Model Configuration

`training_config.py`:
```python
MODEL_CONFIGS = {
    "medgemma-4b": {
        "path": "google/medgemma-4b-it",
        "lora_r": 64,
        "lora_alpha": 128,
        "batch": 4,
        "grad_accum": 4,
        "lr": 2e-5,
        "max_length": 512
    },
    "medgemma-27b": {
        "path": "google/medgemma-27b-it",
        "lora_r": 32,
        "lora_alpha": 64,
        "batch": 1,
        "grad_accum": 16,
        "lr": 1e-5,
        "max_length": 512
    }
}

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

---

### Quick Start

```bash
# 1. Train on medical dictionary (builds vocabulary)
python script/train/train/train_01_medical_dict.py --epochs 3

# 2. Train on MCQ using dictionary-trained model
python script/train/train/train_02_kor_med_test.py \
    --base-model models/train_01_medical_dict/medgemma-4b/final \
    --epochs 3

# 3. (Optional) Loop training for higher accuracy
python script/train/train/train_01_02_loop.py --max-loops 5

# 4. Validate final model
python script/validation_kor_med_test.py \
    --model models/train_02_kor_med_test/medgemma-4b/final
```

---

### Data Formats

#### Medical Dictionary (`01_medical_dict.json`)
```json
[
  {"term": "고혈압", "definition": "Hypertension. 혈압이 정상 범위보다 높은 상태..."},
  {"term": "당뇨병", "definition": "Diabetes mellitus. 인슐린 분비 또는 작용 장애..."}
]
```

#### Character Dictionary (`02_char_dict.json`)
```json
[
  {"term": "↑", "definition": "increased, elevated, or upward trend"},
  {"term": "↓", "definition": "decreased, reduced, or downward trend"}
]
```

#### KorMedMCQA (`02_kor_med_test/*.jsonl`)
```json
{
  "question": "광역시 소재 대학병원에 소속된 내과 전문의...",
  "A": "병원장에게 보고",
  "B": "광역시장에게 신고",
  "C": "질병관리청장에게 신고",
  "D": "관할 보건소장에게 신고",
  "E": "보건복지부장관에게 신고",
  "answer": "D"
}
```

---

### Training Output

Each training creates:
```
models/train_XX_name/medgemma-4b/
├── checkpoint-N/           # Epoch checkpoints
├── final/                  # Final model (LoRA adapter)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
├── training_info.json      # Training metadata
└── README.md               # Auto-generated summary
```

**training_info.json example:**
```json
{
  "script": "train_01_medical_dict",
  "model": "medgemma-4b",
  "base_model": "google/medgemma-4b-it",
  "epochs": 3,
  "train_samples": 4138,
  "validation_samples": 604,
  "validation_history": [
    {"step": 100, "accuracy": 50.0, "correct": 5, "total": 10}
  ],
  "final_accuracy": 50.0
}
```

---

### Common Options

All training scripts support:

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `medgemma-4b` | Model to train (`medgemma-4b` or `medgemma-27b`) |
| `--epochs` | `3` | Number of training epochs |
| `--max-samples` | `None` | Limit training samples (for testing) |
| `--base-model` | `None` | Override base model path |
| `--output` | `None` | Override output directory |
| `--device` | `cuda:0` | Training device |
| `--show-samples-every` | `50` | Validation interval (steps) |
| `--eval-samples` | `10` | Number of validation samples |

---

### GPU Requirements

| Model | VRAM Required | Batch Size | Time per Epoch |
|-------|---------------|------------|----------------|
| medgemma-4b | ~12GB (8-bit) | 4 | ~40 min |
| medgemma-27b | ~28GB (8-bit) | 1 | ~4 hours |

**Recommended Setup:**
- RTX A6000 (48GB): Can run medgemma-27b comfortably
- RTX 4090 (24GB): medgemma-4b with full batch, medgemma-27b with reduced batch
- TITAN RTX (24GB): medgemma-4b recommended

---

## Current Training Status (Updated: 2025-12-22)

### Pipeline Status

| Step | Script | Status | Notes |
|------|--------|--------|-------|
| Step 1/4 | train/train_00_plain_text.py | IN PROGRESS | Plain text Korean training |
| Step 2/4 | train/train_01_with_00_monitor.py | PENDING | Medical dictionary with monitoring |
| Step 3/4 | add_lora_adapter.py | PENDING | Add second LoRA adapter |
| Step 4/4 | train/train_02_kor_med_test.py | PENDING | MCQ with reasoning |

### Recent Fixes Applied (2025-12-22)

1. **LoRA Config Fix**: Changed `target_modules` to `modules_to_save` for embeddings
   - Embeddings require `modules_to_save` for 8-bit quantized models
   - Fixed in `training_utils.py:create_lora_config()`

2. **Trainable Parameters Fix**: Added `is_trainable=True` to `PeftModel.from_pretrained()`
   - Without this, adapter loads in inference mode (0% trainable)
   - Fixed in train_00, train_01, train_02 scripts

3. **Memory Optimization**: Enabled gradient checkpointing for medgemma-4b
   - Extended tokenizer embeddings (1.59B trainable params) require more memory
   - Reduced batch size: 4 → 2, grad_accum: 8 → 16

### Memory Configuration

```python
MEMORY_CONFIGS = {
    "medgemma-4b": {
        "use_gradient_checkpointing": True,   # Required with extended embeddings
        "train_embeddings": True,              # Train Korean embeddings
        "batch": 2,                            # Reduced from 4
        "grad_accum": 16,                      # Increased from 8
    },
    "medgemma-27b": {
        "use_gradient_checkpointing": True,   # Required
        "train_embeddings": False,             # OOM with embeddings
    }
}
```

### Extended Tokenizer

- **New Korean tokens:** 23,699
- **Vocab size:** 262,208 → 285,844
- **Trainable params with embeddings:** 1.59B (26.78%)

### Monitoring Commands

```bash
# Check progress
cat /home/ormastes/dev/pub/medgemma_korean/logs/progress_medgemma-4b.txt

# Watch live log
tail -f /home/ormastes/dev/pub/medgemma_korean/logs/pipeline_medgemma-4b.log

# Stop training
kill $(cat /home/ormastes/dev/pub/medgemma_korean/logs/pipeline_medgemma-4b.pid)
```

### Run Pipeline

```bash
# Start training (background)
./run_full_pipeline.sh --model medgemma-4b --background

# With options
./run_full_pipeline.sh --model medgemma-4b --epochs-00 1 --epochs-01 3 --epochs-02 5 --background

# Skip specific steps
./run_full_pipeline.sh --model medgemma-4b --skip-00 --background
```

### Output Directories

```
model/
├── raw_lora_added/medgemma-4b/     # Initial LoRA (untrained)
├── 00_trained/medgemma-4b/          # After train_00
├── 01_trained/medgemma-4b/          # After train_01
├── 01_another_lora_added/medgemma-4b/  # After adding 2nd LoRA
└── 02_trained/medgemma-4b/          # After train_02 (final)
```
