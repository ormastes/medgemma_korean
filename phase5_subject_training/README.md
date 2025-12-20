# Phase 5: Subject Training - Division-Based Medical Training

## Overview

Phase 5 adds **medical division annotations** to all training data using DeepSeek on A6000, validates annotations, and trains models with per-division performance tracking.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│            Phase 5: Subject Training                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Step 1: DeepSeek Annotation (A6000, cuda:0)  │
│           data/reviewed → data/division             │
│           Adds: divisions, primary_division         │
│                                                     │
│  Step 2: Validation & Fix                          │
│           Validates division IDs against taxonomy   │
│           Auto-fixes malformed annotations          │
│                                                     │
│  Step 3: Division-Aware Training (A6000, cuda:0)   │
│           Tracks accuracy/loss per division         │
│           Generates division_report.json            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Medical Divisions

10 major divisions with hierarchical sub-divisions:

1. Cardiovascular Medicine (e.g., `1.4.1` = Ischemic Heart Disease)
2. Respiratory Medicine
3. Gastroenterology and Hepatology
4. Nephrology
5. Endocrinology and Metabolism
6. Hematology and Oncology
7. Neurology
8. Infectious Diseases (Cross-Cutting)
9. Emergency and Critical Care (Cross-Cutting)
10. Ethics, Law, and Patient Safety (Cross-Cutting)

Full taxonomy in `med_division.json`.

## Data Flow

```
data/reviewed/type1_text/train/data.jsonl
          ↓ (DeepSeek annotation)
data/division/type1_text/train.jsonl
          ↓ (Validation & fix)
data/division/type1_text/train.jsonl (fixed)
          ↓ (Training)
phase5_subject_training/models/type1_text/
    ├── final/                  (trained model)
    └── division_report.json    (performance per division)
```

## Scripts

| Script | Purpose | GPU |
|--------|---------|-----|
| `annotate_with_deepseek.py` | Annotate with DeepSeek | cuda:0 (A6000) |
| `validate_divisions.py` | Validate & fix annotations | CPU |
| `train_with_divisions.py` | Train with division tracking | cuda:0 (A6000) |
| `run_pipeline.sh` | Full pipeline | Both |

## Usage

### Quick Start (Full Pipeline)

```bash
bash phase5_subject_training/scripts/run_pipeline.sh
```

### Manual Steps

#### 1. Annotate with DeepSeek

```bash
python3 phase5_subject_training/scripts/annotate_with_deepseek.py \
    --input data/reviewed/type1_text/train/data.jsonl \
    --output data/division/type1_text/train.jsonl \
    --device cuda:0
```

#### 2. Validate & Fix

```bash
python3 phase5_subject_training/scripts/validate_divisions.py \
    --input data/division/type1_text/train.jsonl \
    --output data/division/type1_text/train_fixed.jsonl \
    --fix
```

#### 3. Train with Division Tracking

```bash
python3 phase5_subject_training/scripts/train_with_divisions.py \
    --train-data data/division/type1_text/train.jsonl \
    --val-data data/division/type1_text/validation.jsonl \
    --model google/gemma-2-2b-it \
    --output-dir phase5_subject_training/models/type1_text \
    --epochs 3 \
    --device cuda:0
```

## Division Report Example

After training, `division_report.json` shows performance per division:

```json
{
  "1": {
    "accuracy": 0.8750,
    "count": 1234,
    "avg_loss": 0.4521
  },
  "1.4.1": {
    "accuracy": 0.6534,
    "count": 234,
    "avg_loss": 0.8234
  },
  "2": {
    "accuracy": 0.9230,
    "count": 876,
    "avg_loss": 0.3821
  }
}
```

**Interpretation:**
- Division `1.4.1` (Ischemic Heart Disease): **Weak** (65% accuracy, high loss)
- Division `2` (Respiratory): **Strong** (92% accuracy, low loss)

**Action:** Add more training data for weak divisions.

## Validation Rules

Checks performed by `validate_divisions.py`:

1. ✅ `divisions` field exists and is a list
2. ✅ `primary_division` field exists
3. ✅ Division IDs are valid (in taxonomy or "UNKNOWN")
4. ✅ `primary_division` is in `divisions` list
5. ✅ Divisions list is non-empty

Auto-fixes:
- Adds missing fields
- Removes invalid IDs
- Sets primary to first valid division
- Adds "UNKNOWN" if empty

## Benefits

1. **Identify weak subject areas**: Know which medical divisions need improvement
2. **Targeted training**: Focus on specific divisions
3. **Quality assurance**: Validate division annotations automatically
4. **Performance monitoring**: Track progress per division during training
5. **Data insights**: Understand division distribution in dataset

## Integration with Main Training

Phase 5 integrates between Phase 4 and Phase 6:

```
Phase 4: Instruction Tuning
    ↓
Phase 5: Subject Training ← INSERT HERE
    ↓
Phase 6: Evaluation
    ↓
Phase 7: Deployment
```

## Output Structure

```
phase5_subject_training/
├── scripts/
│   ├── annotate_with_deepseek.py
│   ├── validate_divisions.py
│   ├── train_with_divisions.py
│   └── run_pipeline.sh
├── models/
│   ├── type1_text/
│   │   ├── final/
│   │   ├── division_report.json
│   │   └── checkpoint-*/
│   ├── type2_text_reasoning/
│   ├── type3_word/
│   └── type4_word_reasoning/
├── logs/
└── results/
```

## Troubleshooting

**DeepSeek OOM on A6000:**
- Use smaller model: `deepseek-ai/deepseek-llm-7b-chat`
- Reduce batch size in annotation script

**Validation fails:**
- Check `med_division.json` is present
- Review malformed samples manually
- Use `--fix` flag to auto-correct

**Training slow:**
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision (already enabled with bf16)

## Next Steps

After Phase 5:
1. Review division reports
2. Identify divisions with accuracy < 80%
3. Add targeted training data for weak divisions
4. Optionally: Re-run Phase 5 with augmented data
5. Proceed to Phase 6 (Evaluation)
