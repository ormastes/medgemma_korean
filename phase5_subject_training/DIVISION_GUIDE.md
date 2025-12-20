# Phase 5: Division-Based Organization - Complete Guide

## 🎯 What Changed

Phase 5 has been **updated** to create division-specific datasets:

**BEFORE:** Train models with division tracking (output: division reports)
**NOW:** Organize data by division → Train division-specific models

## 📋 Three-Step Pipeline

### Step 1: Add Divisions (DeepSeek on A6000)
- Uses DeepSeek to analyze each question/answer
- Assigns medical divisions (e.g., "1" = Cardiovascular)
- Saves annotated data to `data/division_added/{type}/`

### Step 2: Check Quality
- Validates division IDs against taxonomy
- Reports statistics and distribution
- Identifies invalid annotations

### Step 3: Organize by Division
- Creates one folder per division
- Each folder has train.jsonl + validation.jsonl
- Generates division_index.json

## 🚀 Quick Start

```bash
# Full pipeline - one command
bash phase5_subject_training/scripts/run_division_pipeline.sh
```

**Time:** ~6 hours for all 4 types (run overnight)

## 📁 Output Structure

```
data/division_added/
├── type1_text/
│   ├── train.jsonl              (118K samples with divisions)
│   └── validation.jsonl          (13K samples)
├── type2_text_reasoning/
├── type3_word/
├── type4_word_reasoning/
│
├── 1/                           (Cardiovascular division)
│   ├── train.jsonl              (all Cardio samples)
│   ├── validation.jsonl
│   └── metadata.json
├── 2/                           (Respiratory division)
├── 3/                           (Gastroenterology)
├── 4/                           (Nephrology)
├── 5/                           (Endocrinology)
├── 6/                           (Hematology/Oncology)
├── 7/                           (Neurology)
├── 8/                           (Infectious Diseases)
├── 9/                           (Emergency/Critical Care)
├── 10/                          (Ethics/Law)
│
└── division_index.json          (statistics for all divisions)
```

## 📊 Division Index Example

```json
{
  "1": {
    "train_samples": 15234,
    "validation_samples": 1692,
    "total_samples": 16926,
    "path": "data/division_added/1"
  },
  "5": {
    "train_samples": 8234,
    "validation_samples": 915,
    "total_samples": 9149,
    "path": "data/division_added/5"
  }
}
```

## 🎓 Use Cases

### 1. Train Division-Specific Model

Train a specialized Cardiovascular model:

```bash
python3 phase5_subject_training/scripts/train_with_divisions.py \
    --train-data data/division_added/1/train.jsonl \
    --val-data data/division_added/1/validation.jsonl \
    --model google/gemma-2-2b-it \
    --output-dir phase5_subject_training/models/cardio_specialist \
    --epochs 5
```

### 2. Focus on Weak Divisions

If evaluation shows Division 5 (Endocrinology) is weak:

```bash
# Check division has enough data
cat data/division_added/division_index.json | grep "\"5\""

# Train with more epochs and lower LR
python3 phase5_subject_training/scripts/train_with_divisions.py \
    --train-data data/division_added/5/train.jsonl \
    --val-data data/division_added/5/validation.jsonl \
    --output-dir phase5_subject_training/models/endo_specialist \
    --epochs 10 \
    --lr 5e-6
```

### 3. Combine Related Divisions

Train internal medicine model (Cardio + Respiratory + GI):

```bash
cat data/division_added/1/train.jsonl \
    data/division_added/2/train.jsonl \
    data/division_added/3/train.jsonl > internal_med_train.jsonl

cat data/division_added/1/validation.jsonl \
    data/division_added/2/validation.jsonl \
    data/division_added/3/validation.jsonl > internal_med_val.jsonl

python3 phase5_subject_training/scripts/train_with_divisions.py \
    --train-data internal_med_train.jsonl \
    --val-data internal_med_val.jsonl \
    --output-dir phase5_subject_training/models/internal_medicine
```

## 📝 Scripts Reference

| Script | Purpose | Command |
|--------|---------|---------|
| **add_divisions_to_reviewed.py** | Annotate with DeepSeek | `--type all --device cuda:1` |
| **check_divisions.py** | Quality check | `--all` or `--file <path>` |
| **organize_by_division.py** | Create division folders | `--min-samples 10` |
| **run_division_pipeline.sh** | Full pipeline | Run directly |

## ✅ Quality Check Report

After running check_divisions.py:

```
======================================================================
Overall Statistics:
  Total samples: 118431
  Valid: 115234 (97.30%)
  Invalid: 1897 (1.60%)
  Unknown: 1300 (1.10%)

Primary Division Distribution (Top 15):
  Division  Name                                     Count      %
  ------------------------------------------------------------------
  1         Cardiovascular Medicine                  28234   23.8%
  2         Respiratory Medicine                     18921   16.0%
  5         Endocrinology and Metabolism             15432   13.0%
```

## 🔍 Verification Commands

```bash
# Check division index
cat data/division_added/division_index.json | python3 -m json.tool

# Check specific division metadata
cat data/division_added/1/metadata.json

# Count samples in division
wc -l data/division_added/1/train.jsonl
wc -l data/division_added/1/validation.jsonl

# Check all division folders
ls -la data/division_added/ | grep "^d"

# Verify division annotations in file
head -1 data/division_added/type1_text/train.jsonl | python3 -m json.tool | grep division
```

## 🎯 Workflow Integration

```
Phase 4: Instruction Tuning
    ↓
Phase 5: Division Organization ← YOU ARE HERE
    ↓
    ├─→ General training (all divisions mixed)
    └─→ Division-specific training (per division)
         ↓
Phase 6: Evaluation
    ├─→ Overall evaluation
    └─→ Per-division evaluation
         ↓
Phase 7: Deployment
    ├─→ Single general model
    └─→ Ensemble of division models
```

## 💡 Training Strategies

### Strategy 1: General Model First

```bash
# 1. Train general model with all divisions
python3 scripts/train_loop_until_90.py --model medgemma-27b

# 2. Evaluate per division
python3 phase6_evaluation/evaluate_by_division.py

# 3. Identify weak divisions
# (e.g., Division 5 = 65% accuracy)

# 4. Train division-specific model for weak areas
python3 phase5_subject_training/scripts/train_with_divisions.py \
    --train-data data/division_added/5/train.jsonl \
    --val-data data/division_added/5/validation.jsonl \
    --output-dir models/division_5_boost
```

### Strategy 2: Division-First Approach

```bash
# Train specialist for each major division
for div in 1 2 3 4 5 6 7; do
    python3 phase5_subject_training/scripts/train_with_divisions.py \
        --train-data data/division_added/$div/train.jsonl \
        --val-data data/division_added/$div/validation.jsonl \
        --output-dir models/division_${div}_specialist \
        --epochs 5
done

# Ensemble at inference time
# Route questions to appropriate specialist based on division classification
```

### Strategy 3: Curriculum Learning

```bash
# 1. Start with large divisions (more data)
python3 train.py --train-data data/division_added/1/train.jsonl  # Cardio (largest)
python3 train.py --train-data data/division_added/2/train.jsonl  # Respiratory

# 2. Fine-tune on smaller divisions
python3 train.py --train-data data/division_added/10/train.jsonl  # Ethics (smallest)

# 3. Final tuning on all divisions mixed
python3 train.py --train-data data/division_added/*/train.jsonl
```

## 🐛 Troubleshooting

### Issue: DeepSeek OOM on A6000

**Solution:**
```bash
# Use smaller model
python3 add_divisions_to_reviewed.py \
    --type all \
    --model deepseek-ai/deepseek-llm-7b-chat \
    --device cuda:1
```

### Issue: Too many UNKNOWN divisions

**Check:**
```bash
# See how many UNKNOWN
python3 check_divisions.py --file data/division_added/type1_text/train.jsonl | grep Unknown
```

**Fix:**
- Review DeepSeek prompt in `add_divisions_to_reviewed.py`
- Increase temperature (0.3 → 0.5)
- Use larger DeepSeek model

### Issue: Some divisions have too few samples

**Check:**
```bash
cat data/division_added/division_index.json | grep -A3 '"10"'
```

**Solution:**
```bash
# Lower min-samples threshold
python3 organize_by_division.py --min-samples 5

# Or combine small divisions
cat data/division_added/10/train.jsonl >> data/division_added/9/train.jsonl
```

### Issue: Want to re-annotate

```bash
# Delete old annotations
rm -rf data/division_added/type1_text/

# Re-run annotation
python3 add_divisions_to_reviewed.py --type type1_text
```

## 📈 Expected Results

| Division | Typical Sample Count | Training Viability |
|----------|---------------------|-------------------|
| 1 (Cardio) | 15K-20K | ✅ Excellent |
| 2 (Respiratory) | 10K-15K | ✅ Excellent |
| 3 (GI) | 8K-12K | ✅ Good |
| 5 (Endocrine) | 8K-10K | ✅ Good |
| 7 (Neuro) | 6K-8K | ⚠️ Moderate |
| 10 (Ethics) | 1K-2K | ⚠️ May need augmentation |

## 🎉 Summary

**Before Phase 5:**
- All data mixed together
- Hard to identify weak subject areas
- One-size-fits-all training

**After Phase 5:**
- Data organized by medical division
- Clear visibility into division distribution
- Can train division-specific specialists
- Targeted improvement for weak areas
- Flexible training strategies

---

**Ready to start?**

```bash
bash phase5_subject_training/scripts/run_division_pipeline.sh
```

Then check `data/division_added/division_index.json` to see your division distribution!
