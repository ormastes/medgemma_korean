# KorMedMCQA Test Exclusion Guide

## Overview

This guide ensures **no test contamination** in training data by:
1. Extracting KorMedMCQA test set (604 samples)
2. Removing test samples from training data
3. Creating clean training folders for division-based training

## Problem

**KorMedMCQA test set must NOT be in training data** to get valid evaluation results.

## Solution

### One-Command Pipeline

```bash
bash scripts/run_test_exclusion.sh
```

This will:
1. Extract KorMedMCQA test set → `data/kormedmcqa_test/`
2. Remove test samples from training → `data/division_added_clean/`
3. Generate exclusion statistics

## Manual Steps

### Step 1: Extract KorMedMCQA Test Set

```bash
python3 scripts/extract_kormedmcqa_test.py --output data/kormedmcqa_test
```

**Creates:**
```
data/kormedmcqa_test/
├── test_doctor.jsonl      (Doctor exam - ~300 samples)
├── test_nurse.jsonl       (Nurse exam - ~150 samples)
├── test_pharm.jsonl       (Pharmacist - ~100 samples)
├── test_dentist.jsonl     (Dentist - ~50 samples)
├── all_test.jsonl         (Combined - 604 samples)
└── test_questions.txt     (Question texts for exclusion)
```

### Step 2: Remove Test Samples from Training

```bash
python3 scripts/exclude_test_from_training.py \
    --test-file data/kormedmcqa_test/test_questions.txt \
    --source data/division_added \
    --output data/division_added_clean
```

**Process:**
- Reads all samples from `data/division_added/`
- Extracts questions from prompts
- Compares against `test_questions.txt`
- Removes matches
- Saves clean data to `data/division_added_clean/`

**Creates:**
```
data/division_added_clean/
├── type1_text/
│   ├── train.jsonl              (without test samples)
│   └── validation.jsonl
├── type2_text_reasoning/
├── type3_word/
├── type4_word_reasoning/
├── 1/                           (Division folders - cleaned)
├── 2/
├── ...
└── test_exclusion_stats.json    (Removal statistics)
```

## Exclusion Statistics

After running, check stats:

```bash
cat data/division_added_clean/test_exclusion_stats.json
```

**Example output:**
```json
{
  "summary": {
    "total": 166107,
    "removed": 604,
    "kept": 165503,
    "removal_rate": 0.36
  },
  "details": {
    "type1_text/train": {
      "total": 118431,
      "removed": 250,
      "kept": 118181
    },
    "type3_word/train": {
      "total": 16701,
      "removed": 300,
      "kept": 16401
    }
  }
}
```

## Directory Structure

### Before (Contaminated)

```
data/division_added/          ⚠️ Contains test samples
├── type1_text/
├── type2_text_reasoning/
├── type3_word/               ⚠️ KorMedMCQA test samples here
└── type4_word_reasoning/
```

### After (Clean)

```
data/kormedmcqa_test/         ✓ Test set (for evaluation)
├── all_test.jsonl            → 604 test samples

data/division_added_clean/    ✓ Training set (no test)
├── type1_text/               → Clean training data
├── type2_text_reasoning/
├── type3_word/               → Test samples removed
├── type4_word_reasoning/
└── test_exclusion_stats.json → Removal report
```

## Training Workflow

### Before Test Exclusion (WRONG ❌)

```bash
# DON'T DO THIS - test contamination!
python3 scripts/train_with_divisions.py \
    --train-data data/division_added/type3_word/train.jsonl \
    --val-data data/division_added/type3_word/validation.jsonl
```

### After Test Exclusion (CORRECT ✅)

```bash
# Use clean data
python3 scripts/train_with_divisions.py \
    --train-data data/division_added_clean/type3_word/train.jsonl \
    --val-data data/division_added_clean/type3_word/validation.jsonl
```

## Evaluation Workflow

```bash
# 1. Train on CLEAN data
python3 scripts/train_loop_until_90.py \
    --model medgemma-27b \
    --data-dir data/division_added_clean

# 2. Evaluate on TEST set
python3 scripts/evaluate_kormedmcqa.py \
    --model models/final \
    --test-file data/kormedmcqa_test/all_test.jsonl
```

## Korean Test Set Structure

KorMedMCQA test set has 4 specialties:

| Specialty | Samples | File |
|-----------|---------|------|
| Doctor | ~300 | `test_doctor.jsonl` |
| Nurse | ~150 | `test_nurse.jsonl` |
| Pharmacist | ~100 | `test_pharm.jsonl` |
| Dentist | ~50 | `test_dentist.jsonl` |
| **Total** | **604** | `all_test.jsonl` |

### Sample Format

```json
{
  "question": "45세 남성 환자가 당뇨병으로 진단받았습니다. 다음 중 가장 적절한 초기 치료는?",
  "A": "인슐린",
  "B": "메트포르민",
  "C": "설폰요소제",
  "D": "GLP-1 작용제",
  "E": "식이요법만",
  "answer": "B"
}
```

## Verification

### Check if test samples were removed

```bash
# Count samples before and after
echo "Before:"
wc -l data/division_added/type3_word/train.jsonl

echo "After:"
wc -l data/division_added_clean/type3_word/train.jsonl

# Check stats
cat data/division_added_clean/test_exclusion_stats.json | python3 -m json.tool
```

### Verify no overlap

```bash
# Extract questions from clean training data
python3 -c "
import json
with open('data/division_added_clean/type3_word/train.jsonl') as f:
    train_q = set()
    for line in f:
        sample = json.loads(line)
        # Extract question from prompt
        prompt = sample.get('prompt', '')
        train_q.add(prompt)

# Load test questions
with open('data/kormedmcqa_test/test_questions.txt') as f:
    test_q = set(line.strip() for line in f)

# Check overlap
overlap = train_q & test_q
print(f'Overlap: {len(overlap)} samples')
if overlap:
    print('WARNING: Test contamination detected!')
else:
    print('✓ No test contamination')
"
```

## Integration with Phase 5

### Updated Pipeline

```bash
# 1. Annotate with divisions (Phase 5)
bash phase5_subject_training/scripts/run_division_pipeline.sh

# 2. Exclude test samples (NEW)
bash scripts/run_test_exclusion.sh

# 3. Train on clean data
python3 scripts/train_loop_until_90.py \
    --data-dir data/division_added_clean

# 4. Evaluate on test set
python3 scripts/evaluate_kormedmcqa.py \
    --test-file data/kormedmcqa_test/all_test.jsonl
```

## Common Issues

### Issue: Test file not found

```bash
# Error: test_questions.txt not found
# Solution: Run extraction first
python3 scripts/extract_kormedmcqa_test.py
```

### Issue: No samples removed

```bash
# Check if test questions loaded
python3 -c "
with open('data/kormedmcqa_test/test_questions.txt') as f:
    print(f'Test questions: {sum(1 for _ in f)}')
"

# Should show: Test questions: 604 (or similar)
```

### Issue: Want to re-run exclusion

```bash
# Delete clean directory
rm -rf data/division_added_clean

# Re-run
bash scripts/run_test_exclusion.sh
```

## Best Practices

✅ **Always use clean data for training**
```bash
--train-data data/division_added_clean/type3_word/train.jsonl
```

✅ **Keep test set separate**
```bash
# Never mix test with training
data/kormedmcqa_test/  → For evaluation only
```

✅ **Verify no contamination**
```bash
# Check stats after exclusion
cat data/division_added_clean/test_exclusion_stats.json
```

✅ **Document which data was used**
```bash
# In experiment notes:
# Trained on: data/division_added_clean/
# Evaluated on: data/kormedmcqa_test/all_test.jsonl
# Test samples excluded: 604
```

## Summary

| Step | Command | Output |
|------|---------|--------|
| Extract test | `python3 scripts/extract_kormedmcqa_test.py` | `data/kormedmcqa_test/` |
| Clean data | `python3 scripts/exclude_test_from_training.py` | `data/division_added_clean/` |
| **Full pipeline** | `bash scripts/run_test_exclusion.sh` | Both above |

**Result:**
- ✅ 604 test samples extracted for evaluation
- ✅ Test samples removed from training data  
- ✅ Clean training folders ready
- ✅ No test contamination

**Use `data/division_added_clean/` for all training!**
