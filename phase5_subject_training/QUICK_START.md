# Phase 5: Quick Start Guide

## What is Phase 5?

Phase 5 adds **medical subject division annotations** to your training data using DeepSeek, then trains models with per-division performance tracking.

**Key Benefits:**
- 📊 Know which medical subjects are weak
- 🎯 Target specific divisions for improvement
- ✅ Auto-validate division annotations
- 📈 Track accuracy/loss per medical specialty

## Prerequisites

✅ Completed Phase 4 (reviewed data in `data/reviewed/`)
✅ GPU setup: A6000 (cuda:1) + RTX A6000 (cuda:0)
✅ DeepSeek model accessible via HuggingFace

## One-Command Run

```bash
bash phase5_subject_training/scripts/run_pipeline.sh
```

This will:
1. ✨ Annotate all 4 types with DeepSeek (A6000)
2. ✅ Validate and fix annotations
3. 🚀 Train with division tracking (A6000)
4. 📊 Generate division reports

**Time:** ~4-8 hours for all types (depends on dataset size)

## Step-by-Step Run

### Step 1: Annotate with DeepSeek (~2 hours)

```bash
python3 phase5_subject_training/scripts/annotate_with_deepseek.py \
    --input data/reviewed/type1_text/train/data.jsonl \
    --output data/division/type1_text/train.jsonl \
    --device cuda:1
```

**What it does:**
- Loads DeepSeek on A6000
- Analyzes each question/answer
- Assigns medical divisions (e.g., "1.4.1" = Cardiovascular/Ischemic Heart Disease)
- Saves checkpoints every 100 samples

### Step 2: Validate Annotations (~5 minutes)

```bash
python3 phase5_subject_training/scripts/validate_divisions.py \
    --input data/division/type1_text/train.jsonl \
    --output data/division/type1_text/train_fixed.jsonl \
    --fix
```

**What it does:**
- Checks division IDs are valid
- Fixes missing/malformed fields
- Reports statistics

### Step 3: Train with Division Tracking (~2 hours)

```bash
python3 phase5_subject_training/scripts/train_with_divisions.py \
    --train-data data/division/type1_text/train.jsonl \
    --val-data data/division/type1_text/validation.jsonl \
    --model google/gemma-2-2b-it \
    --output-dir phase5_subject_training/models/type1_text \
    --epochs 3
```

**What it does:**
- Trains model normally
- During evaluation: tracks accuracy/loss per division
- Saves `division_report.json`

## Understanding Results

After training, check `phase5_subject_training/models/type1_text/division_report.json`:

```json
{
  "1": {
    "accuracy": 0.8750,
    "count": 1234,
    "avg_loss": 0.4521
  },
  "5.4.1": {
    "accuracy": 0.6534,
    "count": 234,
    "avg_loss": 0.8234
  }
}
```

**Interpretation:**
- ✅ Division "1" (Cardiovascular): Strong (87.5% accuracy)
- ⚠️ Division "5.4.1" (Diabetes): Weak (65% accuracy, high loss)

**Action:** Add more training data for weak divisions!

## Medical Divisions Reference

1. **Cardiovascular Medicine**
2. **Respiratory Medicine**
3. **Gastroenterology and Hepatology**
4. **Nephrology**
5. **Endocrinology and Metabolism**
6. **Hematology and Oncology**
7. **Neurology**
8. **Infectious Diseases**
9. **Emergency and Critical Care**
10. **Ethics, Law, and Patient Safety**

Full taxonomy: `med_division.json`

## Troubleshooting

**DeepSeek OOM on A6000:**
```bash
# Use smaller model
--model deepseek-ai/deepseek-llm-7b-chat
```

**Annotation too slow:**
- Checkpoint system auto-saves every 100 samples
- Can resume from checkpoint
- Reduce temperature for faster generation

**Training fails:**
```bash
# Check data format
python3 phase5_subject_training/scripts/validate_divisions.py \
    --input data/division/type1_text/train.jsonl
```

**Want to re-annotate:**
```bash
# Just delete old annotations and re-run
rm -rf data/division/type1_text/
```

## What's Next?

After Phase 5:
1. 📊 Review all division reports
2. 🔍 Identify divisions with accuracy < 80%
3. 📚 Add more training data for weak divisions
4. 🔁 Optionally: Re-run Phase 5
5. ➡️ Proceed to Phase 6 (Evaluation)

## Quick Verification

Test your setup:
```bash
python3 phase5_subject_training/scripts/test_setup.py
```

Should show:
```
✓ All tests passed! Phase 5 setup is complete.
```

## Files Generated

```
data/division/
├── type1_text/
│   ├── train.jsonl          (118K samples with divisions)
│   └── validation.jsonl     (13K samples with divisions)
├── type2_text_reasoning/
├── type3_word/
└── type4_word_reasoning/

phase5_subject_training/models/
├── type1_text/
│   ├── final/               (trained model)
│   ├── division_report.json (🎯 CHECK THIS!)
│   └── checkpoint-*/
...
```

## Tips

💡 **Run overnight**: Full pipeline takes 4-8 hours
💡 **Monitor GPU**: `watch -n 1 nvidia-smi`
💡 **Check logs**: tail -f phase5_subject_training/logs/*.log
💡 **Save reports**: Division reports are key outputs!

---

Ready? Run this:

```bash
bash phase5_subject_training/scripts/run_pipeline.sh
```
