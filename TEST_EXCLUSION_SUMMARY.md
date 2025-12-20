# KorMedMCQA Test Exclusion - Quick Summary

## 🎯 Critical: Prevent Test Contamination

**Problem:** KorMedMCQA test set (604 samples) may be in training data
**Solution:** Extract test set and remove from training

## ⚡ One Command

```bash
bash scripts/run_test_exclusion.sh
```

**Output:**
- `data/kormedmcqa_test/` - 604 test samples for evaluation
- `data/division_added_clean/` - Training data WITHOUT test

## 📊 What You Get

### Test Set (For Evaluation)
```
data/kormedmcqa_test/
├── test_doctor.jsonl      (~300 samples)
├── test_nurse.jsonl       (~150 samples)
├── test_pharm.jsonl       (~100 samples)
├── test_dentist.jsonl     (~50 samples)
└── all_test.jsonl         (604 total)
```

### Clean Training Data
```
data/division_added_clean/
├── type1_text/            (test samples removed)
├── type2_text_reasoning/
├── type3_word/            (test samples removed)
├── type4_word_reasoning/
├── 1/                     (division folders - cleaned)
├── 2/
├── ...
└── test_exclusion_stats.json
```

## ✅ Correct Usage

```bash
# ✅ Train on CLEAN data
python3 train.py --data data/division_added_clean/

# ✅ Evaluate on TEST set
python3 evaluate.py --test data/kormedmcqa_test/all_test.jsonl
```

## ❌ Wrong Usage

```bash
# ❌ DON'T use original folder (test contamination)
python3 train.py --data data/division_added/
```

## 🔍 Verification

```bash
# Check how many test samples were removed
cat data/division_added_clean/test_exclusion_stats.json

# Should show:
# "removed": 604 (or similar)
# "removal_rate": ~0.36%
```

## 📝 Scripts

| Script | Purpose |
|--------|---------|
| `extract_kormedmcqa_test.py` | Extract 604 test samples |
| `exclude_test_from_training.py` | Remove test from training |
| `run_test_exclusion.sh` | Run both above |

## 🔄 Complete Workflow

```
1. Phase 5: Add divisions
   bash phase5_subject_training/scripts/run_division_pipeline.sh
   
2. Exclude test (NEW - REQUIRED)
   bash scripts/run_test_exclusion.sh
   
3. Train on CLEAN data
   python3 train.py --data data/division_added_clean/
   
4. Evaluate on test set
   python3 evaluate.py --test data/kormedmcqa_test/all_test.jsonl
```

## 📈 Why This Matters

| Without Exclusion | With Exclusion |
|-------------------|----------------|
| ❌ Test samples in training | ✅ Test excluded |
| ❌ Inflated accuracy | ✅ True accuracy |
| ❌ Invalid results | ✅ Valid results |
| ❌ Data leakage | ✅ No leakage |

## 📖 Documentation

- **KORMEDMCQA_TEST_EXCLUSION.md** - Complete guide
- **CLAUDE.md** - Updated with test exclusion
- **KOREAN_VALIDATION_SUMMARY.md** - Korean proficiency

## 🎓 Key Points

1. ⚠️ **Always use `data/division_added_clean/` for training**
2. 📊 **Use `data/kormedmcqa_test/all_test.jsonl` for evaluation**
3. ✅ **Run `run_test_exclusion.sh` before training**
4. 🔍 **Check `test_exclusion_stats.json` to verify**

---

**Status:** Test exclusion system ready!

**Run:** `bash scripts/run_test_exclusion.sh`
