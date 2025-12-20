# Phase 5: Medical Division Training - Summary

## Division Tagging Attempt

**Goal:** Add medical specialty tags (Cardiology, Respiratory, etc.) to enable division-specific training

**Result:** ⚠️ **PAUSED** - Too slow for practical use

### What We Tried

1. **First attempt (Type 1):**
   - Processed 20,200 / 118,431 samples (17%)
   - Runtime: 16 hours
   - **Issue:** English prompt caused misclassifications
   
2. **Second attempt (Type 4 with Korean prompt):**
   - Processed 20 / 7,957 samples (0.25%)
   - Speed: 1.1 sec/sample
   - **Issue:** Too slow (~64 hours total for all types)

### Why So Slow?

- Long reasoning completions in Type 2 & 4 (avg 5,779 chars)
- DeepSeek-7B processes slowly even with 4-bit quantization
- Batch processing limited by memory (batch_size=4)

### Data Created

✅ **20,200 Type 1 samples with divisions** in `data/division_added/type1_text/train.jsonl`
- Quality uncertain (English prompt used)
- Can be used for initial testing

## Decision: Skip Division Tagging For Now

**Reasons:**
1. **Time cost:** 64+ hours for uncertain quality
2. **Not critical:** Can train without divisions first
3. **Can add later:** If performance plateaus on specific specialties
4. **Simpler approach:** Use all data together initially

## Current Plan

### Phase 5a: Train on All Data (No Divisions)

Use `data/reviewed/` directly:
- Type 1: 131,591 samples
- Type 2: 25,576 samples
- Type 3: 18,547 samples
- Type 4: 8,842 samples

### Phase 5b: Division Training (Future)

**If needed later:**
1. Use faster model (GPT-4o-mini API)
2. Or manually tag subset for each division
3. Or train first, then add divisions for weak specialties

## Revised Training Folder Structure

```
phase5_instruction_tuning/     # Train all types together
├── scripts/
│   ├── train_type1_text.py
│   ├── train_type2_text_reasoning.py
│   ├── train_type3_word.py
│   ├── train_type4_word_reasoning.py
│   └── train_loop_until_90.py
└── models/
    ├── loop_1/
    ├── loop_2/
    └── final/

phase6_division_training/      # Future: If needed
└── (Not started yet)
```

## Next Steps

1. ✅ Data refined → `data/reviewed/`
2. ✅ KorMedMCQA test samples excluded
3. 🔄 **Next:** Start loop training (Type 4→3→2→1)
4. 📊 Evaluate on KorMedMCQA test set
5. 🎯 Target: ≥90% accuracy

---

**Status:** Ready to start training without divisions
**ETA:** ~20 hours for one full loop through all 4 types
