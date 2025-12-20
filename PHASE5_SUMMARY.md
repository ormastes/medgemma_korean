# Phase 5: Division-Based Subject Training - Summary
**Created:** 2025-12-19 03:41 UTC

## Current Situation ✅ Strategy Set

### What's Running
- **Type 2 annotation:** 16,100/23,018 samples (70%) on TITAN RTX
  - Process PID 251072, running 20+ hours
  - File: `data/division_added/type2_text_reasoning/train.jsonl` (165MB)
  - **Status:** Active (110% CPU), file last updated 03:35

### Critical Problem 🔴
**Division quality is very poor:**
- 87.5% annotations = "UNKNOWN" (should be <5%)
- 12.5% annotations = numbers "1", "2", "3" instead of "내과", "외과"
- **Root cause:** DeepSeek prompt/parsing failing for Korean medical divisions

### Agreed Strategy: Option A + Sequential (4→3→1)

#### Phase 1: Let Type 2 Complete ⏳
- **Action:** Monitor and wait
- **Reason:** 70% done, too much progress to waste
- **Monitor:** 
  ```bash
  watch -n 60 'ls -lh data/division_added/type2_text_reasoning/train.jsonl'
  ```
- **ETA:** Unknown (file not growing, may complete soon or may be stuck)

#### Phase 2: Analyze & Fix 🔧
When Type 2 completes:
1. **Count divisions:**
   ```bash
   cat data/division_added/type2_text_reasoning/train.jsonl | \
     jq -r '.primary_division' | sort | uniq -c | sort -rn
   ```

2. **Identify issues:**
   - Why 87% "UNKNOWN"? Non-medical contamination?
   - Why numbers instead of division names?
   - Check DeepSeek prompt effectiveness

3. **Fix annotation script:**
   - Improve Korean medical division extraction
   - Add validation for division names
   - Test on small Type 4 sample

#### Phase 3: Sequential Annotation (4→3→1) 🎯

**Order:** Smallest → Largest (test fixes on small data first)

1. **Type 4 (WORD_REASONING)** - 7,957 samples
   - Good for testing fixes (smallest dataset)
   - Run with improved script
   - Device: TITAN RTX (cuda:1)
   - **ETA:** ~2-3 hours

2. **Type 3 (WORD)** - 16,701 samples
   - If Type 4 quality good, proceed
   - Device: TITAN RTX (cuda:1)
   - **ETA:** ~4-5 hours

3. **Type 1 (TEXT)** - 118,431 samples
   - Largest - run last
   - Device: TITAN RTX (cuda:1)
   - **ETA:** ~30-36 hours

### Total Timeline
- **Type 2 finish:** 2-6 hours (uncertain)
- **Fix & validate:** 1-2 hours  
- **Type 4:** 2-3 hours
- **Type 3:** 4-5 hours
- **Type 1:** 30-36 hours
- **Total:** ~40-52 hours

### Files Updated
- ✅ `CLAUDE.md` - Added Phase 5 status section
- ✅ `DIVISION_ANNOTATION_STARTED.md` - Strategy doc
- ✅ `DIVISION_ANNOTATION_STATUS.md` - Current status
- ✅ `FAST_DIVISION_STATUS.md` - Urgent checks
- ✅ `PHASE5_SUMMARY.md` - This file

### Next Manual Actions

**Immediate (now):**
1. Monitor Type 2 file growth every hour
2. Check process state if no growth after 1 hour

**After Type 2 completes:**
1. Analyze division distribution
2. Fix `scripts/fast_division_annotation.py`
3. Test on small Type 4 sample (100 samples)
4. If quality good (>80% valid divisions), run full Type 4
5. Proceed to Type 3, then Type 1

**Don't start new annotation until Type 2 finishes and fixes are validated!**

---

## Division Annotation Quality Targets

| Metric | Target | Type 2 Current |
|--------|--------|----------------|
| Valid divisions | ≥80% | 12.5% ❌ |
| UNKNOWN rate | ≤5% | 87.5% ❌ |
| Numeric errors | 0% | 12.5% ❌ |
| Medical relevance | ≥95% | Unknown |

**Conclusion:** Type 2 quality is far below targets. Fixes required before Types 4, 3, 1.

