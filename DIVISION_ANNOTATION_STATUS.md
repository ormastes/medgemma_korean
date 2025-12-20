# Division Annotation Status Report
**Updated:** 2025-12-19 03:37 UTC

## Current Status

### ✅ Type 2 Running (20+ hours)
- **Process:** PID 251072, Running since Dec 18
- **Script:** `scripts/fast_division_annotation.py`
- **Device:** cuda:1 (TITAN RTX) - 98% util
- **Progress:** Train file size 165MB
- **Issue:** ⚠️ **Divisions = "UNKNOWN"** for most samples

### Sample Check (Type 2)
```json
{
  "divisions": ["UNKNOWN"],
  "primary_division": "UNKNOWN",
  "source": "medical_o1_reasoning_ko",
  "type": "type2_text_reasoning"
}
```

**Problem:** DeepSeek is returning "UNKNOWN" instead of actual medical divisions!

This sample is about **philosophy (personhood debate)**, NOT medical content. The "medical_o1_reasoning_ko" dataset may contain non-medical reasoning examples.

## GPU Status (03:37 UTC)

| GPU | Model | Util | Memory | Temp | Status |
|-----|-------|------|--------|------|--------|
| cuda:0 | A6000 | 95% | 29.8/49.1 GB | 87°C | ❓ (High usage, no process found?) |
| cuda:1 | TITAN RTX | 98% | 8.7/24.6 GB | 86°C | ✅ Type 2 running |

⚠️ **A6000 is 95% utilized but no annotation process found!**

## Action Plan: Option A + Sequential (4→3→1)

### Step 1: Monitor Type 2 Completion ⏳
- Let Type 2 finish naturally
- **Issue:** Unknown ETA (no batch progress in logs)
- **File size:** 165MB suggests significant progress
- Check if process is actually progressing or stuck

### Step 2: Investigate Division Quality 🔍
When Type 2 completes:
1. **Count division distribution:**
   ```bash
   cat data/division_added/type2_text_reasoning/train.jsonl | \
     jq -r '.divisions[]' | sort | uniq -c | sort -rn
   ```

2. **Check if medical content:**
   - Filter samples with actual medical terms
   - Identify non-medical contamination

3. **Fix division annotation logic:**
   - Improve DeepSeek prompt for Korean medical divisions
   - Add medical division validation

### Step 3: Sequential Processing (4→3→1) 🎯
After fixes:
1. **Type 4** (7,957 samples) - ~2-3 hours
2. **Type 3** (16,701 samples) - ~4-5 hours  
3. **Type 1** (118,431 samples) - ~30-36 hours

**Total ETA:** Type 2 finish + 40-44 hours

## Immediate Actions Needed

1. **Check if Type 2 is progressing:**
   ```bash
   # Watch file size grow
   watch -n 60 'ls -lh data/division_added/type2_text_reasoning/train.jsonl'
   ```

2. **Find what's using A6000:**
   ```bash
   fuser -v /dev/nvidia0
   # or
   lsof /dev/nvidia0
   ```

3. **Prepare division quality check script**

