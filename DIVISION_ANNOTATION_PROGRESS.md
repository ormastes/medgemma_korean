# Division Annotation Progress Report
**Updated:** 2025-12-19 03:32 UTC (~21 hours running)

## ✅ GREAT NEWS: Type 2 is 70% Done!

### Current Status

| Type | Status | Progress | Output | Runtime | ETA |
|------|--------|----------|--------|---------|-----|
| Type 1 (TEXT) | ❌ Crashed | 608/118,431 (0.5%) | 600 samples | ~45min | - |
| Type 2 (TEXT_REASONING) | ✅ RUNNING | 16,076/23,018 (70%) | 16,000 samples | 21 hours | **6 hours** |
| Type 3 (WORD) | ❌ Crashed | 1,024/16,701 (6%) | 1,000 samples | ~45min | - |
| Type 4 (WORD_REASONING) | ❌ Crashed | 80/7,957 (1%) | 0 samples | ~10min | - |

### GPU Status
- **cuda:0 (A6000):** 23% idle - available for restart
- **cuda:1 (TITAN RTX):** 98% util - Type 2 running strong

## Type 2 Will Complete by 09:30 UTC Today!

**Current speed:** ~4 samples/batch at ~4.6s/batch = ~1.15s/sample
**Remaining:** 7,018 samples
**Time needed:** 7,018 × 1.15s = 8,070s = **2.2 hours**
**ETA:** **05:45 UTC** (in 2.2 hours)

After train completes, it will process validation (~2,600 samples) = +50min
**Final ETA: 06:35 UTC (in 3 hours)**

## Quality Check

### Type 2 (16K samples annotated):
```json
{
  "divisions": ["UNKNOWN"],
  "prompt": "13세 남자 환자...",
  "completion": "..."
}
```

⚠️ **PROBLEM:** Most divisions are "UNKNOWN"! DeepSeek is not assigning proper divisions.

### Type 3 (1K samples annotated):
```json
{
  "divisions": ["1", "2", "1"],
  "prompt": "배뇨장애...",
  "completion": "D"
}
```

⚠️ **PROBLEM:** Divisions are numbers "1", "2" instead of division names like "비뇨의학과"!

## Issues Found

1. **Crashes:** Types 1, 3, 4 all crashed after 10-45 minutes
   - Likely OOM or GPU contention
   - Partial data was saved before crash

2. **Division Quality:** Annotations are malformed
   - Type 2: Returning "UNKNOWN" instead of real divisions
   - Type 3: Returning "1", "2" instead of division names
   - Need to verify/fix division extraction logic

## Next Steps

### Option A: Let Type 2 finish, then fix & restart
1. Wait 3 hours for Type 2 to complete (06:35 UTC)
2. Analyze what went wrong with division parsing
3. Fix the malformed division script
4. Restart all 4 types with fixes

### Option B: Stop everything, fix now, restart all
1. Stop Type 2 (lose 16K samples)
2. Fix division parsing immediately
3. Restart all 4 types fresh

### Option C: Continue Type 2, start others with fixes
1. Let Type 2 finish (3 hours)
2. Fix division script NOW
3. Start Types 1, 3, 4 with fixed script
4. Fix Type 2 divisions retroactively

---

**Recommendation:** **Option C**
- Type 2 is 70% done - too much progress to waste
- Can fix divisions post-processing with malform checker
- Start Types 1, 3, 4 NOW with reduced batch sizes to avoid crashes
