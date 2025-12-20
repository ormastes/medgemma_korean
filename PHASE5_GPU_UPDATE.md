# Division Annotation - All 4 Types Running

**Started:** 2025-12-18 06:45 UTC (Type 4) + 06:52 UTC (Types 1,2,3)

## Current Status: ✅ ALL RUNNING IN PARALLEL

| Type | GPU | Batch | Status | Log |
|------|-----|-------|--------|-----|
| Type 4 (WORD_REASONING) | cuda:1 (TITAN RTX) | 8 | Running | `logs/type4_direct.log` |
| Type 3 (WORD) | cuda:0 (A6000) | 8 | Running | `logs/type3_final.log` |
| Type 2 (TEXT_REASONING) | cuda:1 (TITAN RTX) | 4 | Running | `logs/type2_final.log` |
| Type 1 (TEXT) | cuda:0 (A6000) | 4 | Running | `logs/type1_final.log` |

## GPU Allocation

**cuda:0 (A6000 48GB):** Type 3 + Type 1 (alternating batches)
**cuda:1 (TITAN RTX 24GB):** Type 4 + Type 2 (alternating batches)

## Why So Slow?

The first batch takes 4+ minutes! This is because:
1. **Long prompts:** Type 4 completions are 5,779 chars (truncated to 200)
2. **Model loading:** First batch includes compilation/optimization
3. **Batching overhead:** Multiple models sharing same GPU

## Expected Timeline

Assuming first batch is slow but later batches are 3s each:

| Type | Samples | Est. Time | Complete By |
|------|---------|-----------|-------------|
| Type 4 | 7,957 | ~1 hour | 07:52 UTC |
| Type 3 | 16,701 | ~2 hours | 08:52 UTC |
| Type 2 | 23,018 | ~3 hours | 09:52 UTC |
| Type 1 | 118,431 | ~15 hours | 21:52 UTC |

**Earliest all complete:** Tonight ~22:00 UTC

## Monitor

```bash
# Check progress
./monitor_division_annotation.sh

# Live logs
tail -f logs/type4_direct.log
tail -f logs/type3_final.log
tail -f logs/type2_final.log
tail -f logs/type1_final.log

# Check sample counts
wc -l data/division_added/*/train.jsonl
```

---

**Status:** ✅ All 4 types running in parallel
**ETA:** ~22:00 UTC tonight (15-16 hours from now)
