# Division Annotation Strategy - Sequential Execution
**Started:** 2025-12-19 03:35 UTC

## Strategy: Option A + Sequential (4→3→1)

### Phase 1: Wait for Type 2 to Complete ✅
- **Current:** 16,076/23,018 (70%)
- **ETA:** 06:35 UTC (3 hours)
- **Action:** Monitor and wait

### Phase 2: Analyze & Fix Division Quality Issues
- Examine Type 2 output for "UNKNOWN" divisions
- Fix division extraction/parsing logic
- Update annotation script with fixes
- **ETA Start:** 06:35 UTC

### Phase 3: Sequential Annotation (4→3→1)
After Type 2 completes and fixes are applied:

1. **Type 4 (WORD_REASONING)** - 7,957 samples
   - Smallest dataset - good for testing fixes
   - **Device:** TITAN RTX (cuda:1)
   - **ETA:** ~2-3 hours

2. **Type 3 (WORD)** - 16,701 samples  
   - Medium dataset
   - **Device:** TITAN RTX (cuda:1)
   - **ETA:** ~4-5 hours

3. **Type 1 (TEXT)** - 118,431 samples
   - Largest dataset - run last
   - **Device:** TITAN RTX (cuda:1)
   - **ETA:** ~30-36 hours

### Total Timeline
- **Type 2 finish:** 06:35 UTC (3h)
- **Fix & restart:** 07:00 UTC (+25min)
- **Type 4:** 07:00-10:00 UTC (3h)
- **Type 3:** 10:00-15:00 UTC (5h)
- **Type 1:** 15:00-Next Day 03:00 UTC (36h)
- **Total:** ~48 hours to complete all divisions

## Monitoring

### Current Type 2 Progress
```bash
# Check progress
tail -f logs/division_annotation_type2.log | grep "Batch\|samples"

# Check GPU
nvidia-smi -l 5

# Check output files
ls -lh data/division_added/type2_text_reasoning/
```

### Next Steps (Manual)
1. ✅ Let Type 2 finish (wait ~3 hours)
2. ⏳ Analyze division quality when complete
3. ⏳ Fix division parsing issues
4. ⏳ Start Type 4 with fixes
5. ⏳ Monitor and proceed to Type 3
6. ⏳ Finally process Type 1
