# 🎯 Phase 5: Subject Training - Quick Reference

## What's New in Phase 5?

Phase 5 adds **medical division tracking** to your training pipeline. Every data sample gets tagged with medical subject areas (Cardiology, Respiratory, etc.), then models are trained with performance tracking per division.

## One-Line Quick Start

```bash
bash phase5_subject_training/scripts/run_pipeline.sh
```

## What This Does

1. **Annotates** all training data with medical divisions using DeepSeek on TITAN RTX
2. **Validates** division annotations and auto-fixes malformed data
3. **Trains** models with per-division performance tracking on RTX A6000
4. **Generates** division reports showing which medical subjects are weak/strong

## Division Report Example

```json
{
  "1.4.1": {
    "accuracy": 0.65,     // ⚠️ WEAK - needs more data!
    "count": 234,
    "avg_loss": 0.82
  },
  "2.4.3": {
    "accuracy": 0.92,     // ✅ STRONG
    "count": 156,
    "avg_loss": 0.34
  }
}
```

## Medical Divisions (10 Major)

1. Cardiovascular Medicine
2. Respiratory Medicine
3. Gastroenterology and Hepatology
4. Nephrology
5. Endocrinology and Metabolism
6. Hematology and Oncology
7. Neurology
8. Infectious Diseases
9. Emergency and Critical Care
10. Ethics, Law, and Patient Safety

## Files Created

```
data/division/          # Division-annotated data (NEW)
phase5_subject_training/
├── models/            # Trained models
│   └── */division_report.json  # 🎯 KEY OUTPUT
└── scripts/           # Pipeline scripts
```

## Time Estimates

- **Annotation**: ~5.5 hours (DeepSeek on TITAN RTX)
- **Training**: ~4.5 hours (on RTX A6000)
- **Total**: ~10 hours (run overnight)

## Verification

```bash
python3 phase5_subject_training/scripts/test_setup.py
```

Should show: `✓ All tests passed!`

## Documentation

- `phase5_subject_training/QUICK_START.md` - Quick start guide
- `phase5_subject_training/README.md` - Full documentation
- `CLAUDE.md` - Updated with Phase 5 section
- `PHASE5_SUMMARY.md` - Complete setup summary

## Next Steps

After Phase 5:
1. Review division reports
2. Identify weak divisions (accuracy < 80%)
3. Add more training data for weak areas
4. Proceed to Phase 6 (Evaluation)

---

**Ready to run?**

```bash
bash phase5_subject_training/scripts/run_pipeline.sh
```
