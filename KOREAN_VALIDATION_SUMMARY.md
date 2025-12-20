# Korean Proficiency - Quick Summary

## ✅ Status: STRONG Korean Coverage

**Total:** 157,167 Korean medical samples (85% of dataset)

## 📊 Data Breakdown

| Type | Korean? | Samples | Ratio |
|------|---------|---------|-------|
| Type 1 (TEXT) | ✅ Yes | 131,591 | 89% Korean |
| Type 2 (REASONING) | ✅ Yes | 25,576 | 80% Korean |
| Type 3 (WORD) | ⚠️ Letters | 18,547 | A/B/C/D/E |
| Type 4 (WORD_REASONING) | ⚠️ English | 8,842 | English reasoning |

## 🎯 Primary Validation: KorMedMCQA

**Dataset:** 604 Korean medical MCQs
**Target:** ≥90% accuracy
**Use:** Gold standard for Korean medical knowledge

```bash
# Train until 90% KorMedMCQA
python3 scripts/train_loop_until_90.py --model medgemma-27b
```

## 🔍 Quick Check

```bash
# Validate Korean proficiency
python3 scripts/validate_korean_proficiency.py --all
```

**Expected Output:**
```
Type 1: 100% Korean (✓)
Type 2: 100% Korean (✓)
Average Korean ratio: 89%
Medical terms: 80% coverage
```

## 📖 Available Benchmarks

1. **KorMedMCQA** - 604 test samples (PRIMARY)
2. **KMMLU-Medical** - Korean MMLU medical subjects
3. **MedQA-Korean** - 22.9K Korean medical questions

## ⚡ Quick Commands

```bash
# Check proficiency
python3 scripts/validate_korean_proficiency.py --all

# Evaluate KorMedMCQA
python3 scripts/train_loop_until_90.py --model medgemma-27b

# List benchmarks
python3 scripts/validate_korean_proficiency.py --benchmarks
```

## 📈 Quality Metrics

✅ **Korean character ratio:** 85-89% (excellent)
✅ **Medical terminology:** 70-80% coverage (good)
✅ **Sample count:** 157K+ (sufficient)
⚠️ **Type 4 reasoning:** English (can be improved)

## 📝 Documentation

- **KOREAN_PROFICIENCY.md** - Complete validation guide
- **scripts/validate_korean_proficiency.py** - Validation script
- **CLAUDE.md** - Updated with Korean section

## 🎓 Conclusion

**Korean proficiency is STRONG** for medical content:
- 157K high-quality Korean medical samples
- 89% Korean character ratio in text responses
- 80% medical terminology coverage
- KorMedMCQA available for standardized evaluation

**Use KorMedMCQA (604 samples) as primary metric** for Korean medical knowledge validation.
