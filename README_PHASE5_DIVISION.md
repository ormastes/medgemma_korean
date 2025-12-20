# 🎯 Phase 5: Division-Based Organization - Quick Start

## What Is This?

Phase 5 creates **division-specific datasets** from your reviewed medical data:
- Uses DeepSeek AI to classify questions by medical specialty
- Organizes data into division folders (Cardiology, Respiratory, etc.)
- Enables targeted training for specific medical subjects

## 🚀 One-Command Start

```bash
bash phase5_subject_training/scripts/run_division_pipeline.sh
```

**Time:** ~6 hours | **GPU:** TITAN RTX (cuda:1)

## 📊 What You Get

```
data/division_added/
├── 1/                           Cardiovascular (~15K samples)
├── 2/                           Respiratory (~10K samples)
├── 3/                           Gastroenterology (~8K samples)
├── 5/                           Endocrinology (~8K samples)
├── 7/                           Neurology (~6K samples)
└── division_index.json          Statistics for all divisions
```

Each division folder contains:
- `train.jsonl` - Training samples for that specialty
- `validation.jsonl` - Validation samples
- `metadata.json` - Division statistics

## 🎓 Use Cases

### 1. Train Cardiology Specialist
```bash
python3 phase5_subject_training/scripts/train_with_divisions.py \
    --train-data data/division_added/1/train.jsonl \
    --val-data data/division_added/1/validation.jsonl \
    --output-dir models/cardio_specialist
```

### 2. Boost Weak Division
If evaluation shows Division 5 (Endocrinology) is weak:
```bash
python3 phase5_subject_training/scripts/train_with_divisions.py \
    --train-data data/division_added/5/train.jsonl \
    --val-data data/division_added/5/validation.jsonl \
    --output-dir models/endo_boost \
    --epochs 10 --lr 5e-6
```

### 3. Combine Related Divisions
```bash
# Internal medicine: Cardio + Respiratory + GI
cat data/division_added/{1,2,3}/train.jsonl > internal_train.jsonl
python3 train.py --train-data internal_train.jsonl
```

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| **PHASE5_DIVISION_SUMMARY.md** | ⭐ Complete summary (read this first) |
| **phase5_subject_training/DIVISION_GUIDE.md** | Comprehensive usage guide |
| **phase5_subject_training/scripts/README.md** | Scripts reference |
| **CLAUDE.md** | Full training guide (updated) |

## ✅ Verification

```bash
python3 phase5_subject_training/scripts/verify_setup.py
```

Should show: `✅ All checks passed!`

## 📁 10 Medical Divisions

1. **Cardiovascular Medicine** - Heart, vessels, circulation
2. **Respiratory Medicine** - Lungs, breathing
3. **Gastroenterology** - Digestive system, liver
4. **Nephrology** - Kidneys, fluid balance
5. **Endocrinology** - Hormones, metabolism
6. **Hematology/Oncology** - Blood, cancer
7. **Neurology** - Brain, nerves
8. **Infectious Diseases** - Infections, antibiotics
9. **Emergency/Critical Care** - Acute conditions
10. **Ethics/Law** - Medical ethics, regulations

## ⚡ Quick Commands

**Full pipeline:**
```bash
bash phase5_subject_training/scripts/run_division_pipeline.sh
```

**Check results:**
```bash
cat data/division_added/division_index.json | python3 -m json.tool
```

**Count division samples:**
```bash
for d in data/division_added/*/train.jsonl; do
    echo "$(dirname $d): $(wc -l < $d) samples"
done
```

**Train specific division:**
```bash
python3 phase5_subject_training/scripts/train_with_divisions.py \
    --train-data data/division_added/1/train.jsonl \
    --val-data data/division_added/1/validation.jsonl \
    --output-dir models/division_1
```

## 🔄 Workflow

```
Step 1: Annotate (DeepSeek on TITAN RTX)
           ↓
Step 2: Check Quality
           ↓
Step 3: Organize by Division
           ↓
Step 4: Train Division-Specific Models (Optional)
           ↓
Step 5: Evaluate & Deploy
```

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Annotate all types | ~6 hours |
| Check quality | ~5 minutes |
| Organize | ~10 minutes |
| Train per division | ~2 hours each |

## 💡 Tips

✅ **Run overnight** - Full pipeline takes ~6 hours
✅ **Monitor GPU** - `watch -n 1 nvidia-smi`
✅ **Check division_index.json** - See which divisions have data
✅ **Start with largest divisions** - More data = better results
✅ **Combine small divisions** - If <100 samples, merge with related

## 🐛 Troubleshooting

**DeepSeek OOM:**
```bash
--model deepseek-ai/deepseek-llm-7b-chat
```

**Too many UNKNOWN:**
- Check that input data is medical-related
- Review annotation quality report

**No division folders created:**
```bash
--min-samples 5  # Lower threshold
```

## 🎯 Next Steps

1. ✅ Verify setup: `python3 verify_setup.py`
2. 🚀 Run pipeline: `bash run_division_pipeline.sh`
3. 📊 Check results: `cat division_index.json`
4. 🎓 Train specialists for weak divisions
5. 📈 Evaluate & deploy

---

**Ready to start?**

```bash
bash phase5_subject_training/scripts/run_division_pipeline.sh
```

See `PHASE5_DIVISION_SUMMARY.md` for complete details!
