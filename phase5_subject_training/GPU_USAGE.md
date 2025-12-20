# Phase 5 GPU Usage - A6000 Only

## Important: No TITAN RTX Usage

Phase 5 uses **A6000 (cuda:0) ONLY** for both annotation and training.

## Why A6000 Only?

1. **Consistency:** Same GPU for all operations
2. **Performance:** A6000 is faster and has more memory
3. **Simplification:** No need to manage multiple GPUs
4. **Memory:** 49GB vs 24GB (TITAN RTX)

## GPU Assignment

| Operation | GPU | Device | Memory |
|-----------|-----|--------|--------|
| **DeepSeek Annotation** | A6000 | cuda:0 | ~15-20GB |
| **Training (MedGemma-27B)** | A6000 | cuda:0 | ~40-48GB |
| **Training (MedGemma-4B)** | A6000 | cuda:0 | ~18GB |
| **Validation** | A6000 | cuda:0 | Variable |

**TITAN RTX:** ❌ Not used in Phase 5

## Workflow

### Step 1: Annotation (when Phase 4 completes)
```bash
# Wait for Phase 4 to finish
# Then run annotation on A6000
python3 phase5_subject_training/scripts/add_divisions_to_reviewed.py \
    --type all \
    --device cuda:0    # A6000
```

### Step 2: Training (after annotation)
```bash
# Train with divisions on A6000
python3 phase5_subject_training/scripts/train_with_divisions.py \
    --model medgemma-27b \
    --device cuda:0    # A6000
```

## Sequential Execution

Phase 5 operations run **sequentially** on A6000:

1. **Phase 4 completes** → A6000 freed
2. **Run annotation** → Uses A6000 (15-20GB)
3. **Annotation completes** → Free memory
4. **Run training** → Uses A6000 (40-48GB)

## Memory Management

### Annotation Phase
- **DeepSeek-7B:** ~15-20GB
- **Safety margin:** 29-34GB free
- **No conflict** with training

### Training Phase
- **MedGemma-27B:** ~40-48GB
- **Safety margin:** 1-9GB
- **Maximum utilization**

## Default Device Settings

All Phase 5 scripts default to **cuda:0** (A6000):

```python
# annotate_with_deepseek.py
parser.add_argument('--device', default='cuda:0')  # A6000

# train_with_divisions.py  
parser.add_argument('--device', default='cuda:0')  # A6000

# add_divisions_to_reviewed.py
parser.add_argument('--device', default='cuda:0')  # A6000
```

## TITAN RTX Status

**TITAN RTX (cuda:1):** Idle / Available for other tasks

The TITAN RTX is **not used** in Phase 5 but remains available for:
- Other projects
- Testing
- Development
- Phase 6/7 if needed

## Verification

Check GPU usage:
```bash
nvidia-smi

# Should show:
# GPU 0 (A6000): In use (Phase 5)
# GPU 1 (TITAN): Idle
```

## Summary

✅ **All Phase 5 operations use A6000 (cuda:0)**  
✅ **No TITAN RTX (cuda:1) usage**  
✅ **Sequential workflow on single GPU**  
✅ **Simpler configuration and management**
