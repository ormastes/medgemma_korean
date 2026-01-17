# GPU Memory Guide

## ✅ GPU Memory Status: Plenty of Space Available!

### GPU 0: NVIDIA RTX A6000 (Recommended)
- **Total:** 48 GB
- **Free:** 47.4 GB (~99% available)
- **Status:** ✅ Ready for any model
- **Use for:** All training (train_00, 01, 02)

### GPU 1: NVIDIA TITAN RTX
- **Total:** 24 GB  
- **Free:** 20.8 GB (~87% available)
- **Status:** ⚠️ Currently running translation
- **Use for:** 4B model training (backup)

---

## Model Memory Requirements

### medgemma-4b ✅ Recommended
| Component | Memory |
|-----------|--------|
| Model (8-bit) | ~5 GB |
| Training overhead | ~3 GB |
| Batch overhead | ~2 GB |
| **Total minimum** | **~10 GB** |

**Verdict:** ✅ Fits comfortably on both GPUs

### medgemma-27b
| Component | Memory |
|-----------|--------|
| Model (8-bit) | ~18 GB |
| Training overhead | ~8 GB |
| Batch overhead | ~4 GB |
| **Total minimum** | **~30 GB** |

**Verdict:** ⚠️ Only fits on GPU 0

---

## Training Commands (Ready to Use)

### Check Memory First
```bash
python script/check_gpu_memory.py --model medgemma-4b
```

### train_00: Plain Text
```bash
python script/train/train/train_00_plain_text.py \
  --model medgemma-4b \
  --device cuda:0 \
  --epochs 3
```

### train_01: Medical Dictionary
```bash
python script/train/train/train_01_medical_dict.py \
  --model medgemma-4b \
  --device cuda:0 \
  --epochs 5
```

### train_02: Korean MCQ
```bash
python script/train/train/train_02_kor_med_test.py \
  --model medgemma-4b \
  --device cuda:0 \
  --epochs 3
```

---

## Memory Optimization (If Needed)

### If You See OOM Errors

1. **Reduce batch size**
   - Edit `training_config.py`: `"batch": 2` → `"batch": 1`

2. **Reduce gradient accumulation**
   - Edit `training_config.py`: `"grad_accum": 8` → `"grad_accum": 4`

3. **Reduce sequence length**
   - Edit `training_config.py`: `"max_length": 512` → `"max_length": 256`

4. **Use smaller model**
   - Use `--model medgemma-4b` instead of `medgemma-27b`

---

## Current Status Summary

| Resource | Status | Available |
|----------|--------|-----------|
| GPU 0 Memory | ✅ Excellent | 47.4 GB / 48 GB |
| GPU 1 Memory | ✅ Good | 20.8 GB / 24 GB |
| Disk Space | ✅ Adequate | Check with `df -h` |
| RAM | ✅ Good | 110 GB available |

**Conclusion: You have plenty of GPU memory for training medgemma-4b! 🚀**

---

## Quick Reference

```bash
# Check GPU status anytime
nvidia-smi

# Check detailed memory
python script/check_gpu_memory.py

# Monitor during training
watch -n 1 nvidia-smi
```

---

Generated: 2025-12-20 08:47 UTC
