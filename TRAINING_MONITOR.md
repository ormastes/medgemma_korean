# Training Monitor & Resume Guide

## Quick Commands

### 1. Check Training Status
```bash
./check_training_status.sh
```
Shows:
- Is training running?
- Latest checkpoints
- Recent loss values
- GPU usage
- Last log activity

### 2. Watch Live Training Logs
```bash
./watch_training.sh
```
Shows live updates of:
- Loss values
- Epoch progress
- Accuracy metrics
- Errors/warnings

Press `Ctrl+C` to stop watching (training continues)

### 3. Resume Training

**Foreground (see output directly):**
```bash
./resume_training.sh
```

**Background (run in background):**
```bash
./resume_training.sh --background
```

**How Resume Works:**
- HuggingFace Trainer automatically detects checkpoints in `model/00_training/medgemma-4b/`
- Resumes from the latest checkpoint (checkpoint-3500 currently)
- Continues training from step 3501
- No data loss!

### 4. View Full Raw Log
```bash
tail -f logs/train_00_plain_text.log
```
Press `Ctrl+C` to stop watching

### 5. Stop Training
```bash
# Find the process ID
ps aux | grep train_00_plain_text.py

# Kill it
kill <PID>

# Or if you saved PID file:
kill $(cat logs/train_00_plain_text.pid)
```

---

## Current Training Status

**Checkpoint:** 3500/7813 (44.8% complete)
**Loss:** 6.14 → 2.68 (56.3% reduction)
**Epoch:** 0.448 / 1.0
**Estimated remaining:** ~37 hours

**Resume ready:** ✅ Yes, can resume from checkpoint-3500

---

## Log Analysis

### Extract Loss Summary
```bash
python3 << 'EOF'
import json
with open('model/00_training/medgemma-4b/checkpoint-3500/trainer_state.json') as f:
    data = json.load(f)
losses = [x for x in data['log_history'] if 'loss' in x]
print(f"Steps: {len(losses)}")
print(f"Current: {losses[-1]['step']} / {data['max_steps']}")
print(f"Loss: {losses[-1]['loss']:.4f}")
print(f"Epoch: {data['epoch']:.3f}")
EOF
```

### Check Checkpoint Files
```bash
ls -lth model/00_training/medgemma-4b/checkpoint-*/
```

### View Training Arguments
```bash
python3 << 'EOF'
import torch
args = torch.load('model/00_training/medgemma-4b/checkpoint-3500/training_args.bin')
print(f"Learning rate: {args.learning_rate}")
print(f"Batch size: {args.per_device_train_batch_size}")
print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
print(f"Max steps: {args.max_steps}")
print(f"Save steps: {args.save_steps}")
EOF
```

---

## Training Flow

```
Start → Load checkpoint-3500 → Resume from step 3501 → Train to step 7813 → Done
```

**Checkpoints saved every 500 steps:**
- checkpoint-3000 ✅
- checkpoint-3500 ✅ (current)
- checkpoint-4000 (next)
- checkpoint-4500
- ...
- checkpoint-7500
- Final model → model/00_trained/medgemma-4b/

---

## GPU Monitoring

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Check GPU Memory
```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

---

## Troubleshooting

### Training hangs / no progress
```bash
# Check if process is alive
ps aux | grep train_00

# Check GPU is being used
nvidia-smi

# Check last log entry time
stat logs/train_00_plain_text.log
```

### Out of memory
- Training uses gradient checkpointing (enabled)
- Batch size: 2, gradient accumulation: 16
- Uses 8-bit quantization
- Should fit in RTX A6000 (48GB)

### Want to change settings
Edit `script/train/training_config.py`:
```python
MODEL_CONFIGS = {
    "medgemma-4b": {
        "batch": 2,           # Reduce if OOM
        "grad_accum": 16,     # Increase for smaller batch
        "lr": 0.0001,         # Learning rate
        ...
    }
}
```

---

## Example Workflow

```bash
# 1. Check current status
./check_training_status.sh

# 2. Resume training in background
./resume_training.sh --background

# 3. Watch progress (in another terminal)
./watch_training.sh

# 4. Check status periodically
watch -n 60 ./check_training_status.sh

# 5. When done, check results
ls -lh model/00_trained/medgemma-4b/
```

---

## File Locations

| File/Directory | Purpose |
|----------------|---------|
| `model/00_training/medgemma-4b/` | Training checkpoints |
| `model/00_training/medgemma-4b/checkpoint-3500/` | Latest checkpoint |
| `model/00_trained/medgemma-4b/` | Final trained model |
| `logs/train_00_plain_text.log` | Training log |
| `logs/train_00_plain_text.pid` | Process ID |

---

## Auto-Resume Feature

✅ **Automatic Resume:** If training crashes or is stopped, simply run `./resume_training.sh` again. The Trainer will automatically:
1. Detect the latest checkpoint
2. Load model weights
3. Restore optimizer state
4. Continue from the exact step

**No manual intervention needed!**
