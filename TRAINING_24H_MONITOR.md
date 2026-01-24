# 24-Hour Training Run - Live Monitor

**Start Time:** 2026-01-23 03:55:21 UTC  
**Process ID:** 211463  
**Status:** ✅ RUNNING  

---

## Training Configuration

### Checkpoint & Model
```
Base Checkpoint: checkpoint-3500
Best Loss: 2.6689 (step 3280)
Model: medgemma-4b (4B parameters)
Trainable Params: 1,528,110,080 (26.10%)
Extended Tokenizer: 272,843 tokens
```

### Training Parameters
```
Epochs: 100
Samples per Epoch: 50,000
Total Samples: 5,000,000 (100 epochs × 50K)
Batch Size: 2
Gradient Accumulation: 16
Learning Rate: 0.0001
Max Length: 512
```

### Dataset & Caching
```
Dataset: 1.5GB Korean plain text (50K subset per epoch)
Cache Status: ✅ ENABLED
Cache Loading Time: <1 second
Cache Location: data/02_refined/00_plain_text/.cache/
Cache Size: ~2.5GB
```

### GPU & Memory
```
GPU: RTX A6000 (49GB total)
Gradient Checkpointing: ENABLED
Current Memory: 7.8GB (16%)
Optimization: Memory-efficient training enabled
```

---

## Training Timeline

### Phase 1: Initialization (15-20 min)
- [x] Load base model
- [x] Load LoRA adapter
- [x] Load tokenizer
- [x] Load cached dataset
- [ ] Baseline evaluation
- [ ] Start training

### Phase 2: Continuous Training (23-30 hours)
- Epoch 1-10: ~160-180 min (2.5-3 hours)
- Epoch 11-50: ~600-700 min (10-12 hours)
- Epoch 51-100: ~900-1000 min (15-16+ hours)
- Checkpoints saved every epoch

### Phase 3: Finalization (5-10 min)
- Save final model
- Generate validation results
- Archive training logs
- Cleanup

**Estimated Total: 24-30 hours**

---

## Monitoring Commands

### Live Output
```bash
# Real-time training log
tail -f training_24h.log

# Last 100 lines
tail -100 training_24h.log

# Search for accuracy/loss
grep -E "accuracy|loss|Epoch|step" training_24h.log | tail -20
```

### Process Monitoring
```bash
# Check if running
ps aux | grep 211463

# CPU/Memory usage
ps aux | grep "train_00_plain_text" | grep -v grep

# Full process info
ps -p 211463 -o pid,user,cpu,%mem,vsize,rss,etime,command
```

### GPU Monitoring
```bash
# Real-time GPU status
watch -n 1 nvidia-smi

# Just RTX A6000
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader

# Continuous update every 5 seconds
watch -n 5 'nvidia-smi | grep -E "RTX A6000|Processes|python"'
```

### Cache Monitoring
```bash
# Cache size growth
du -sh data/02_refined/00_plain_text/.cache/

# Monitor continuously
watch -n 60 'du -sh data/02_refined/00_plain_text/.cache/'
```

### Training Progress
```bash
# Count epochs completed
grep "Epoch" training_24h.log | tail -5

# Get latest accuracy
grep "accuracy" training_24h.log | tail -1

# Estimate remaining time
echo "Calculate: (100 - <current_epoch>) × 16 minutes"
```

---

## Expected Behavior

### Startup (First 20 minutes)
```
[INFO] Loading model...
[INFO] Loading training data...
Loading cached datasets from: .../cache
[INFO] Loaded 50000 training samples
[INFO] Loading pre-initialized LoRA model...
[INFO] Resizing embeddings...
[INFO] BASELINE: Evaluating before training...
```

### Training Loop (Repeats 100 times)
```
Epoch 1/100: 
  Training: [████████████████] 25000/50000
  Step 1234: loss=2.45, learning_rate=0.0001
  Step 2345: loss=2.38, learning_rate=0.0001
  ...
  Training complete
  Saving checkpoint-XXXX
  
Epoch 2/100: ...
```

### Finalization
```
[INFO] Training complete!
[INFO] Final model saved to: .../model/00_trained/medgemma-4b
[INFO] TRAIN 00 COMPLETE
```

---

## Key Metrics to Watch

### Loss Progression
- **Starting Loss:** ~2.67 (from checkpoint-3500)
- **Expected Trend:** Gradual decrease with oscillations
- **Target:** <2.5 after 24 hours

### Validation Accuracy (if running)
- **Starting:** ~17.88% (from previous run)
- **Expected Improvement:** +0.5-2% per 10 epochs
- **Target:** 18-20% after 24 hours

### Training Speed
- **Per Epoch:** ~16 minutes (50K samples)
- **Per Step:** ~3-4 ms
- **Expected Stable:** After epoch 5-10

### Memory Usage
- **Expected Peak:** 20-25GB (with checkpointing)
- **Normal Range:** 15-20GB
- **Spikes:** During checkpoint saves

---

## Troubleshooting

### If Process Stops
```bash
# Check last log lines
tail -20 training_24h.log

# Search for errors
grep -i "error\|exception\|traceback" training_24h.log

# Check exit code
ps -p 211463 > /dev/null && echo "Still running" || echo "Stopped (exit code: $?)"
```

### If GPU Memory Issues
```bash
# Check GPU memory
nvidia-smi

# If OOM, graceful shutdown
kill -TERM 211463  # Allows final save
```

### If Stuck/Frozen
```bash
# Check if making progress
tail -5 training_24h.log
# If same output as 5 minutes ago, it's frozen

# Force stop
kill -9 211463
```

---

## Output Files

### Training Log
```
training_24h.log          - Full training output
training_24h.pid          - Process ID (211463)
```

### Checkpoints (updated after each epoch)
```
model/00_training/medgemma-4b/checkpoint-XXXX/
├── adapter_config.json
├── adapter_model.safetensors
├── training_args.bin
├── trainer_state.json
└── tokenizer files
```

### Final Model
```
model/00_trained/medgemma-4b/
├── adapter_model.safetensors   (3.1GB trained weights)
├── tokenizer.json              (extended Korean tokenizer)
├── training_info.json          (metadata)
├── kormedmcqa_validation_results.json
└── ...other files
```

---

## Quick Status Check

Run this to get current status:

```bash
#!/bin/bash
echo "=== 24H Training Status ==="
echo "Time Elapsed:"
ps -p 211463 -o etime= || echo "Not running"
echo ""
echo "Last 5 Lines of Log:"
tail -5 training_24h.log
echo ""
echo "GPU Usage:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo ""
echo "Process CPU/Memory:"
ps aux | grep 211463 | grep -v grep | awk '{print "CPU: " $3 "%, Memory: " $4 "%"}'
```

---

## Estimated Completion

- **Start:** 2026-01-23 03:55:21 UTC
- **Duration:** ~24-30 hours
- **Estimated End:** 2026-01-24 04:00:00 UTC (± 6 hours)
- **Backup Est:** 2026-01-24 09:00:00 UTC (outer bound)

---

## Next Steps After Completion

1. **Verify Training:**
   ```bash
   tail -20 training_24h.log
   ```

2. **Check Final Model:**
   ```bash
   ls -lh model/00_trained/medgemma-4b/
   ```

3. **Review Validation Results:**
   ```bash
   cat model/00_trained/medgemma-4b/kormedmcqa_validation_results.json
   ```

4. **Analyze Improvements:**
   - Compare baseline (17.88%) vs. final accuracy
   - Check loss trajectory
   - Evaluate checkpoint quality

5. **Next Training Phase:**
   - Run train_01 (medical dictionary)
   - Or run train_02 (MCQ with reasoning)
   - Or continue with additional epochs

---

## References

- **Config:** script/train/training_config.py
- **Utils:** script/train/training_utils.py
- **Script:** script/train/train_00_plain_text.py
- **Cache:** data/02_refined/00_plain_text/.cache/

---

**Status: ✅ ACTIVELY TRAINING**  
**Last Updated:** 2026-01-23 03:55:21 UTC  
**Monitor with:** `tail -f training_24h.log`
