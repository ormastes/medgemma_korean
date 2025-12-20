# Phase 5 Scripts - Quick Reference

## Main Pipeline

```bash
bash run_division_pipeline.sh
```

Runs all 3 steps automatically.

## Individual Scripts

### 1. Add Divisions (`add_divisions_to_reviewed.py`)

**Purpose:** Annotate reviewed data with medical divisions using DeepSeek

**Usage:**
```bash
# All types
python3 add_divisions_to_reviewed.py --type all --device cuda:1

# Single type
python3 add_divisions_to_reviewed.py --type type1_text --device cuda:1
```

**Options:**
- `--type`: Data type (type1_text, type2_text_reasoning, type3_word, type4_word_reasoning, all)
- `--model`: DeepSeek model (default: deepseek-ai/deepseek-llm-7b-chat)
- `--device`: GPU device (default: cuda:1 for TITAN RTX)

**Output:** `data/division_added/{type}/train.jsonl` and `validation.jsonl`

---

### 2. Check Divisions (`check_divisions.py`)

**Purpose:** Validate division annotations and report statistics

**Usage:**
```bash
# Check all types
python3 check_divisions.py --all

# Check single file
python3 check_divisions.py --file data/division_added/type1_text/train.jsonl
```

**Output:**
- Validity statistics
- Division distribution
- Invalid IDs
- Errors report

---

### 3. Organize by Division (`organize_by_division.py`)

**Purpose:** Create division-specific folders with train/val data

**Usage:**
```bash
python3 organize_by_division.py \
    --source data/division_added \
    --output data/division_added \
    --min-samples 10
```

**Options:**
- `--source`: Source directory
- `--output`: Output directory
- `--min-samples`: Minimum samples to create division folder (default: 10)

**Output:**
- `data/division_added/{division_id}/train.jsonl`
- `data/division_added/{division_id}/validation.jsonl`
- `data/division_added/{division_id}/metadata.json`
- `data/division_added/division_index.json`

---

## Legacy Scripts (from original Phase 5)

### `annotate_with_deepseek.py`
Original annotation script - now replaced by `add_divisions_to_reviewed.py`

### `validate_divisions.py`
Validation and auto-fix - functionality now in `check_divisions.py`

### `train_with_divisions.py`
Division-aware training - still used for training division-specific models

### `run_pipeline.sh`
Old pipeline - replaced by `run_division_pipeline.sh`

---

## Workflow

```
1. add_divisions_to_reviewed.py
       ↓
   Annotates reviewed data
       ↓
2. check_divisions.py
       ↓
   Validates annotations
       ↓
3. organize_by_division.py
       ↓
   Creates division folders
       ↓
4. train_with_divisions.py (optional)
       ↓
   Trains division-specific models
```

## Quick Examples

**Full pipeline:**
```bash
bash run_division_pipeline.sh
```

**Just annotate type1:**
```bash
python3 add_divisions_to_reviewed.py --type type1_text
```

**Check quality:**
```bash
python3 check_divisions.py --all
```

**Organize with lower threshold:**
```bash
python3 organize_by_division.py --min-samples 5
```

**Train Cardiovascular specialist:**
```bash
python3 train_with_divisions.py \
    --train-data data/division_added/1/train.jsonl \
    --val-data data/division_added/1/validation.jsonl \
    --output-dir ../models/cardio_specialist
```
