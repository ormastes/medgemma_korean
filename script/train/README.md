# MedGemma Korean - Training Scripts

This directory contains training scripts for Korean medical language model fine-tuning using progressive LoRA training.

## Quick Start

```bash
# Step 1: Plain text continued pretraining
python script/train/train_00_plain_text.py --model medgemma-4b --epochs 1

# Step 2: Mixed training (translation + MCQ)
python script/train/train_01_mixed.py --model medgemma-4b --epochs 1

# Step 3: MCQ reasoning training
python script/train/train_02_kor_med_test.py --model medgemma-4b --epochs 5
```

---

## Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROGRESSIVE LORA TRAINING                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  train_00: Plain Text (Korean)                                  │
│     ├── Input:  google/medgemma-*-it (base model)               │
│     ├── Output: model/00_trained/                               │
│     └── Creates: LoRA_0 (Korean language knowledge)             │
│            ↓                                                    │
│  train_01: Mixed (Translation + MCQ)                            │
│     ├── Input:  model/00_trained/ (LoRA_0)                      │
│     ├── Output: model/01_mixed/                                 │
│     └── Continues: LoRA_0 training (no new LoRA)                │
│            ↓                                                    │
│  train_02: MCQ Reasoning                                        │
│     ├── Input:  model/01_trained/ (LoRA_0 + LoRA_1)             │
│     ├── Output: model/02_trained/                               │
│     └── Creates: LoRA_2 (reasoning skills)                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
script/train/
├── README.md                    # This file
├── __init__.py                  # Package init
│
├── train_00_plain_text.py       # Phase 0: Korean continued pretraining
├── train_01_mixed.py            # Phase 1: Translation + MCQ training
├── train_02_kor_med_test.py     # Phase 2: MCQ with reasoning
│
├── training_config.py           # Model configs, LoRA settings, memory
├── training_utils.py            # Shared utilities (LoRA, data loading)
├── data_validation.py           # Data length validation
│
├── _train_text_format.py        # Prompt templates (MCQ, translation)
├── _validation.py               # Output format validation
├── _add_lora.py                 # LoRA adapter management
├── _callbacks.py                # Training callbacks
├── _mcq_evaluation.py           # MCQ evaluation during training
├── _logging.py                  # Logging utilities
├── _paths.py                    # Path management
│
├── test_train_pipeline.py       # Pipeline integration tests
└── test_add_lora.py             # LoRA addition tests
```

---

## Training Scripts

### train_00_plain_text.py - Korean Continued Pretraining

**Purpose:** Learn Korean language patterns from plain text (Wikipedia, Namu Wiki, C4)

**Input:**
- Base model: `google/medgemma-4b-it` or `google/medgemma-27b-text-it`
- Data: `data/02_refined/00_plain_text/train.jsonl`

**Output:** `model/00_trained/{model}/`

**Features:**
- Creates new LoRA_0 adapter
- Extends tokenizer embeddings with Korean tokens
- Uses gradient checkpointing for memory optimization

```bash
# Basic training
python script/train/train_00_plain_text.py --model medgemma-4b --epochs 1

# Limit samples for testing
python script/train/train_00_plain_text.py --model medgemma-4b --max-samples 1000

# Use specific GPU
python script/train/train_00_plain_text.py --model medgemma-4b --device cuda:1
```

---

### train_01_mixed.py - Translation + MCQ Combined

**Purpose:** Learn English-Korean translation and basic MCQ answering

**Input:**
- LoRA from train_00: `model/00_trained/{model}/`
- Translation data: `data/02_refined/01_english_korean/`
- MCQ data: `data/02_refined/02_kor_med_test/`

**Output:** `model/01_mixed/{model}/`

**Features:**
- Continues training existing LoRA (no new adapter)
- Bidirectional translation (en→ko, ko→en)
- MCQ training with format validation
- Configurable translation vs MCQ ratio
- Separate validation logs for MCQ and translation

```bash
# Basic training
python script/train/train_01_mixed.py --model medgemma-4b --epochs 1

# Limit data samples
python script/train/train_01_mixed.py --model medgemma-4b \
    --max-translation 10000 \
    --max-mcq 2000

# Adjust translation ratio (default: 0.7)
python script/train/train_01_mixed.py --model medgemma-4b \
    --translation-ratio 0.8

# Custom evaluation interval
python script/train/train_01_mixed.py --model medgemma-4b \
    --eval-interval 100 \
    --mcq-val-samples 10 \
    --trans-val-samples 5
```

---

### train_02_kor_med_test.py - MCQ with Chain-of-Thought Reasoning

**Purpose:** Learn structured reasoning for medical MCQs

**Input:**
- Previous LoRAs: `model/00_trained/` + `model/01_trained/`
- MCQ data: `data/02_refined/02_kor_med_test/`

**Output:** `model/02_trained/{model}/`

**Two Training Modes:**

1. **FULL MODE** (first half of training)
   - Uses detailed prompt with examples
   - Teaches structured reasoning format
   - Uses subset of data (--full-samples)
   - Runs until reasoning_score >= threshold

2. **NORMAL MODE** (second half)
   - Uses simple prompt
   - Standard MCQ training
   - Uses all samples
   - Starts after FULL mode completes

**Progressive LoRA:**
- Merges LoRA_0 and LoRA_1 into base model
- Adds new trainable LoRA_2
- Previous knowledge is frozen

```bash
# Basic training
python script/train/train_02_kor_med_test.py --model medgemma-4b --epochs 5

# Adjust mode switching
python script/train/train_02_kor_med_test.py --model medgemma-4b \
    --full-samples 500 \
    --reasoning-threshold 0.7

# Custom evaluation
python script/train/train_02_kor_med_test.py --model medgemma-4b \
    --eval-interval 50 \
    --eval-samples 10

# Skip data validation at startup
python script/train/train_02_kor_med_test.py --model medgemma-4b \
    --skip-validation
```

---

## Model Configuration

### training_config.py

```python
MODEL_CONFIGS = {
    "medgemma-4b": {
        "path": "google/medgemma-4b-it",
        "lora_r": 16,           # LoRA rank
        "lora_alpha": 32,       # LoRA alpha (2 * r)
        "use_rslora": True,     # Rank-stabilized LoRA
        "lr": 1e-4,             # Learning rate
        "batch": 2,             # Batch size
        "grad_accum": 16,       # Gradient accumulation
        "max_length": 512,      # Default max length
    },
    "medgemma-27b": {
        "path": "google/medgemma-27b-text-it",
        "lora_r": 16,
        "lora_alpha": 32,
        "use_rslora": True,
        "lr": 1e-4,
        "batch": 2,
        "grad_accum": 16,
        "max_length": 512,
    }
}
```

### LoRA Target Modules

Only attention projections (per research recommendations):
- `q_proj` - Query projection
- `k_proj` - Key projection
- `v_proj` - Value projection
- `o_proj` - Output projection

**Note:** MLP projections (gate/up/down) are excluded to reduce parameters from ~120M to ~9.2M.

### Memory Configurations

| Model | Gradient Checkpointing | Train Embeddings | Peak VRAM |
|-------|------------------------|------------------|-----------|
| medgemma-4b | Yes | Yes | ~42 GB |
| medgemma-27b | Yes | No | ~37 GB |

---

## Prompt Templates

### MCQ Training Template

```
<start_of_turn>user
question:
다음 중 당뇨병성 케톤산증(DKA) 초기 처치로 가장 적절한 것은?
A) 포도당 수액을 먼저 투여한다
...
answer_type:
select_A_E
remind:
Answer must be exactly one character: A, B, C, D, or E.
translate:
<end_of_turn>
<start_of_turn>model
Which initial management is most appropriate for DKA?
...
reasoning:
facts:
- Initial DKA management prioritizes isotonic fluid resuscitation.
candidates:
- A, B, C, D, E
criteria:
- Initial priority (hemodynamics)
...
evaluation:
- 평가기준: 초기 우선순위(수액), 안전성, 가이드라인 부합
- 점수표: A=0, B=0, C=3, D=0, E=0
- 근거요약: ...
answer:
C<end_of_turn>
```

### Translation Templates

**English → Korean:**
```
<start_of_turn>user
translate question:
{english}
translate:
<end_of_turn>
<start_of_turn>model
{korean}<end_of_turn>
```

**Korean → English:**
```
<start_of_turn>user
korean translate question:
{korean}
korean translate:
<end_of_turn>
<start_of_turn>model
{english}<end_of_turn>
```

---

## Output Validation

### MCQ Format Validation

The model output is validated for:
1. **Answer format:** `answer:\n` followed by `A/B/C/D/E`
2. **Required fields:** translate, reasoning, facts, candidates, criteria, analysis, evaluation, answer
3. **Korean evaluation fields:** 평가기준, 점수표, 근거요약

**Scoring:**
- Format score: 0.0 if no proper `answer:\n` pattern, 1.0 if valid
- Answer score: 1.0 if correct, 0.0 if wrong
- Total score: format_score × (1/3) + answer_score × (2/3)

### Translation Validation

Token overlap comparison:
- **Precision:** matched tokens / produced tokens
- **Recall:** matched tokens / expected tokens
- **F1 Score:** harmonic mean of precision and recall

---

## Data Requirements

| Script | Data Directory | Format |
|--------|----------------|--------|
| train_00 | `data/02_refined/00_plain_text/` | `{"text": "..."}` |
| train_01 | `data/02_refined/01_english_korean/` | `{"english": "...", "korean": "..."}` |
| train_01 | `data/02_refined/02_kor_med_test/` | `{"question": "...", "A": "...", ..., "answer": "A"}` |
| train_02 | `data/02_refined/02_kor_med_test/` | Same as above |

---

## Output Directory Structure

```
model/
├── 00_trained/
│   └── medgemma-4b/
│       ├── lora_adapter/           # LoRA_0 weights
│       │   ├── adapter_config.json
│       │   └── adapter_model.safetensors
│       ├── training_info.json      # Training metadata
│       └── tokenizer files         # Extended tokenizer
│
├── 01_mixed/
│   └── medgemma-4b/
│       ├── training/               # Checkpoints
│       └── final/                  # Final model
│
└── 02_trained/
    └── medgemma-4b/
        ├── full_mode/              # FULL mode checkpoints
        ├── normal_mode/            # NORMAL mode checkpoints
        └── final/                  # Final model with LoRA_2
```

---

## Common Arguments

All training scripts support:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `medgemma-4b` | Model to train |
| `--epochs` | `3` | Number of training epochs |
| `--max-samples` | `None` | Limit training samples |
| `--base-model` | `None` | Override base model path |
| `--output` | `None` | Override output directory |
| `--device` | `cuda:0` | Training device |

---

## GPU Requirements

| Script | Model | VRAM Required | Time per Epoch |
|--------|-------|---------------|----------------|
| train_00 | medgemma-4b | ~42 GB | ~2 hours |
| train_00 | medgemma-27b | ~37 GB | ~8 hours |
| train_01 | medgemma-4b | ~20 GB | ~1 hour |
| train_02 | medgemma-4b | ~25 GB | ~3 hours |

**Recommended:**
- RTX A6000 (48GB): All models
- RTX 4090 (24GB): medgemma-4b only (disable embeddings)
- TITAN RTX (24GB): medgemma-4b only (reduced batch)

---

## Logging and Monitoring

### Log Files

Training creates log files in `logs/`:
```
logs/
├── train_00_medgemma-4b_YYYYMMDD_HHMMSS.log
├── train_01_mixed/
│   ├── medgemma-4b_main_*.log        # Main training log
│   ├── medgemma-4b_validation_*.log  # Validation results
│   ├── medgemma-4b_mcq_*.log         # MCQ validation details
│   └── medgemma-4b_translation_*.log # Translation validation
└── train_02/
    └── train_02_debug.log
```

### Monitoring Commands

```bash
# Watch training progress
tail -f logs/train_01_mixed/medgemma-4b_main_*.log

# Check GPU memory
nvidia-smi -l 5

# Check validation scores
grep "MCQ:" logs/train_01_mixed/medgemma-4b_mcq_*.log | tail -20
```

---

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
--batch 1 --grad-accum 32

# Reduce max length
--max-length 512

# Disable embedding training (in training_config.py)
MEMORY_CONFIGS["medgemma-4b"]["train_embeddings"] = False
```

### LoRA Not Found

```
Error: LoRA not found in model/00_trained/medgemma-4b
```
→ Run train_00 first to create LoRA_0

### Template Hash Mismatch

```
Template changed, invalidating cache...
```
→ Normal behavior when templates are updated. Cache will be regenerated.

### Data Overflow Warning

```
⚠️ OVERFLOW DETECTED: 42 samples exceed max_length=512
```
→ These samples will be truncated. Increase `--max-length` if needed.

---

## Testing

```bash
# Run pipeline tests
python script/train/test_train_pipeline.py

# Run LoRA addition tests
python script/train/test_add_lora.py
```

---

## Version History

- **2024-12:** Initial training pipeline
- **2024-12:** Added progressive LoRA training
- **2024-12:** Added MCQ reasoning validation
- **2024-12:** Refactored utilities into separate modules
