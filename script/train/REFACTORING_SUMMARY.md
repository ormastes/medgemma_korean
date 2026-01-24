# Training Scripts Refactoring Summary

## New Shared Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `_logging.py` | Unified logging | `TrainingLogger`, `clear_gpu_memory()`, `check_gpu_status()` |
| `_paths.py` | Path resolution | `BASE_DIR`, `get_model_output_dir()`, `find_lora_path()` |
| `_callbacks.py` | Training callbacks | `LossLoggingCallback`, `MCQValidationCallback`, `get_terminators()` |
| `_mcq_evaluation.py` | MCQ evaluation | `MCQEvaluator`, `evaluate_mcq_batch()`, `extract_answer_letter()` |
| `_add_lora.py` | LoRA loading | `load_for_train_00()`, `load_for_train_01()`, `load_for_train_02()` |

## Duplicated Code Removed

### 1. Logging (60+ lines each script → shared)

**Before (duplicated in each script):**
```python
def log(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] [{level}] {msg}"
    print(log_msg)
    # ... file writing ...

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

**After:**
```python
from _logging import TrainingLogger, clear_gpu_memory

logger = TrainingLogger("train_00", log_dir)
logger.log("Starting training...")
```

### 2. Path Resolution (30+ lines each → shared)

**Before:**
```python
BASE_DIR = Path(__file__).parent.parent.parent
RAW_LORA_DIR = BASE_DIR / "model" / "raw_lora_added"
# ... manual path checking ...
```

**After:**
```python
from _paths import BASE_DIR, get_model_output_dir, find_lora_path

output_dir = get_model_output_dir("00", "medgemma-4b")
lora_path = find_lora_path(MODEL_DIR / "00_trained", "medgemma-4b")
```

### 3. Model Loading (50+ lines each → shared)

**Before (train_00, train_01, train_02 each had this):**
```python
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(...)
base_model = prepare_model_for_kbit_training(...)
if tokenizer_vocab_size > base_vocab_size:
    base_model.resize_token_embeddings(...)
model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=True)
```

**After:**
```python
from _add_lora import load_for_train_00, load_for_train_01, load_for_train_02

model, tokenizer = load_for_train_00(model_path, device="cuda:0", model_name="medgemma-4b")
```

### 4. Terminators Setup (5 lines × 3 scripts → shared)

**Before:**
```python
end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
terminators = [tokenizer.eos_token_id]
if end_of_turn_id != tokenizer.unk_token_id:
    terminators.append(end_of_turn_id)
```

**After:**
```python
from _callbacks import get_terminators

terminators = get_terminators(tokenizer)
```

### 5. MCQ Evaluation (100+ lines each → shared)

**Before:**
```python
def evaluate_kormedmcqa(model, tokenizer, test_data, device, ...):
    model.eval()
    correct = 0
    # ... 50+ lines of evaluation logic ...
```

**After:**
```python
from _mcq_evaluation import MCQEvaluator

evaluator = MCQEvaluator(tokenizer, test_data, prompt_template, device)
result = evaluator.evaluate(model)
quick_result = evaluator.quick_evaluate(model, num_samples=10)
```

### 6. Callbacks (30+ lines duplicated → shared)

**Before:**
```python
class LossLoggingCallback(TrainerCallback):
    # Duplicated in train_00 and train_01
```

**After:**
```python
from _callbacks import LossLoggingCallback

callback = LossLoggingCallback(log_fn=logger.log_val)
```

## Migration Guide

### train_00_plain_text.py

```python
# Old imports
from datetime import datetime
import gc

# New imports
from _logging import TrainingLogger, clear_gpu_memory, check_gpu_status
from _paths import BASE_DIR, get_model_output_dir, get_raw_lora_model_path
from _callbacks import LossLoggingCallback, get_terminators
from _mcq_evaluation import MCQEvaluator
from _add_lora import load_for_train_00

# Old: ~200 lines of utility functions
# New: ~20 lines of imports

# Usage
logger = TrainingLogger("train_00", TRAINING_DIR)
model, tokenizer = load_for_train_00(model_path, device=args.device, model_name=args.model)
evaluator = MCQEvaluator(tokenizer, test_data, MCQ_VALIDATE_TEMPLATE, args.device)
```

### train_01_mixed.py

```python
from _logging import TrainingLogger, clear_gpu_memory
from _paths import get_model_output_dir, get_previous_phase_lora
from _callbacks import LossLoggingCallback, MCQValidationCallback
from _add_lora import load_for_train_01

model, tokenizer = load_for_train_01(lora_path, device=args.device, model_name=args.model)
```

### train_02_kor_med_test.py

```python
from _logging import TrainingLogger, clear_gpu_memory
from _paths import get_model_output_dir, get_lora_chain
from _mcq_evaluation import MCQEvaluator
from _add_lora import load_for_train_02

model, tokenizer = load_for_train_02(lora_path, device=args.device, model_name=args.model)
evaluator = MCQEvaluator(tokenizer, validation_data, VALIDATION_TEMPLATE, args.device)
```

## Lines of Code Comparison

| Script | Before | After (est.) | Reduction |
|--------|--------|--------------|-----------|
| train_00 | 662 | ~350 | ~47% |
| train_01 | 618 | ~300 | ~51% |
| train_02 | 975 | ~500 | ~49% |
| **Shared modules** | 0 | ~800 | - |
| **Total** | 2255 | ~1950 | ~14% |

The total LOC is slightly lower, but more importantly:
- Code is DRY (Don't Repeat Yourself)
- Easier to maintain and update
- Consistent behavior across all scripts
- Easier to test shared components

## Shared Module Locations

```
script/train/
├── __init__.py              # Package exports
├── _logging.py              # NEW: Logging utilities
├── _paths.py                # NEW: Path utilities
├── _callbacks.py            # NEW: Training callbacks
├── _mcq_evaluation.py       # NEW: MCQ evaluation
├── _add_lora.py             # UPDATED: LoRA loading
├── _validation.py           # EXISTING: Output validation
├── _train_text_format.py    # EXISTING: Prompt templates
├── training_config.py       # EXISTING: Model configs
├── training_utils.py        # EXISTING: Training utilities
├── train_00_plain_text.py   # TO UPDATE
├── train_01_mixed.py        # TO UPDATE
└── train_02_kor_med_test.py # TO UPDATE
```

## Testing

```bash
# Test LoRA loading
python script/train/test_add_lora.py --device cuda:1

# Test shared modules
python -c "from script.train import *; print('Imports OK')"
```
