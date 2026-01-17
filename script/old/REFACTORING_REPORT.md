# Code Duplication Analysis & Refactoring

## Analysis Results

### Before Refactoring

**Total Lines:** 1,456 lines across 7 scripts

**Duplicated Code Patterns:**

1. **Data Loading** (3x duplicated)
   - `load_data()` function in train_00, train_01, train_02
   - ~15 lines each = 45 lines total

2. **Model Loading** (3x duplicated)
   - `AutoModelForCausalLM.from_pretrained()` with BitsAndBytesConfig
   - ~10 lines each = 30 lines total

3. **LoRA Configuration** (3x duplicated)
   - `LoraConfig()` creation
   - ~8 lines each = 24 lines total

4. **Training Arguments** (3x duplicated)
   - `SFTConfig()` creation
   - ~15 lines each = 45 lines total

5. **Tokenizer Loading** (3x duplicated)
   - ~5 lines each = 15 lines total

**Total Duplicated:** ~160 lines

---

## After Refactoring

### Created: `training_utils.py` (215 lines)

**Shared Functions:**
- `load_jsonl_data()` - Load train/val data
- `load_tokenizer()` - Load tokenizer with standard settings
- `load_model_8bit()` - Load model with 8-bit quantization
- `create_lora_config()` - Create LoRA configuration
- `setup_model_with_lora()` - Complete model setup
- `create_training_args()` - Create training arguments
- `save_training_info()` - Save training metadata

### Refactored Scripts

**train/train_02_kor_med_test_refactored.py:**
- Before: 218 lines
- After: 106 lines
- **Reduction: 51%**

**Expected per script:**
- train_00: 266 → ~140 lines (47% reduction)
- train_01: 295 → ~150 lines (49% reduction)
- train_02: 218 → 106 lines (51% reduction)

---

## Benefits

1. ✅ **51% code reduction** in training scripts
2. ✅ **Single source of truth** for common operations
3. ✅ **Easier maintenance** - update once, apply everywhere
4. ✅ **Consistent behavior** across all training scripts
5. ✅ **Better testing** - test utilities once
6. ✅ **Cleaner code** - focus on script-specific logic

---

## File Structure

```
script/
├── training_config.py         # Model configs, hyperparameters
├── training_utils.py          # Shared training functions (NEW)
├── train/train_00_plain_text.py     # (to be refactored)
├── train/train_01_medical_dict.py   # (to be refactored)
├── train/train_02_kor_med_test.py   # Original (218 lines)
└── train/train_02_kor_med_test_refactored.py  # Refactored (106 lines)
```

---

## Next Steps

1. Replace train_02 with refactored version
2. Refactor train_00 and train_01 using training_utils
3. Update validation scripts to use shared functions
4. Remove old versions after testing

---

## Code Quality Improvements

### Before:
```python
# Duplicated in 3 scripts
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    cfg['path'],
    quantization_config=bnb_config,
    device_map="cuda:0",
    trust_remote_code=True,
    attn_implementation="sdpa"
)
model = prepare_model_for_kbit_training(model)
```

### After:
```python
# Single line, shared function
model = load_model_8bit(model_path, device="cuda:0")
```

---

Generated: 2025-12-20
