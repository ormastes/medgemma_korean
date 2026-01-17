#!/bin/bash
# =============================================================================
# Check Max Config Settings
# =============================================================================
# Validates max length configurations for all training stages:
# 1. Train 00 - Plain text max length
# 2. Train 01 - Medical dictionary max length
# 3. Create dummy model (01_another_lora_added_dummy) for train_02 check
# 4. Train 02 - FULL and NORMAL prompt max lengths
#
# Also reports GPU memory usage for each model load.
#
# Usage:
#   ./check_max_config.sh --model medgemma-4b
#   ./check_max_config.sh --model medgemma-27b --device cuda:1
# =============================================================================

set -e

# =============================================================================
# Default values
# =============================================================================
MODEL="medgemma-4b"
DEVICE="cuda:0"

# =============================================================================
# Parse arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Check max length configurations for all training stages."
            echo "Also reports GPU memory usage for each model."
            echo ""
            echo "Options:"
            echo "  --model MODEL    Model name (medgemma-4b or medgemma-27b)"
            echo "  --device DEVICE  CUDA device (default: cuda:0)"
            echo "  -h, --help       Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Setup paths
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

RAW_LORA_DIR="$BASE_DIR/model/raw_lora_added/$MODEL"
DUMMY_DIR="$BASE_DIR/model/01_another_lora_added_dummy/$MODEL"
MEM_FILE="/tmp/check_max_config_mem_$$.json"

# Max lengths from config
# train_02: FULL prompt max=633, NORMAL max=485, + ~300 response = ~933 max
if [[ "$MODEL" == "medgemma-4b" ]]; then
    MAX_LEN_00=512
    MAX_LEN_01=256
    MAX_LEN_02=1024    # FULL: 633+300=933, fits in 1024
else
    MAX_LEN_00=512
    MAX_LEN_01=256
    MAX_LEN_02=1024    # Memory optimized for 27b
fi

echo "============================================================"
echo "CHECK MAX CONFIG SETTINGS"
echo "============================================================"
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Max lengths: 00=$MAX_LEN_00, 01=$MAX_LEN_01, 02=$MAX_LEN_02"
echo "============================================================"

cd "$SCRIPT_DIR"

# Initialize memory file
echo "{}" > "$MEM_FILE"

# =============================================================================
# Check prerequisites
# =============================================================================
echo ""
echo "============================================================"
echo "Checking Prerequisites"
echo "============================================================"

if [[ ! -d "$RAW_LORA_DIR" ]]; then
    echo "❌ raw_lora_added not found: $RAW_LORA_DIR"
    echo "   Run init_lora_on_raw.py first:"
    echo "   python init_lora_on_raw.py --model $MODEL --device $DEVICE"
    exit 1
fi
echo "✓ raw_lora_added exists: $RAW_LORA_DIR"

# =============================================================================
# Step 1: Check Train 00 max length + Memory for base tokenizer
# =============================================================================
echo ""
echo "============================================================"
echo "Step 1: Check Train 00 Max Length ($MAX_LEN_00)"
echo "============================================================"

python3 << EOF
import sys
sys.path.insert(0, '.')
from data_validation import check_data_lengths
from training_config import MODEL_CONFIGS
from transformers import AutoTokenizer
import torch
import json

model = '$MODEL'
max_len = $MAX_LEN_00
mem_file = '$MEM_FILE'

cfg = MODEL_CONFIGS[model]
tokenizer = AutoTokenizer.from_pretrained(cfg['path'], trust_remote_code=True)

# Load sample data
data_file = '../data/02_refined/00_plain_text/train.jsonl'
samples = []
try:
    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            samples.append(json.loads(line))
except FileNotFoundError:
    print(f'⚠️ Data file not found: {data_file}')
    sys.exit(0)

result = check_data_lengths(samples, tokenizer, max_len)
print(f'  Samples checked: {result["total_samples"]}')
print(f'  Token range: {result["min_tokens"]} - {result["max_tokens"]}')
print(f'  Avg tokens: {result["avg_tokens"]:.0f}')
if result['overflow_count'] > 0:
    print(f'  ⚠️ Overflow: {result["overflow_count"]} samples exceed max_length')
else:
    print(f'  ✓ All samples fit within max_length')
EOF

# =============================================================================
# Step 2: Check Train 01 max length
# =============================================================================
echo ""
echo "============================================================"
echo "Step 2: Check Train 01 Max Length ($MAX_LEN_01)"
echo "============================================================"

python3 << EOF
import sys
sys.path.insert(0, '.')
from data_validation import check_data_lengths
from training_config import MODEL_CONFIGS
from transformers import AutoTokenizer
import json

model = '$MODEL'
max_len = $MAX_LEN_01

cfg = MODEL_CONFIGS[model]
tokenizer = AutoTokenizer.from_pretrained(cfg['path'], trust_remote_code=True)

# Load sample data
data_file = '../data/02_refined/01_medical_dict.json'
try:
    with open(data_file, 'r') as f:
        data = json.load(f)
    samples = data[:100] if isinstance(data, list) else []
except FileNotFoundError:
    print(f'⚠️ Data file not found: {data_file}')
    sys.exit(0)

# Convert to expected format
formatted_samples = []
for s in samples:
    text = f"{s.get('term', '')}: {s.get('definition', '')}"
    formatted_samples.append({'text': text})

result = check_data_lengths(formatted_samples, tokenizer, max_len)
print(f'  Samples checked: {result["total_samples"]}')
print(f'  Token range: {result["min_tokens"]} - {result["max_tokens"]}')
print(f'  Avg tokens: {result["avg_tokens"]:.0f}')
if result['overflow_count'] > 0:
    print(f'  ⚠️ Overflow: {result["overflow_count"]} samples exceed max_length')
else:
    print(f'  ✓ All samples fit within max_length')
EOF

# =============================================================================
# Step 3: Create dummy model + Measure memory for LoRA model load
# =============================================================================
echo ""
echo "============================================================"
echo "Step 3: Create Dummy Model + Measure Memory"
echo "============================================================"

python3 << EOF
import sys
sys.path.insert(0, '.')
import torch
import json
import os
from pathlib import Path

model = '$MODEL'
device = '$DEVICE'
raw_lora_dir = '$RAW_LORA_DIR'
dummy_dir = '$DUMMY_DIR'
mem_file = '$MEM_FILE'

# Check if dummy already exists
if os.path.exists(dummy_dir):
    print(f'Dummy model already exists: {dummy_dir}')
    print('Loading to measure memory...')
else:
    print(f'Creating dummy model at: {dummy_dir}')

# Clear GPU cache and reset stats
device_id = int(device.split(':')[1]) if ':' in device else 0
if torch.cuda.is_available():
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_id)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(raw_lora_dir, trust_remote_code=True)

# Load model
print('Loading base model (8-bit)...')
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=False
)

try:
    # Try loading as PEFT model
    base_model = AutoModelForCausalLM.from_pretrained(
        raw_lora_dir,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    model_loaded = PeftModel.from_pretrained(base_model, raw_lora_dir)
    print('✓ Loaded as PEFT model')
except Exception as e:
    print(f'Loading as base model: {e}')
    model_loaded = AutoModelForCausalLM.from_pretrained(
        raw_lora_dir,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

# Get memory after loading raw_lora model
mem_raw_lora_gb = 0
if torch.cuda.is_available():
    mem_raw_lora_gb = torch.cuda.max_memory_allocated(device_id) / 1024**3
print(f'  Memory (raw_lora_added): {mem_raw_lora_gb:.2f} GB')

# Create dummy if needed
if not os.path.exists(dummy_dir):
    from training_config import LORA_TARGET_MODULES, TRAINING_DEFAULTS

    # Add new adapter
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=TRAINING_DEFAULTS['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )

    if hasattr(model_loaded, 'add_adapter'):
        model_loaded.add_adapter('mcq_reasoning_dummy', lora_config)
        model_loaded.set_adapter('mcq_reasoning_dummy')
    else:
        model_loaded = get_peft_model(model_loaded, lora_config)

    # Save
    Path(dummy_dir).mkdir(parents=True, exist_ok=True)
    model_loaded.save_pretrained(dummy_dir)
    tokenizer.save_pretrained(dummy_dir)
    print(f'✓ Saved dummy model to: {dummy_dir}')

# Get peak memory after adding adapter
mem_with_adapter_gb = 0
if torch.cuda.is_available():
    mem_with_adapter_gb = torch.cuda.max_memory_allocated(device_id) / 1024**3
print(f'  Memory (with 2nd LoRA): {mem_with_adapter_gb:.2f} GB')

# Save memory info
mem_data = {
    'raw_lora_added_gb': round(mem_raw_lora_gb, 2),
    'with_second_lora_gb': round(mem_with_adapter_gb, 2)
}
with open(mem_file, 'w') as f:
    json.dump(mem_data, f)

# Clean up
del model_loaded
if torch.cuda.is_available():
    torch.cuda.empty_cache()
EOF

# =============================================================================
# Step 4: Check Train 02 max length (FULL and NORMAL prompts) + Memory
# =============================================================================
echo ""
echo "============================================================"
echo "Step 4: Check Train 02 Max Length ($MAX_LEN_02)"
echo "============================================================"

python3 << EOF
import sys
sys.path.insert(0, '.')
from training_config import MODEL_CONFIGS
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import json

model = '$MODEL'
max_len = $MAX_LEN_02
dummy_dir = '$DUMMY_DIR'
device = '$DEVICE'
mem_file = '$MEM_FILE'

device_id = int(device.split(':')[1]) if ':' in device else 0

# Clear GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_id)

# Load tokenizer from dummy model
print(f'Loading tokenizer from: {dummy_dir}')
tokenizer = AutoTokenizer.from_pretrained(dummy_dir, trust_remote_code=True)

# Load model to measure memory
print(f'Loading model from dummy dir...')
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=False
)

try:
    base_model = AutoModelForCausalLM.from_pretrained(
        dummy_dir,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    model_loaded = PeftModel.from_pretrained(base_model, dummy_dir)
except Exception as e:
    print(f'Note: {e}')
    model_loaded = AutoModelForCausalLM.from_pretrained(
        dummy_dir,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

# Get memory for train_02 model
mem_train02_gb = 0
if torch.cuda.is_available():
    mem_train02_gb = torch.cuda.max_memory_allocated(device_id) / 1024**3
print(f'  Memory (train_02 model): {mem_train02_gb:.2f} GB')

# Update memory file
with open(mem_file, 'r') as f:
    mem_data = json.load(f)
mem_data['train02_model_gb'] = round(mem_train02_gb, 2)
with open(mem_file, 'w') as f:
    json.dump(mem_data, f)

# Clean up model
del model_loaded
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Load sample data
data_file = '../data/02_refined/02_kor_med_test/train.jsonl'
samples = []
try:
    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 50:
                break
            samples.append(json.loads(line))
except FileNotFoundError:
    print(f'⚠️ Data file not found: {data_file}')
    sys.exit(0)

# FULL prompt template (with example)
FULL_TEMPLATE = '''당신은 한국어 의료 전문 AI입니다. 객관식 문제에 답할 때 다음 형식을 따르세요:

1. 먼저 <R> 태그 안에 추론 과정을 작성하세요
2. 그 다음 정답 알파벳(A, B, C, D, E 중 하나)만 답하세요

예시:
질문: 심전도에서 ST 분절 상승이 관찰되는 경우는?
A) 안정형 협심증
B) 급성 심근경색
C) 심방세동
D) 동서맥
E) 심실조기수축

<R>ST 분절 상승은 심근 손상의 급성기 소견입니다. 안정형 협심증은 ST 하강, 심방세동은 P파 소실, 동서맥과 심실조기수축은 ST 변화 없음. 급성 심근경색에서 ST 상승이 특징적입니다.</R>
B

이제 다음 문제에 답하세요:

질문: {question}
{choices}
'''

# NORMAL prompt template (simple)
NORMAL_TEMPLATE = '''당신은 한국어 의료 전문 AI입니다.
먼저 <R> 태그 안에 추론하고, 정답 알파벳(A, B, C, D, E)만 답하세요.

질문: {question}
{choices}
'''

print('')
print('--- FULL Prompt Mode ---')
full_tokens = []
for sample in samples:
    question = sample.get('question', '')
    choices = '\n'.join([f"{k}) {v}" for k, v in sample.items() if k in ['A', 'B', 'C', 'D', 'E']])
    prompt = FULL_TEMPLATE.format(question=question, choices=choices)
    tokens = tokenizer.encode(prompt)
    full_tokens.append(len(tokens))

print(f'  Samples checked: {len(full_tokens)}')
print(f'  Token range: {min(full_tokens)} - {max(full_tokens)}')
print(f'  Avg tokens: {sum(full_tokens)/len(full_tokens):.0f}')
overflow = sum(1 for t in full_tokens if t > max_len)
if overflow > 0:
    print(f'  ⚠️ Overflow: {overflow} samples exceed max_length')
else:
    print(f'  ✓ All samples fit within max_length')

print('')
print('--- NORMAL Prompt Mode ---')
normal_tokens = []
for sample in samples:
    question = sample.get('question', '')
    choices = '\n'.join([f"{k}) {v}" for k, v in sample.items() if k in ['A', 'B', 'C', 'D', 'E']])
    prompt = NORMAL_TEMPLATE.format(question=question, choices=choices)
    tokens = tokenizer.encode(prompt)
    normal_tokens.append(len(tokens))

print(f'  Samples checked: {len(normal_tokens)}')
print(f'  Token range: {min(normal_tokens)} - {max(normal_tokens)}')
print(f'  Avg tokens: {sum(normal_tokens)/len(normal_tokens):.0f}')
overflow = sum(1 for t in normal_tokens if t > max_len)
if overflow > 0:
    print(f'  ⚠️ Overflow: {overflow} samples exceed max_length')
else:
    print(f'  ✓ All samples fit within max_length')

print('')
print('--- Token Budget Summary ---')
print(f'  Max length setting: {max_len}')
print(f'  FULL prompt overhead: ~{int(sum(full_tokens)/len(full_tokens) - sum(normal_tokens)/len(normal_tokens))} tokens')
print(f'  Remaining for response: ~{max_len - max(full_tokens)} tokens (FULL), ~{max_len - max(normal_tokens)} tokens (NORMAL)')
EOF

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo ""
echo "Max Length Settings:"
echo "  Train 00 (Plain Text):      $MAX_LEN_00 tokens"
echo "  Train 01 (Medical Dict):    $MAX_LEN_01 tokens"
echo "  Train 02 (MCQ Reasoning):   $MAX_LEN_02 tokens"
echo ""
echo "Models:"
echo "  raw_lora_added:             $RAW_LORA_DIR"
echo "  dummy (for testing):        $DUMMY_DIR"

# Print memory usage
echo ""
echo "============================================================"
echo "GPU MEMORY USAGE"
echo "============================================================"

python3 << EOF
import json

mem_file = '$MEM_FILE'
model = '$MODEL'

try:
    with open(mem_file, 'r') as f:
        mem_data = json.load(f)

    print('')
    print(f'Model: {model}')
    print('-' * 40)

    if 'raw_lora_added_gb' in mem_data:
        print(f'  raw_lora_added (8-bit):    {mem_data["raw_lora_added_gb"]:.2f} GB')

    if 'with_second_lora_gb' in mem_data:
        print(f'  + second LoRA adapter:     {mem_data["with_second_lora_gb"]:.2f} GB')

    if 'train02_model_gb' in mem_data:
        print(f'  train_02 model (8-bit):    {mem_data["train02_model_gb"]:.2f} GB')

    print('')

    # Estimate training memory (usually 1.5-2x inference)
    max_mem = max(mem_data.values()) if mem_data else 0
    print(f'Estimated training memory:')
    print(f'  Inference:   ~{max_mem:.1f} GB')
    print(f'  Training:    ~{max_mem * 1.5:.1f} - {max_mem * 2:.1f} GB')
    print(f'  With grad:   ~{max_mem * 2:.1f} - {max_mem * 2.5:.1f} GB')

except Exception as e:
    print(f'Could not read memory data: {e}')
EOF

# Cleanup
rm -f "$MEM_FILE"

echo ""
echo "============================================================"
echo "✓ Configuration check complete!"
echo "============================================================"
echo ""
echo "To run training pipeline:"
echo "  ./run_full_pipeline.sh --model $MODEL --device $DEVICE"
