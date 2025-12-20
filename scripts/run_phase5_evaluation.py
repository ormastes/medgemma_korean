#!/usr/bin/env python3
"""
Phase 5: Evaluate Korean Medical Capabilities

Evaluate the instruction-tuned model on Korean and English medical benchmarks.
"""

import sys
import os
sys.path.insert(0, '.')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
import json

# GPU setup
from config.gpu_utils import setup_gpu, print_memory_usage
device = setup_gpu("config/gpu_config.json")

print_memory_usage()

# =============================================================================
# Configuration
# =============================================================================
# Use instruction-tuned model
MODEL_DIR = "models/instruction_tuned"
BASE_MODEL_DIR = "models/final/korean_medgemma_expanded"
EVAL_DATA_DIR = "data/processed/kormedmcqa_eval"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Model: {MODEL_DIR}")
print(f"Base Model: {BASE_MODEL_DIR}")
print(f"Results: {RESULTS_DIR}")

# =============================================================================
# Load Model
# =============================================================================
print("\n" + "=" * 60)
print("Loading Model")
print("=" * 60)

# Use float32 to avoid NaN issues with half precision on this GPU
# Note: 4-bit quantization causes NaN in inference on this hardware
USE_FLOAT32 = True

# Check if it's a LoRA adapter or full model
adapter_config_path = os.path.join(MODEL_DIR, "adapter_config.json")
if os.path.exists(adapter_config_path):
    print("Loading base model with LoRA adapter...")
    from peft import PeftModel

    if USE_FLOAT32:
        # Load in float32 for stability
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_DIR,
            dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    else:
        # 4-bit quantization (may cause NaN on some GPUs)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_DIR,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )

    # Load adapter
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
else:
    print("Loading full model...")
    if USE_FLOAT32:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()

print(f"Model loaded!")
print_memory_usage()

# =============================================================================
# Helper Functions
# =============================================================================
def create_mcqa_prompt(example):
    """Create evaluation prompt for MCQA using ChatML format (same as training)"""
    question = example["question"]
    choices = []
    for letter in ['A', 'B', 'C', 'D', 'E']:
        if letter in example and example[letter]:
            choices.append(f"{letter}. {example[letter]}")

    formatted_choices = "\n".join(choices)

    # Use ChatML format to match training data
    prompt = f"""<|im_start|>system
당신은 한국어 의료 전문 AI 어시스턴트입니다. 정확하고 도움이 되는 의료 정보를 제공하세요.
<|im_end|>
<|im_start|>user
다음 의료 관련 질문에 답하세요. 정답 알파벳(A, B, C, D, E 중 하나)만 답하세요.

질문: {question}

선택지:
{formatted_choices}

정답은?
<|im_end|>
<|im_start|>assistant
"""
    return prompt


def extract_answer(response):
    """Extract answer letter from model response"""
    response = response.strip().upper()

    # Check for direct letter
    for letter in ['A', 'B', 'C', 'D', 'E']:
        if response.startswith(letter):
            return letter

    # Check for number mapping
    number_to_letter = {'1': 'A', '2': 'B', '3': 'C', '4': 'D', '5': 'E'}
    for num, letter in number_to_letter.items():
        if num in response[:5]:
            return letter

    # Check anywhere in response
    for letter in ['A', 'B', 'C', 'D', 'E']:
        if letter in response[:20]:
            return letter

    return None


def answer_idx_to_letter(idx):
    """Convert 1-indexed answer to letter"""
    mapping = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}
    return mapping.get(idx, 'A')


# =============================================================================
# Load Evaluation Data
# =============================================================================
print("\n" + "=" * 60)
print("Loading Evaluation Data")
print("=" * 60)

if os.path.exists(EVAL_DATA_DIR):
    eval_dataset = load_from_disk(EVAL_DATA_DIR)
    print(f"Loaded evaluation dataset: {len(eval_dataset)} examples")
else:
    # Load directly from HuggingFace
    print("Loading KorMedMCQA from HuggingFace...")
    eval_dataset = load_dataset("sean0042/KorMedMCQA", split="test")
    print(f"Loaded: {len(eval_dataset)} examples")

# Preview sample
sample = eval_dataset[0]
print("\nSample evaluation example:")
print(f"Question: {sample['question'][:100]}...")
print(f"Answer: {sample['answer']}")

# =============================================================================
# Run KorMedMCQA Evaluation
# =============================================================================
print("\n" + "=" * 60)
print("Running KorMedMCQA Evaluation")
print("=" * 60)

correct = 0
total = 0
results = []

# Evaluate all samples (or limit for testing)
max_samples = len(eval_dataset)  # Use all samples

for i, example in enumerate(tqdm(eval_dataset, total=max_samples, desc="Evaluating")):
    if i >= max_samples:
        break

    try:
        # Create prompt
        prompt = create_mcqa_prompt(example)

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode response (only new tokens)
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # Extract predicted answer
        predicted = extract_answer(response)

        # Get correct answer
        correct_answer = answer_idx_to_letter(example["answer"])

        # Check correctness
        is_correct = predicted == correct_answer
        if is_correct:
            correct += 1
        total += 1

        # Save result
        results.append({
            "question": example["question"],
            "predicted": predicted,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "response": response,
        })

        # Clear GPU cache periodically
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"\nError at sample {i}: {e}")
        # Save result as error
        results.append({
            "question": example.get("question", "Unknown"),
            "predicted": None,
            "correct_answer": answer_idx_to_letter(example.get("answer", 1)),
            "is_correct": False,
            "response": f"ERROR: {str(e)}",
        })
        total += 1

    # Save intermediate results every 100 samples
    if (i + 1) % 100 == 0:
        intermediate_results = {
            "model": MODEL_DIR,
            "base_model": BASE_MODEL_DIR,
            "benchmark": "KorMedMCQA",
            "accuracy": correct / total * 100 if total > 0 else 0,
            "correct": correct,
            "total": total,
            "results": results,
            "status": "in_progress",
            "last_sample": i,
        }
        with open(f"{RESULTS_DIR}/korean_eval_results_checkpoint.json", "w", encoding="utf-8") as f:
            json.dump(intermediate_results, f, ensure_ascii=False, indent=2)
        print(f"\n[Checkpoint saved at sample {i+1}]")

# Calculate accuracy
accuracy = correct / total * 100

print(f"\n" + "=" * 60)
print(f"KorMedMCQA Results")
print(f"=" * 60)
print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

# Show some examples
print("\nSample predictions:")
print("-" * 60)
for i in range(min(5, len(results))):
    r = results[i]
    status = "✓" if r["is_correct"] else "✗"
    print(f"\n{status} Q: {r['question'][:80]}...")
    print(f"   Predicted: {r['predicted']}, Correct: {r['correct_answer']}")
    print(f"   Response: {r['response'][:50]}")

# =============================================================================
# Qualitative Evaluation
# =============================================================================
print("\n" + "=" * 60)
print("Qualitative Evaluation (Open-ended Questions)")
print("=" * 60)

test_questions = [
    "고혈압의 주요 증상과 위험 요인은 무엇인가요?",
    "당뇨병 환자가 일상에서 주의해야 할 점은 무엇인가요?",
    "감기와 독감의 차이점을 설명해주세요.",
    "두통이 자주 발생할 때 어떻게 대처해야 하나요?",
]

qualitative_results = []

for question in test_questions:
    # Use ChatML format to match training data
    prompt = f"""<|im_start|>system
당신은 한국어 의료 전문 AI 어시스턴트입니다.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    print(f"\nQ: {question}")
    print(f"A: {response[:500]}")
    print("-" * 40)

    qualitative_results.append({
        "question": question,
        "response": response,
    })

# =============================================================================
# Save Results
# =============================================================================
print("\n" + "=" * 60)
print("Saving Results")
print("=" * 60)

eval_results = {
    "model": MODEL_DIR,
    "base_model": BASE_MODEL_DIR,
    "benchmark": "KorMedMCQA",
    "accuracy": accuracy,
    "correct": correct,
    "total": total,
    "results": results,
    "qualitative_results": qualitative_results,
}

results_path = f"{RESULTS_DIR}/korean_eval_results.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(eval_results, f, ensure_ascii=False, indent=2)

print(f"Results saved to {results_path}")

print("\n" + "=" * 60)
print("Phase 5.1: Korean Medical Evaluation Complete!")
print("=" * 60)
print(f"\nKorMedMCQA Accuracy: {accuracy:.2f}%")
print(f"\nResults saved to: {results_path}")
print("\nNext steps:")
print("  Run Phase 6 deployment")
