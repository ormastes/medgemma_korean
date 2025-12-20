#!/usr/bin/env python3
"""
Phase 6: Deployment Preparation

Prepare the instruction-tuned model for deployment:
1. AWQ Quantization for efficient inference
2. Create deployment scripts
"""

import sys
import os
sys.path.insert(0, '.')

import torch
import json

# GPU setup
from config.gpu_utils import setup_gpu, print_memory_usage, clear_memory
device = setup_gpu("config/gpu_config.json")

print_memory_usage()

# =============================================================================
# Configuration
# =============================================================================
# Use instruction-tuned model (LoRA adapter)
MODEL_DIR = "models/instruction_tuned"
BASE_MODEL_DIR = "models/final/korean_medgemma_expanded"
OUTPUT_DIR = "models/korean_medgemma_awq"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Input model: {MODEL_DIR}")
print(f"Base model: {BASE_MODEL_DIR}")
print(f"Output dir: {OUTPUT_DIR}")

# AWQ configuration
AWQ_CONFIG = {
    "w_bit": 4,
    "q_group_size": 128,
    "zero_point": True,
    "version": "GEMM",
}

print("\nAWQ Configuration:")
for key, value in AWQ_CONFIG.items():
    print(f"  {key}: {value}")

# =============================================================================
# Step 1: Merge LoRA with Base Model
# =============================================================================
print("\n" + "=" * 60)
print("Step 1: Merging LoRA Adapter with Base Model")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Check if we have a LoRA adapter
adapter_config_path = os.path.join(MODEL_DIR, "adapter_config.json")
merged_model_dir = "models/instruction_tuned_merged"

if os.path.exists(adapter_config_path):
    print("Found LoRA adapter, merging with base model...")

    # Load base model in float16 for merging
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)

    # Merge and unload
    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    # Save merged model
    os.makedirs(merged_model_dir, exist_ok=True)
    print(f"Saving merged model to {merged_model_dir}...")
    model.save_pretrained(merged_model_dir)

    # Load and save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(merged_model_dir)

    print("Merged model saved!")

    # Clear memory
    del model, base_model
    clear_memory()

    MODEL_FOR_QUANTIZATION = merged_model_dir
else:
    print("No LoRA adapter found, using model directly.")
    MODEL_FOR_QUANTIZATION = MODEL_DIR

print_memory_usage()

# =============================================================================
# Step 2: Check if AWQ is available
# =============================================================================
print("\n" + "=" * 60)
print("Step 2: Checking AWQ Availability")
print("=" * 60)

try:
    from awq import AutoAWQForCausalLM
    AWQ_AVAILABLE = True
    print("AWQ library is available!")
except ImportError:
    AWQ_AVAILABLE = False
    print("AWQ library not installed.")
    print("Install with: pip install autoawq")
    print("\nSkipping AWQ quantization, creating deployment scripts only...")

# =============================================================================
# Step 3: AWQ Quantization (if available)
# =============================================================================
if AWQ_AVAILABLE:
    print("\n" + "=" * 60)
    print("Step 3: AWQ Quantization")
    print("=" * 60)

    # Calibration texts
    calibration_texts = [
        "고혈압은 혈압이 정상보다 높은 상태를 말합니다. 수축기 혈압이 140mmHg 이상이거나 이완기 혈압이 90mmHg 이상인 경우를 고혈압으로 정의합니다.",
        "당뇨병은 인슐린 분비나 작용에 문제가 생겨 혈당이 높아지는 대사 질환입니다. 제1형 당뇨병과 제2형 당뇨병으로 분류됩니다.",
        "폐렴은 폐에 염증이 생기는 질환으로, 세균, 바이러스, 곰팡이 등이 원인이 될 수 있습니다. 기침, 발열, 호흡곤란이 주요 증상입니다.",
        "심근경색은 심장 근육에 혈액 공급이 차단되어 발생하는 응급 상황입니다. 가슴 통증, 호흡곤란, 식은땀이 주요 증상입니다.",
        "뇌졸중은 뇌혈관이 막히거나 터져서 발생하는 질환입니다. 갑작스러운 마비, 언어 장애, 두통이 나타날 수 있습니다.",
        "Hypertension is defined as blood pressure consistently above 140/90 mmHg. It is a major risk factor for heart disease, stroke, and kidney disease.",
        "Diabetes mellitus is a metabolic disorder characterized by elevated blood glucose levels. Type 2 diabetes is the most common form.",
        "Pneumonia is an infection that inflames the air sacs in one or both lungs. Symptoms include cough, fever, and difficulty breathing.",
        "Myocardial infarction occurs when blood flow to the heart muscle is blocked. Prompt treatment is essential to minimize heart damage.",
        "Stroke is a medical emergency caused by disrupted blood supply to the brain. Symptoms include sudden weakness and speech difficulties.",
    ]

    print(f"Prepared {len(calibration_texts)} calibration texts")

    # Load model for quantization
    print(f"\nLoading model from {MODEL_FOR_QUANTIZATION}...")

    model = AutoAWQForCausalLM.from_pretrained(
        MODEL_FOR_QUANTIZATION,
        trust_remote_code=True,
        safetensors=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_FOR_QUANTIZATION)

    print("Model loaded!")
    print_memory_usage()

    # Quantize
    print("\nApplying AWQ quantization...")
    print("This may take 10-30 minutes depending on model size.")

    quant_config = {
        "zero_point": AWQ_CONFIG["zero_point"],
        "q_group_size": AWQ_CONFIG["q_group_size"],
        "w_bit": AWQ_CONFIG["w_bit"],
        "version": AWQ_CONFIG["version"],
    }

    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calibration_texts,
    )

    print("\nQuantization complete!")
    print_memory_usage()

    # Test quantized model
    print("\nTesting quantized model...")

    test_prompt = """<|im_start|>system
당신은 의료 AI 어시스턴트입니다.
<|im_end|>
<|im_start|>user
고혈압의 증상은 무엇인가요?
<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nTest response: {response[len(test_prompt):][:200]}...")

    # Save quantized model
    print(f"\nSaving quantized model to {OUTPUT_DIR}...")
    model.save_quantized(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Quantized model saved!")

    # Save quantization info
    quant_info = {
        "source_model": MODEL_FOR_QUANTIZATION,
        "quantization_method": "AWQ",
        "config": AWQ_CONFIG,
        "calibration_samples": len(calibration_texts),
    }

    with open(f"{OUTPUT_DIR}/quantization_info.json", "w") as f:
        json.dump(quant_info, f, indent=2)

    # Check model sizes
    def get_folder_size(folder):
        total = 0
        for path, dirs, files in os.walk(folder):
            for f in files:
                fp = os.path.join(path, f)
                total += os.path.getsize(fp)
        return total

    original_size = get_folder_size(MODEL_FOR_QUANTIZATION) / (1024**3)
    quantized_size = get_folder_size(OUTPUT_DIR) / (1024**3)

    print(f"\nModel size comparison:")
    print(f"  Original: {original_size:.2f} GB")
    print(f"  Quantized: {quantized_size:.2f} GB")
    print(f"  Compression: {original_size / quantized_size:.1f}x")

    DEPLOY_MODEL = OUTPUT_DIR
else:
    DEPLOY_MODEL = MODEL_FOR_QUANTIZATION if os.path.exists(merged_model_dir) else MODEL_DIR

# =============================================================================
# Step 4: Create Deployment Scripts
# =============================================================================
print("\n" + "=" * 60)
print("Step 4: Creating Deployment Scripts")
print("=" * 60)

# vLLM deployment script
vllm_script = f'''#!/bin/bash
# Korean MedGemma vLLM Deployment Script

MODEL_PATH="{os.path.abspath(DEPLOY_MODEL)}"
HOST="0.0.0.0"
PORT="8000"

echo "Starting Korean MedGemma vLLM Server..."
echo "Model: $MODEL_PATH"
echo "Server: http://$HOST:$PORT"

python -m vllm.entrypoints.openai.api_server \\
    --model $MODEL_PATH \\
    --host $HOST \\
    --port $PORT \\
    --max-model-len 4096 \\
    --gpu-memory-utilization 0.9 \\
    --dtype half \\
    {"--quantization awq" if AWQ_AVAILABLE else ""} \\
    --trust-remote-code
'''

with open("scripts/deploy_vllm.sh", "w") as f:
    f.write(vllm_script)
os.chmod("scripts/deploy_vllm.sh", 0o755)
print("Created: scripts/deploy_vllm.sh")

# Python client
client_code = f'''#!/usr/bin/env python3
"""
Korean MedGemma Python Client

Usage:
    python korean_medgemma_client.py "고혈압의 증상은 무엇인가요?"
"""

import requests
import sys

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "{DEPLOY_MODEL}"

def ask_medical_question(question, language="ko"):
    """Ask a medical question to Korean MedGemma."""

    if language == "ko":
        system_prompt = "당신은 한국어 의료 전문 AI 어시스턴트입니다."
    else:
        system_prompt = "You are a medical AI assistant."

    payload = {{
        "model": MODEL_NAME,
        "messages": [
            {{"role": "system", "content": system_prompt}},
            {{"role": "user", "content": question}}
        ],
        "max_tokens": 512,
        "temperature": 0.7,
    }}

    response = requests.post(API_URL, json=payload)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "고혈압의 증상과 치료법은 무엇인가요?"

    print(f"Question: {{question}}")
    print("\\nAnswer:")

    try:
        answer = ask_medical_question(question)
        print(answer)
    except Exception as e:
        print(f"Error: {{e}}")
        print("Make sure the vLLM server is running.")
'''

with open("scripts/korean_medgemma_client.py", "w") as f:
    f.write(client_code)
os.chmod("scripts/korean_medgemma_client.py", 0o755)
print("Created: scripts/korean_medgemma_client.py")

# Dockerfile
dockerfile = '''# Korean MedGemma Docker Deployment
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \\
    python3 python3-pip git && \\
    rm -rf /var/lib/apt/lists/*

RUN pip3 install vllm autoawq

WORKDIR /app

EXPOSE 8000

CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \\
     "--model", "/app/model", \\
     "--host", "0.0.0.0", \\
     "--port", "8000", \\
     "--quantization", "awq", \\
     "--trust-remote-code"]
'''

with open("Dockerfile", "w") as f:
    f.write(dockerfile)
print("Created: Dockerfile")

# Docker Compose
docker_compose = '''version: "3.8"

services:
  korean-medgemma:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models/korean_medgemma_awq:/app/model:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
    restart: unless-stopped
'''

with open("docker-compose.yml", "w") as f:
    f.write(docker_compose)
print("Created: docker-compose.yml")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Phase 6: Deployment Preparation Complete!")
print("=" * 60)

print(f"""
Model ready for deployment: {DEPLOY_MODEL}

Deployment files created:
  - scripts/deploy_vllm.sh (vLLM server launcher)
  - scripts/korean_medgemma_client.py (Python client)
  - Dockerfile (Docker deployment)
  - docker-compose.yml (Docker Compose)

To start the server:
  bash scripts/deploy_vllm.sh

Or with Docker:
  docker-compose up

API endpoint: http://localhost:8000/v1/chat/completions
""")

# Save deployment info
deployment_info = {
    "phase": "deployment",
    "model_dir": DEPLOY_MODEL,
    "awq_quantized": AWQ_AVAILABLE,
    "files_created": [
        "scripts/deploy_vllm.sh",
        "scripts/korean_medgemma_client.py",
        "Dockerfile",
        "docker-compose.yml",
    ],
}

with open("models/deployment_info.json", "w") as f:
    json.dump(deployment_info, f, indent=2)

print("Deployment info saved to models/deployment_info.json")
print("\nKorean MedGemma training pipeline complete!")
