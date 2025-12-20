# Korean MedGemma Training Plan: Complete Implementation Guide

## Overview

This document provides a detailed, step-by-step training plan for adding Korean language support to MedGemma using a hybrid approach that combines:

1. **Vocabulary Expansion** - Add ~10K Korean tokens for efficient tokenization
2. **Embedding Alignment** - Initialize embeddings using bilingual dictionaries (WECHSEL-style)
3. **EEVE-Style Staged Training** - Progressive unfreezing for stable adaptation
4. **LoRA/Adapters with Embeddings** - Parameter-efficient training including input/output embeddings

---

## Phase 0: Data Preparation

### 0.1 Collect Korean Medical Corpus

**Required Data Types:**

| Data Type | Source | Target Size | Purpose |
|-----------|--------|-------------|---------|
| Korean Medical Text | PubMed Korean, Korean medical journals | 1-2B tokens | Language modeling |
| Korean Wikipedia (Medical) | Wikipedia dumps | 100M tokens | General Korean |
| Korean Health News | News crawl | 200M tokens | Contemporary language |
| Parallel Medical Corpus | Translated medical texts | 50M tokens | Cross-lingual alignment |
| KorMedMCQA | HuggingFace dataset | 7,469 QA pairs | Evaluation & fine-tuning |
| English Medical (retain) | Original MedGemma data | 200M tokens | Prevent forgetting |

**Data Collection Script:**

```python
# scripts/00_collect_data.py
from datasets import load_dataset, concatenate_datasets
import os

def collect_korean_medical_data(output_dir):
    """Collect and prepare Korean medical training data"""

    os.makedirs(output_dir, exist_ok=True)

    # 1. Korean Wikipedia (filter medical categories)
    wiki_ko = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train")
    medical_keywords = ["의학", "질병", "증상", "치료", "약물", "병원", "건강", "의사", "환자"]

    def is_medical(example):
        text = example["text"].lower()
        return any(kw in text for kw in medical_keywords)

    wiki_medical = wiki_ko.filter(is_medical)
    wiki_medical.save_to_disk(f"{output_dir}/wiki_medical_ko")

    # 2. KorMedMCQA for evaluation and instruction tuning
    kormedmcqa = load_dataset("sean0042/KorMedMCQA")
    kormedmcqa.save_to_disk(f"{output_dir}/kormedmcqa")

    # 3. General Korean corpus (for tokenizer training)
    # Use OSCAR or mC4 Korean subset
    oscar_ko = load_dataset("oscar-corpus/OSCAR-2301", "ko", split="train", streaming=True)

    # Save first 10GB for tokenizer training
    tokenizer_corpus = []
    total_chars = 0
    target_chars = 10 * 1024 * 1024 * 1024  # 10GB

    for example in oscar_ko:
        tokenizer_corpus.append(example["text"])
        total_chars += len(example["text"])
        if total_chars >= target_chars:
            break

    with open(f"{output_dir}/korean_corpus_for_tokenizer.txt", "w") as f:
        f.write("\n".join(tokenizer_corpus))

    print(f"Collected {total_chars / 1e9:.2f}GB of Korean text")
    return output_dir

if __name__ == "__main__":
    collect_korean_medical_data("./data/raw")
```

### 0.2 Prepare Bilingual Medical Dictionary

**For WECHSEL-style embedding alignment:**

```python
# scripts/01_prepare_bilingual_dict.py
import json

def create_medical_bilingual_dict():
    """Create bilingual dictionary for medical terms"""

    # Core medical terminology (English -> Korean)
    medical_dict = {
        # Symptoms
        "fever": "발열",
        "cough": "기침",
        "headache": "두통",
        "fatigue": "피로",
        "nausea": "메스꺼움",
        "vomiting": "구토",
        "diarrhea": "설사",
        "pain": "통증",
        "inflammation": "염증",
        "swelling": "부종",

        # Diseases
        "diabetes": "당뇨병",
        "hypertension": "고혈압",
        "cancer": "암",
        "pneumonia": "폐렴",
        "influenza": "독감",
        "arthritis": "관절염",
        "asthma": "천식",
        "stroke": "뇌졸중",
        "heart attack": "심장마비",
        "infection": "감염",

        # Body parts
        "heart": "심장",
        "lung": "폐",
        "liver": "간",
        "kidney": "신장",
        "brain": "뇌",
        "stomach": "위",
        "intestine": "장",
        "blood": "혈액",
        "bone": "뼈",
        "muscle": "근육",

        # Medical professionals
        "doctor": "의사",
        "nurse": "간호사",
        "surgeon": "외과의사",
        "patient": "환자",
        "pharmacist": "약사",

        # Treatments
        "medication": "약물",
        "surgery": "수술",
        "treatment": "치료",
        "diagnosis": "진단",
        "prescription": "처방",
        "injection": "주사",
        "vaccine": "백신",
        "therapy": "치료법",

        # Medical terms
        "symptom": "증상",
        "disease": "질병",
        "chronic": "만성",
        "acute": "급성",
        "benign": "양성",
        "malignant": "악성",
        "contagious": "전염성",
        "hereditary": "유전성",

        # Add more from UMLS or medical dictionaries...
    }

    # Extend with UMLS mappings if available
    # medical_dict.update(load_umls_korean_mappings())

    with open("./data/bilingual_medical_dict.json", "w", encoding="utf-8") as f:
        json.dump(medical_dict, f, ensure_ascii=False, indent=2)

    print(f"Created bilingual dictionary with {len(medical_dict)} entries")
    return medical_dict

if __name__ == "__main__":
    create_medical_bilingual_dict()
```

### 0.3 Data Preprocessing and Formatting

```python
# scripts/02_preprocess_data.py
from transformers import AutoTokenizer
import json

def preprocess_for_training(data_dir, output_dir):
    """Format data for different training stages"""

    # Stage 1-5: Pure language modeling (Korean text)
    # Format: plain text, one document per line

    # Stage 6-7: Instruction tuning (Korean QA)
    # Format: ChatML or similar instruction format

    instruction_template = """<|im_start|>system
당신은 한국어 의료 전문 AI 어시스턴트입니다. 정확하고 도움이 되는 의료 정보를 제공하세요.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
{answer}
<|im_end|>"""

    # Process KorMedMCQA for instruction tuning
    def format_mcqa_for_training(example):
        question = example["question"]
        choices = example["choices"]
        answer_idx = example["answer"]

        formatted_choices = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
        full_question = f"{question}\n\n{formatted_choices}"
        answer = f"정답은 {answer_idx+1}번입니다. {choices[answer_idx]}"

        return instruction_template.format(question=full_question, answer=answer)

    # Save formatted data
    # ... implementation details ...

if __name__ == "__main__":
    preprocess_for_training("./data/raw", "./data/processed")
```

---

## Phase 1: Tokenizer Preparation

### 1.1 Train Korean SentencePiece Tokenizer

```python
# scripts/03_train_korean_tokenizer.py
import sentencepiece as spm

def train_korean_tokenizer(corpus_path, output_dir, vocab_size=40000):
    """Train SentencePiece tokenizer on Korean corpus"""

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=f"{output_dir}/korean_sp",
        vocab_size=vocab_size,
        character_coverage=0.9995,  # High coverage for Korean
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=[],  # Add special medical symbols if needed
        num_threads=16,
        train_extremely_large_corpus=True,
        max_sentence_length=16384,
    )

    print(f"Trained Korean tokenizer with {vocab_size} tokens")
    return f"{output_dir}/korean_sp.model"

if __name__ == "__main__":
    train_korean_tokenizer(
        "./data/korean_corpus_for_tokenizer.txt",
        "./models/tokenizer",
        vocab_size=40000
    )
```

### 1.2 Filter Tokens by Frequency (EEVE Method)

```python
# scripts/04_filter_tokens.py
import sentencepiece as spm
from collections import Counter

def filter_tokens_by_frequency(
    korean_sp_path,
    corpus_path,
    min_frequency=6000,
    output_path="./models/tokenizer/filtered_korean_tokens.txt"
):
    """Filter Korean tokens by frequency (EEVE: keep tokens with >=6000 occurrences)"""

    sp = spm.SentencePieceProcessor()
    sp.Load(korean_sp_path)

    # Count token frequencies in corpus
    token_counts = Counter()

    with open(corpus_path, "r") as f:
        for line in f:
            tokens = sp.EncodeAsPieces(line.strip())
            token_counts.update(tokens)

    # Filter by frequency
    filtered_tokens = [
        token for token, count in token_counts.items()
        if count >= min_frequency
    ]

    # Save filtered tokens
    with open(output_path, "w") as f:
        for token in filtered_tokens:
            f.write(f"{token}\n")

    print(f"Filtered {len(filtered_tokens)} tokens (from {sp.GetPieceSize()})")
    print(f"Frequency threshold: {min_frequency}")

    return filtered_tokens

if __name__ == "__main__":
    filter_tokens_by_frequency(
        "./models/tokenizer/korean_sp.model",
        "./data/korean_corpus_for_tokenizer.txt",
        min_frequency=6000
    )
```

### 1.3 Merge Tokenizers (SentencePiece Protobuf Method)

```python
# scripts/05_merge_tokenizers.py
import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as sp_pb2_model
from transformers import AutoTokenizer

def merge_tokenizers(
    base_model_path,
    korean_tokens_path,
    output_path
):
    """Merge Korean tokens into base MedGemma tokenizer"""

    # Load base tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_vocab = set(base_tokenizer.get_vocab().keys())
    original_vocab_size = len(base_tokenizer)

    # Load filtered Korean tokens
    with open(korean_tokens_path, "r") as f:
        korean_tokens = [line.strip() for line in f]

    # Filter out tokens already in base vocab
    new_tokens = [t for t in korean_tokens if t not in base_vocab]

    # Method 1: Using HuggingFace add_tokens (simpler)
    base_tokenizer.add_tokens(new_tokens)
    base_tokenizer.save_pretrained(output_path)

    print(f"Original vocab size: {original_vocab_size}")
    print(f"New tokens added: {len(new_tokens)}")
    print(f"Final vocab size: {len(base_tokenizer)}")

    # Save mapping for embedding initialization
    token_mapping = {
        "original_vocab_size": original_vocab_size,
        "new_tokens": new_tokens,
        "new_token_ids": {t: base_tokenizer.convert_tokens_to_ids(t) for t in new_tokens}
    }

    import json
    with open(f"{output_path}/token_mapping.json", "w") as f:
        json.dump(token_mapping, f, ensure_ascii=False, indent=2)

    return base_tokenizer, token_mapping

if __name__ == "__main__":
    merge_tokenizers(
        "google/medgemma-4b-it",
        "./models/tokenizer/filtered_korean_tokens.txt",
        "./models/merged_tokenizer"
    )
```

---

## Phase 2: Model Preparation and Embedding Initialization

### 2.1 Resize Model Embeddings

```python
# scripts/06_resize_embeddings.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

def resize_model_embeddings(
    base_model_path,
    merged_tokenizer_path,
    output_path
):
    """Resize model embeddings to accommodate new tokens"""

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(merged_tokenizer_path)

    # Load token mapping
    with open(f"{merged_tokenizer_path}/token_mapping.json", "r") as f:
        token_mapping = json.load(f)

    original_vocab_size = token_mapping["original_vocab_size"]
    new_vocab_size = len(tokenizer)

    print(f"Resizing embeddings: {original_vocab_size} -> {new_vocab_size}")

    # Resize token embeddings
    model.resize_token_embeddings(new_vocab_size)

    # Save resized model (embeddings not yet initialized properly)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    return model, tokenizer, token_mapping

if __name__ == "__main__":
    resize_model_embeddings(
        "google/medgemma-4b-it",
        "./models/merged_tokenizer",
        "./models/resized_model"
    )
```

### 2.2 Initialize Embeddings with Hybrid Method (Subword + WECHSEL Alignment)

```python
# scripts/07_initialize_embeddings.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np
from typing import Dict, List, Optional

class EmbeddingInitializer:
    """Hybrid embedding initialization combining EEVE subword and WECHSEL alignment"""

    def __init__(
        self,
        model,
        new_tokenizer,
        old_tokenizer,
        bilingual_dict: Optional[Dict[str, str]] = None
    ):
        self.model = model
        self.new_tokenizer = new_tokenizer
        self.old_tokenizer = old_tokenizer
        self.bilingual_dict = bilingual_dict or {}

        # Get embedding layers
        self.input_embeds = model.get_input_embeddings().weight.data
        self.output_embeds = model.lm_head.weight.data

        # Original vocab size
        self.original_vocab_size = len(old_tokenizer)

    def initialize_with_subword_average(self, token: str, token_idx: int):
        """EEVE method: Initialize using subword decomposition"""

        # Tokenize with OLD tokenizer to get subword decomposition
        subword_ids = self.old_tokenizer.encode(token, add_special_tokens=False)

        if len(subword_ids) == 0:
            # Fallback to mean of all embeddings
            self.input_embeds[token_idx] = self.input_embeds[:self.original_vocab_size].mean(dim=0)
            self.output_embeds[token_idx] = self.output_embeds[:self.original_vocab_size].mean(dim=0)
            return "mean_fallback"

        # Input embeddings: AVERAGE of all subword embeddings
        self.input_embeds[token_idx] = self.input_embeds[subword_ids].mean(dim=0)

        # Output embeddings: FIRST subword only (EEVE finding)
        self.output_embeds[token_idx] = self.output_embeds[subword_ids[0]]

        return "subword_average"

    def initialize_with_bilingual_alignment(self, korean_token: str, token_idx: int):
        """WECHSEL method: Initialize using bilingual dictionary alignment"""

        # Check if we have a translation
        english_equivalent = None

        # Direct lookup
        for en, ko in self.bilingual_dict.items():
            if ko == korean_token or korean_token in ko:
                english_equivalent = en
                break

        if english_equivalent is None:
            return None  # No alignment found

        # Get English token embedding
        en_token_ids = self.old_tokenizer.encode(english_equivalent, add_special_tokens=False)

        if len(en_token_ids) == 0:
            return None

        # Use average of English subword embeddings
        en_embedding = self.input_embeds[en_token_ids].mean(dim=0)

        # Set Korean token to aligned English embedding
        self.input_embeds[token_idx] = en_embedding
        self.output_embeds[token_idx] = self.output_embeds[en_token_ids[0]]

        return "bilingual_aligned"

    def initialize_all_new_tokens(self, new_tokens: List[str], prefer_bilingual: bool = True):
        """Initialize all new tokens with hybrid method"""

        stats = {"subword_average": 0, "bilingual_aligned": 0, "mean_fallback": 0}

        for token in new_tokens:
            token_idx = self.new_tokenizer.convert_tokens_to_ids(token)

            if token_idx < self.original_vocab_size:
                continue  # Skip original tokens

            method_used = None

            # Try bilingual alignment first if preferred
            if prefer_bilingual and self.bilingual_dict:
                method_used = self.initialize_with_bilingual_alignment(token, token_idx)

            # Fallback to subword average
            if method_used is None:
                method_used = self.initialize_with_subword_average(token, token_idx)

            stats[method_used] += 1

        print(f"Embedding initialization stats:")
        for method, count in stats.items():
            print(f"  {method}: {count}")

        return stats


def initialize_embeddings_hybrid(
    model_path,
    merged_tokenizer_path,
    original_tokenizer_path,
    bilingual_dict_path,
    output_path
):
    """Main function to initialize embeddings with hybrid method"""

    # Load model and tokenizers
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu"  # CPU for embedding manipulation
    )
    new_tokenizer = AutoTokenizer.from_pretrained(merged_tokenizer_path)
    old_tokenizer = AutoTokenizer.from_pretrained(original_tokenizer_path)

    # Load bilingual dictionary
    with open(bilingual_dict_path, "r") as f:
        bilingual_dict = json.load(f)

    # Load token mapping
    with open(f"{merged_tokenizer_path}/token_mapping.json", "r") as f:
        token_mapping = json.load(f)

    new_tokens = token_mapping["new_tokens"]

    # Initialize embeddings
    initializer = EmbeddingInitializer(
        model=model,
        new_tokenizer=new_tokenizer,
        old_tokenizer=old_tokenizer,
        bilingual_dict=bilingual_dict
    )

    stats = initializer.initialize_all_new_tokens(new_tokens, prefer_bilingual=True)

    # Save initialized model
    model.save_pretrained(output_path)
    new_tokenizer.save_pretrained(output_path)

    # Save initialization stats
    with open(f"{output_path}/initialization_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Model saved to {output_path}")
    return model, stats

if __name__ == "__main__":
    initialize_embeddings_hybrid(
        "./models/resized_model",
        "./models/merged_tokenizer",
        "google/medgemma-4b-it",
        "./data/bilingual_medical_dict.json",
        "./models/initialized_model"
    )
```

---

## Phase 3: EEVE-Style Staged Training with LoRA + Embeddings

### 3.1 Training Configuration

```python
# config/training_config.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class StageConfig:
    """Configuration for each training stage"""
    name: str
    train_input_embeddings: bool
    train_output_embeddings: bool
    train_lora_layers: bool
    freeze_old_embeddings: bool
    learning_rate: float
    num_epochs: int
    warmup_ratio: float
    lora_r: int = 0  # 0 means no LoRA
    lora_alpha: int = 0
    lora_target_modules: List[str] = None

# EEVE-Style 7-Stage Training Configuration
TRAINING_STAGES = [
    # Stage 1: New input embeddings only
    StageConfig(
        name="stage_1_new_input_embeds",
        train_input_embeddings=True,
        train_output_embeddings=False,
        train_lora_layers=False,
        freeze_old_embeddings=True,
        learning_rate=1e-4,
        num_epochs=1,
        warmup_ratio=0.03,
    ),

    # Stage 2: New output embeddings only
    StageConfig(
        name="stage_2_new_output_embeds",
        train_input_embeddings=False,
        train_output_embeddings=True,
        train_lora_layers=False,
        freeze_old_embeddings=True,
        learning_rate=1e-4,
        num_epochs=1,
        warmup_ratio=0.03,
    ),

    # Stage 3: Both new embedding sets
    StageConfig(
        name="stage_3_both_new_embeds",
        train_input_embeddings=True,
        train_output_embeddings=True,
        train_lora_layers=False,
        freeze_old_embeddings=True,
        learning_rate=5e-5,
        num_epochs=1,
        warmup_ratio=0.03,
    ),

    # Stage 4: All output embeddings (including old)
    StageConfig(
        name="stage_4_all_output_embeds",
        train_input_embeddings=False,
        train_output_embeddings=True,
        train_lora_layers=False,
        freeze_old_embeddings=False,  # Train ALL output embeddings
        learning_rate=2e-5,
        num_epochs=1,
        warmup_ratio=0.03,
    ),

    # Stage 5: New input + all output embeddings
    StageConfig(
        name="stage_5_new_input_all_output",
        train_input_embeddings=True,
        train_output_embeddings=True,
        train_lora_layers=False,
        freeze_old_embeddings=True,  # Freeze old INPUT, train all OUTPUT
        learning_rate=2e-5,
        num_epochs=1,
        warmup_ratio=0.03,
    ),

    # Stage 6: Full model with QLoRA (main adaptation)
    StageConfig(
        name="stage_6_qlora_full",
        train_input_embeddings=True,
        train_output_embeddings=True,
        train_lora_layers=True,
        freeze_old_embeddings=False,  # Can adjust all embeddings
        learning_rate=2e-4,
        num_epochs=3,
        warmup_ratio=0.03,
        lora_r=64,
        lora_alpha=128,
        lora_target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    ),

    # Stage 7: Cool-down (internal layers only, lower LR)
    StageConfig(
        name="stage_7_cooldown",
        train_input_embeddings=False,
        train_output_embeddings=False,
        train_lora_layers=True,
        freeze_old_embeddings=True,
        learning_rate=5e-5,
        num_epochs=1,
        warmup_ratio=0.1,
        lora_r=32,
        lora_alpha=64,
        lora_target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    ),
]
```

### 3.2 Custom Trainer with Embedding Freezing

```python
# scripts/08_staged_trainer.py
import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk
import json
from typing import Optional
import os

class EmbeddingFreezeHook:
    """Hook to freeze old embeddings during training"""

    def __init__(self, original_vocab_size: int, freeze_old: bool = True):
        self.original_vocab_size = original_vocab_size
        self.freeze_old = freeze_old

    def __call__(self, grad):
        if self.freeze_old:
            # Zero out gradients for original tokens
            grad[:self.original_vocab_size] = 0
        return grad


class StagedTrainer:
    """EEVE-style staged training with LoRA + Embeddings"""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        output_dir: str,
        token_mapping_path: str,
        data_path: str,
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.output_dir = output_dir
        self.data_path = data_path

        # Load token mapping
        with open(token_mapping_path, "r") as f:
            self.token_mapping = json.load(f)

        self.original_vocab_size = self.token_mapping["original_vocab_size"]

        # Hooks for embedding freezing
        self.input_embed_hook = None
        self.output_embed_hook = None

    def load_model_for_stage(self, stage_config, previous_checkpoint: Optional[str] = None):
        """Load model with appropriate configuration for each stage"""

        model_path = previous_checkpoint or self.model_path

        # Load base model
        if stage_config.train_lora_layers:
            # Load with quantization for LoRA stages
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

            # Apply LoRA
            lora_config = LoraConfig(
                r=stage_config.lora_r,
                lora_alpha=stage_config.lora_alpha,
                target_modules=stage_config.lora_target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                # Include embeddings in LoRA if training them
                modules_to_save=self._get_modules_to_save(stage_config),
            )

            model = get_peft_model(model, lora_config)

        else:
            # Load without quantization for embedding-only stages
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        # Configure parameter training
        self._configure_trainable_params(model, stage_config)

        return model

    def _get_modules_to_save(self, stage_config):
        """Get modules to save (not apply LoRA, but train fully)"""
        modules = []
        if stage_config.train_input_embeddings:
            modules.append("embed_tokens")
        if stage_config.train_output_embeddings:
            modules.append("lm_head")
        return modules if modules else None

    def _configure_trainable_params(self, model, stage_config):
        """Configure which parameters are trainable"""

        # First, freeze everything
        for param in model.parameters():
            param.requires_grad = False

        # Get embedding layers
        if hasattr(model, 'model'):
            # For PEFT models
            base_model = model.model if hasattr(model, 'model') else model
            input_embeds = base_model.model.embed_tokens if hasattr(base_model, 'model') else base_model.embed_tokens
            output_embeds = model.lm_head
        else:
            input_embeds = model.get_input_embeddings()
            output_embeds = model.lm_head

        # Configure input embeddings
        if stage_config.train_input_embeddings:
            input_embeds.weight.requires_grad = True

            # Register hook to freeze old embeddings if needed
            if stage_config.freeze_old_embeddings:
                if self.input_embed_hook is not None:
                    self.input_embed_hook.remove()
                hook = EmbeddingFreezeHook(self.original_vocab_size, freeze_old=True)
                self.input_embed_hook = input_embeds.weight.register_hook(hook)

        # Configure output embeddings
        if stage_config.train_output_embeddings:
            output_embeds.weight.requires_grad = True

            # For output, stage 4-5 trains ALL embeddings
            if not stage_config.freeze_old_embeddings:
                if self.output_embed_hook is not None:
                    self.output_embed_hook.remove()
                    self.output_embed_hook = None
            else:
                if self.output_embed_hook is not None:
                    self.output_embed_hook.remove()
                hook = EmbeddingFreezeHook(self.original_vocab_size, freeze_old=True)
                self.output_embed_hook = output_embeds.weight.register_hook(hook)

        # Print trainable params summary
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    def train_stage(self, stage_config, previous_checkpoint: Optional[str] = None):
        """Train a single stage"""

        print(f"\n{'='*60}")
        print(f"Starting {stage_config.name}")
        print(f"{'='*60}")

        # Load model
        model = self.load_model_for_stage(stage_config, previous_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        # Load data
        dataset = load_from_disk(self.data_path)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM
        )

        # Training arguments
        stage_output_dir = f"{self.output_dir}/{stage_config.name}"

        training_args = TrainingArguments(
            output_dir=stage_output_dir,
            num_train_epochs=stage_config.num_epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=stage_config.learning_rate,
            warmup_ratio=stage_config.warmup_ratio,
            lr_scheduler_type="cosine",
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            optim="paged_adamw_8bit" if stage_config.train_lora_layers else "adamw_torch",
            max_grad_norm=1.0,
            report_to="tensorboard",
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            data_collator=data_collator,
        )

        # Train
        trainer.train()

        # Save
        trainer.save_model(stage_output_dir)
        tokenizer.save_pretrained(stage_output_dir)

        # Clean up hooks
        if self.input_embed_hook is not None:
            self.input_embed_hook.remove()
            self.input_embed_hook = None
        if self.output_embed_hook is not None:
            self.output_embed_hook.remove()
            self.output_embed_hook = None

        return stage_output_dir

    def run_all_stages(self, stages):
        """Run all training stages sequentially"""

        previous_checkpoint = None

        for stage_config in stages:
            checkpoint = self.train_stage(stage_config, previous_checkpoint)
            previous_checkpoint = checkpoint

        print(f"\n{'='*60}")
        print(f"All stages complete!")
        print(f"Final model: {previous_checkpoint}")
        print(f"{'='*60}")

        return previous_checkpoint


if __name__ == "__main__":
    from config.training_config import TRAINING_STAGES

    trainer = StagedTrainer(
        model_path="./models/initialized_model",
        tokenizer_path="./models/merged_tokenizer",
        output_dir="./models/staged_training",
        token_mapping_path="./models/merged_tokenizer/token_mapping.json",
        data_path="./data/processed/korean_medical",
    )

    final_model = trainer.run_all_stages(TRAINING_STAGES)
```

### 3.3 DeepSpeed Configuration for Memory Efficiency

```json
// config/ds_config_zero2.json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

---

## Phase 4: Instruction Tuning (Korean Medical QA)

### 4.1 Instruction Tuning with LoRA

```python
# scripts/09_instruction_tuning.py
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
from datasets import load_from_disk

def instruction_tune_korean_medical(
    base_model_path: str,
    instruction_data_path: str,
    output_dir: str,
):
    """Fine-tune on Korean medical instruction data"""

    # Load model from staged training
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Ensure padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config for instruction tuning
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Load instruction data
    dataset = load_from_disk(instruction_data_path)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        max_seq_length=2048,
        dataset_text_field="text",
    )

    trainer.train()
    trainer.save_model(output_dir)

    return output_dir

if __name__ == "__main__":
    instruction_tune_korean_medical(
        "./models/staged_training/stage_7_cooldown",
        "./data/processed/kormedmcqa_instruction",
        "./models/instruction_tuned"
    )
```

---

## Phase 5: Evaluation

### 5.1 Evaluate on KorMedMCQA

```python
# scripts/10_evaluate.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json

def evaluate_kormedmcqa(model_path: str, output_path: str):
    """Evaluate on KorMedMCQA benchmark"""

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load test set
    dataset = load_dataset("sean0042/KorMedMCQA", split="test")

    correct = 0
    total = 0
    results = []

    for example in tqdm(dataset):
        question = example["question"]
        choices = example["choices"]
        answer = example["answer"]

        # Format prompt
        formatted_choices = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])
        prompt = f"""다음 의료 관련 질문에 답하세요. 정답 번호만 답하세요.

질문: {question}

선택지:
{formatted_choices}

정답:"""

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Parse answer
        predicted = None
        for i in range(1, 6):
            if str(i) in response:
                predicted = i - 1
                break

        is_correct = predicted == answer
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question,
            "predicted": predicted,
            "answer": answer,
            "correct": is_correct,
            "response": response,
        })

    accuracy = correct / total * 100

    print(f"\n{'='*60}")
    print(f"KorMedMCQA Results")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    return accuracy

if __name__ == "__main__":
    evaluate_kormedmcqa(
        "./models/instruction_tuned",
        "./results/kormedmcqa_eval.json"
    )
```

### 5.2 Evaluate English Retention

```python
# scripts/11_evaluate_english.py
def evaluate_english_retention(model_path: str, output_path: str):
    """Evaluate English medical capabilities to check for catastrophic forgetting"""

    # Use MedQA or similar English medical benchmark
    # Compare with original MedGemma baseline

    # ... implementation ...
    pass
```

---

## Phase 6: Quantization and Deployment

### 6.1 AWQ Quantization

```python
# scripts/12_quantize.py
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig

def quantize_for_deployment(model_path: str, output_path: str):
    """Quantize model for efficient inference"""

    awq_config = AwqConfig(
        bits=4,
        group_size=128,
        zero_point=True,
        version="GEMM",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=awq_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"Quantized model saved to {output_path}")
    return output_path

if __name__ == "__main__":
    quantize_for_deployment(
        "./models/instruction_tuned",
        "./models/korean_medgemma_awq"
    )
```

### 6.2 vLLM Deployment

```bash
# scripts/deploy_vllm.sh
#!/bin/bash

vllm serve ./models/korean_medgemma_awq \
    --quantization awq \
    --dtype half \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --host 0.0.0.0 \
    --port 8000
```

---

## Complete Training Pipeline Script

```bash
#!/bin/bash
# run_training_pipeline.sh

set -e

echo "=========================================="
echo "Korean MedGemma Training Pipeline"
echo "=========================================="

# Phase 0: Data Preparation
echo "Phase 0: Collecting and preparing data..."
python scripts/00_collect_data.py
python scripts/01_prepare_bilingual_dict.py
python scripts/02_preprocess_data.py

# Phase 1: Tokenizer Preparation
echo "Phase 1: Preparing tokenizer..."
python scripts/03_train_korean_tokenizer.py
python scripts/04_filter_tokens.py
python scripts/05_merge_tokenizers.py

# Phase 2: Model Preparation
echo "Phase 2: Preparing model embeddings..."
python scripts/06_resize_embeddings.py
python scripts/07_initialize_embeddings.py

# Phase 3: Staged Training
echo "Phase 3: Running EEVE-style staged training..."
python scripts/08_staged_trainer.py

# Phase 4: Instruction Tuning
echo "Phase 4: Instruction tuning..."
python scripts/09_instruction_tuning.py

# Phase 5: Evaluation
echo "Phase 5: Evaluating model..."
python scripts/10_evaluate.py
python scripts/11_evaluate_english.py

# Phase 6: Deployment
echo "Phase 6: Quantizing for deployment..."
python scripts/12_quantize.py

echo "=========================================="
echo "Training pipeline complete!"
echo "Final model: ./models/korean_medgemma_awq"
echo "=========================================="
```

---

## Summary: Training Stage Overview

| Stage | What Trains | Data | LR | Epochs | Purpose |
|-------|-------------|------|-----|--------|---------|
| **1** | New input embeddings | Korean LM | 1e-4 | 1 | Learn Korean token recognition |
| **2** | New output embeddings | Korean LM | 1e-4 | 1 | Learn Korean generation |
| **3** | Both new embeddings | Korean LM | 5e-5 | 1 | Align input/output |
| **4** | All output embeddings | Korean LM | 2e-5 | 1 | Integrate full vocabulary |
| **5** | New input + all output | Korean LM | 2e-5 | 1 | Full vocabulary harmony |
| **6** | LoRA + embeddings | Korean LM (90%) + English (10%) | 2e-4 | 3 | Deep adaptation |
| **7** | LoRA only (cooldown) | Mixed | 5e-5 | 1 | Stabilize |
| **8** | LoRA (instruction) | KorMedMCQA | 2e-5 | 3 | Medical QA capability |

---

## Hardware Requirements

| Phase | VRAM (4B Model) | Estimated Time (A5000) |
|-------|-----------------|------------------------|
| Stage 1-5 (Embeddings) | ~10-12 GB | 2-3 days |
| Stage 6-7 (QLoRA) | ~15-20 GB | 3-5 days |
| Stage 8 (Instruction) | ~15 GB | 1 day |
| **Total** | **~20 GB peak** | **~7-10 days** |

---

## Key Success Factors

1. **Never use random embedding initialization** - Always use subword decomposition or bilingual alignment
2. **Progressive unfreezing** - Stabilizes training and prevents catastrophic forgetting
3. **Include 10% English data in Stage 6** - Critical for retaining English capabilities
4. **Use gradient hooks to freeze old embeddings** - Protects original knowledge in early stages
5. **LoRA includes embeddings via `modules_to_save`** - Allows full embedding training with LoRA
6. **Evaluate both Korean AND English** - Ensure no catastrophic forgetting
