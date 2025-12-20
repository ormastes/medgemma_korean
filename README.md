# Korean MedGemma: Adding Korean Language Support to MedGemma

This project implements Korean language adaptation for Google's MedGemma using the EEVE-style vocabulary expansion method combined with LoRA fine-tuning and embedding alignment.

## Project Overview

**Goal**: Create a Korean medical LLM by adapting MedGemma with:
- Vocabulary expansion (~10K Korean tokens)
- Hybrid embedding initialization (Subword + WECHSEL alignment)
- EEVE-style 7-stage progressive training
- LoRA with input/output embedding training

**Hardware Target**:
- Training: NVIDIA RTX A5000 (48GB)
- Inference: NVIDIA RTX 4090 (24GB)

## Project Structure

```
medgemma_korean/
├── README.md
├── config/                          # Configuration files
│   ├── training_config.py
│   └── ds_config_zero2.json
├── data/                            # Training data
│   ├── raw/
│   ├── processed/
│   └── bilingual_dict/
├── models/                          # Model checkpoints
│   ├── tokenizer/
│   ├── initialized/
│   ├── staged_training/
│   └── final/
├── research/                        # Research documents
│   ├── add_language_llvm.md
│   ├── eeve_korean_to_llm.md
│   ├── strategy_comparison.md
│   └── training_plan_detailed.md
│
├── phase0_data_preparation/         # Phase 0: Data Collection
│   ├── 00_setup_environment.ipynb
│   ├── 01_research_datasets.ipynb
│   ├── 02_collect_korean_medical.ipynb
│   ├── 03_collect_bilingual_dict.ipynb
│   └── 04_preprocess_data.ipynb
│
├── phase1_tokenizer/                # Phase 1: Tokenizer
│   ├── 01_train_korean_tokenizer.ipynb
│   ├── 02_filter_tokens.ipynb
│   └── 03_merge_tokenizers.ipynb
│
├── phase2_embedding/                # Phase 2: Embedding Init
│   ├── 01_resize_embeddings.ipynb
│   └── 02_initialize_embeddings.ipynb
│
├── phase3_staged_training/          # Phase 3: EEVE Training
│   ├── 01_stage1_new_input_embeds.ipynb
│   ├── 02_stage2_new_output_embeds.ipynb
│   ├── 03_stage3_both_embeds.ipynb
│   ├── 04_stage4_all_output.ipynb
│   ├── 05_stage5_harmonization.ipynb
│   ├── 06_stage6_qlora_full.ipynb
│   └── 07_stage7_cooldown.ipynb
│
├── phase4_instruction_tuning/       # Phase 4: Instruction Tuning
│   └── 01_instruction_tuning.ipynb
│
├── phase5_evaluation/               # Phase 5: Evaluation
│   ├── 01_evaluate_korean.ipynb
│   └── 02_evaluate_english.ipynb
│
└── phase6_deployment/               # Phase 6: Deployment
    ├── 01_quantize_awq.ipynb
    └── 02_deploy_vllm.ipynb
```

## Training Phases

### Phase 0: Data Preparation
- Setup environment (PyTorch, CUDA, A5000)
- Research and collect Korean medical datasets
- Build bilingual medical dictionary
- Preprocess and format data

### Phase 1: Tokenizer Preparation
- Train Korean SentencePiece tokenizer (40K vocab)
- Filter tokens by frequency (>= 6000 occurrences)
- Merge with MedGemma tokenizer (~10K new tokens)

### Phase 2: Embedding Initialization
- Resize model embeddings
- Hybrid initialization:
  - Subword decomposition (EEVE method)
  - Bilingual alignment (WECHSEL method)

### Phase 3: EEVE-Style Staged Training
| Stage | Parameters Trained | Purpose |
|-------|-------------------|---------|
| 1 | New input embeddings | Learn Korean token recognition |
| 2 | New output embeddings | Learn Korean generation |
| 3 | Both new embeddings | Align representations |
| 4 | All output embeddings | Integrate vocabulary |
| 5 | New input + all output | Full harmonization |
| 6 | LoRA + all embeddings | Deep adaptation |
| 7 | LoRA only (cooldown) | Stabilization |

### Phase 4: Instruction Tuning
- Fine-tune on KorMedMCQA
- Korean medical instruction following

### Phase 5: Evaluation
- KorMedMCQA benchmark (Korean)
- English medical retention test

### Phase 6: Deployment
- AWQ 4-bit quantization
- vLLM serving

## Data Sources

### Korean Medical Data
| Source | Type | Size | URL |
|--------|------|------|-----|
| KorMedMCQA | QA | 7,469 | huggingface.co/datasets/sean0042/KorMedMCQA |
| Korean Wikipedia (Medical) | Text | ~100M tokens | wikimedia/wikipedia |
| OSCAR Korean | General | 10GB+ | oscar-corpus/OSCAR-2301 |
| mC4 Korean | General | Large | mc4 |

### Bilingual Resources
| Source | Type |
|--------|------|
| UMLS Korean mappings | Medical terminology |
| Custom medical dictionary | Domain terms |

## Hardware Requirements

| Phase | VRAM | Time (A5000) |
|-------|------|--------------|
| Phase 0-2 | < 16 GB | 1 day |
| Phase 3 (Stage 1-5) | ~12 GB | 2-3 days |
| Phase 3 (Stage 6-7) | ~20 GB | 3-5 days |
| Phase 4 | ~15 GB | 1 day |
| Phase 5-6 | ~10 GB | < 1 day |
| **Total** | **~20 GB peak** | **~7-10 days** |

## Quick Start

```bash
# 1. Setup environment
cd phase0_data_preparation
jupyter notebook 00_setup_environment.ipynb

# 2. Run phase notebooks sequentially
# Each phase directory contains numbered notebooks

# 3. Final model location
# models/final/korean_medgemma_awq
```

## Key References

- [EEVE Paper](https://arxiv.org/abs/2402.14714) - Efficient vocabulary expansion
- [WECHSEL](https://arxiv.org/abs/2112.06598) - Cross-lingual embedding transfer
- [MedGemma](https://ai.google.dev/gemma/docs/medgemma) - Base model

## License

Apache License 2.0
