# Korean MedGemma Project Status

**Last Updated:** 2025-12-07 (In Progress)

## Hardware Configuration
| GPU | VRAM | Role |
|-----|------|------|
| NVIDIA RTX A6000 | 49GB | Primary Training |
| NVIDIA TITAN RTX | 24GB | Secondary/Inference |

## Project Overview
Adding Korean language support to Google's MedGemma 4B using EEVE-style vocabulary expansion.

---

## Phase Status

### Phase 0: Data Preparation - COMPLETED
| Notebook | Status | Output |
|----------|--------|--------|
| 00_setup_environment | Completed | Yes |
| 01_research_datasets | Completed | Yes |
| 02_collect_korean_medical | Completed | Yes |
| 03_collect_bilingual_dict | Completed | Yes |
| 04_preprocess_data | Completed | Yes |
| 05_mcp_jupyter_setup | Ready | - |

**Data Summary:**
- Korean tokenizer corpus: 232MB (~43,566 texts, 43M tokens)
- Korean medical LM data: 41,196 train / 2,169 val
- Korean medical instruction: 2,244 train / 250 val
- KorMedMCQA evaluation: 604 samples
- Bilingual medical dictionary: 312 entries

### Phase 1: Tokenizer Training - COMPLETED
| Notebook | Status | Notes |
|----------|--------|-------|
| 01_train_korean_tokenizer | Completed | 40K vocab SentencePiece BPE |
| 02_filter_tokens | Completed | 10K Korean tokens selected |
| 03_merge_tokenizers | Completed | 6,211 new tokens added |

**Tokenizer Summary:**
- Original MedGemma vocab: 262,145 tokens
- New Korean tokens: 6,211 (after deduplication)
- Final vocab size: 268,356 tokens

### Phase 2: Embedding Initialization - COMPLETED
| Notebook | Status | Notes |
|----------|--------|-------|
| 01_resize_embeddings | Completed | 262,208 → 268,356 |
| 02_initialize_embeddings | Completed | Hybrid EEVE+WECHSEL init |

**Initialization Summary:**
- Subword average (EEVE): 6,192 tokens (99.7%)
- Bilingual aligned (WECHSEL): 20 tokens (0.3%)
- Model saved to: `models/initialized_model/`

### Phase 3: 7-Stage EEVE Training - IN PROGRESS
| Stage | Notebook | Status | Description | Est. Time |
|-------|----------|--------|-------------|-----------|
| 1 | 01_stage1_new_input_embeds | ✅ Complete | Train new input embeddings only | ~2h |
| 2 | 02_stage2_new_output_embeds | ✅ Complete | Train new output embeddings only | ~2h |
| 3 | 03_stage3_both_new_embeds | ✅ Complete | Train both new embeddings jointly | ~2h |
| 4 | 04_stage4_all_output_embeds | 🔄 Running | Unfreeze all output embeddings | ~2h |
| 5 | 05_stage5_harmonization | Pending | New input + all output harmonization | ~2h |
| 6 | 06_stage6_hybrid_expansion | Pending | **Hybrid: Identity Layers + QLoRA** | ~3h |
| 7 | 07_stage7_cooldown | Pending | Cooldown + merge expanded model | ~2h |

**Stage 1-3 Training:** ✅ COMPLETED
- Output: `models/staged_training/stage{1,2,3}_*/`

**Stage 4 Training Status:** 🔄 RUNNING
- Progress: ~1%
- Rate: ~3.0 seconds per step
- ETA: ~2 hours remaining
- Description: Train ALL output embeddings (lm_head)
- Input: Stage 3 model output

**Training Scripts Ready:**
- `scripts/run_stage2.py` - Stage 2: Train new output embeddings
- `scripts/run_stage3.py` - Stage 3: Train both new embeddings
- `scripts/run_stage4.py` - Stage 4: Train all output embeddings
- `scripts/run_stage5.py` - Stage 5: Harmonization
- `scripts/run_stage6.py` - Stage 6: QLoRA full adaptation
- `scripts/run_stage7.py` - Stage 7: Cooldown + merge

### Phase 4: Instruction Tuning
| Notebook | Status | Notes |
|----------|--------|-------|
| 01_instruction_tuning | Pending | Fine-tune on Korean medical QA |

### Phase 5: Evaluation
| Notebook | Status | Notes |
|----------|--------|-------|
| 01_evaluate_korean | Pending | KorMedMCQA benchmark |
| 02_evaluate_english | Pending | English retention test |

### Phase 6: Deployment
| Notebook | Status | Notes |
|----------|--------|-------|
| 01_quantize_awq | Pending | AWQ 4-bit quantization |
| 02_deploy_vllm | Pending | vLLM server deployment |

### Phase 7: Layer Expansion (Future - MedGemma 27B) - READY
| Notebook | Status | Notes |
|----------|--------|-------|
| 01_identity_layer_expansion | ✅ Ready | LLaMA Pro-style identity init |
| 02_hybrid_expansion_qlora | ✅ Ready | Identity + QLoRA + Embeddings |
| 03_solar_depth_upscale | ✅ Ready | SOLAR DUS layer duplication |

**Research Summary:** (see `research/layer_expansion_techniques.md`)

Two main approaches for adding new layers:
1. **SOLAR (DUS):** Duplicate front/rear layers (+53% params)
2. **LLaMA Pro:** Insert identity-initialized layers (+18% params)

**Recommended Hybrid Approach for MedGemma 27B:**
- Identity layer expansion (2 new layers): ~1.7B new params
- QLoRA on middle layers (r=32): ~0.3B params
- Full embeddings training: ~3B params
- **Total VRAM:** ~43-45 GB (fits A5000/A6000)

| Method | New Params | Layers | Training VRAM | A5000 (48GB) |
|--------|------------|--------|---------------|--------------|
| QLoRA (r=64) | 0.3B (1%) | 32 | ~30 GB | Easy |
| Identity expansion (2) | 1.7B (6%) | 34 | ~43 GB | Fits |
| Identity expansion (4) | 3.4B (12%) | 36 | ~50 GB | Tight |

---

## Models

### Base Model
- **Path:** `models/medgemma-4b-it/`
- **Size:** ~8GB (safetensors)
- **Status:** Downloaded

### Training Outputs (To Be Created)
- `models/korean_tokenizer/` - Korean SentencePiece model
- `models/merged_tokenizer/` - Merged Gemma + Korean tokenizer
- `models/stage{1-7}_checkpoint/` - Stage checkpoints
- `models/instruction_tuned/` - Final instruction-tuned model
- `models/korean_medgemma_awq/` - Quantized deployment model

---

## Current Progress

### Completed
- [x] Project structure setup
- [x] Data collection and preprocessing
- [x] Bilingual dictionary creation
- [x] All notebook templates created (Phases 0-6)
- [x] MCP Jupyter setup documentation
- [x] Phase 1: Tokenizer training (6,211 new Korean tokens)
- [x] Phase 2: Embedding initialization (hybrid EEVE+WECHSEL)
- [x] Phase 3 Stage 1: Train new input embeddings
- [x] Phase 3 Stage 2: Train new output embeddings
- [x] Phase 3 Stage 3: Train both new embeddings jointly
- [x] Phase 7 Research: Layer expansion techniques (SOLAR, LLaMA Pro)

### In Progress
- [ ] Phase 3 Stage 4: Train all output embeddings (~1% complete)

### Blocked
- None

---

## Next Steps

1. **Set up Python environment** with required packages
2. **Run Phase 1 notebooks** (tokenizer training)
3. **Run Phase 2 notebooks** (embedding initialization)
4. **Run Phase 3 notebooks** (7-stage EEVE training)
5. Continue through remaining phases

---

## Environment Requirements

```bash
# Core packages
pip install torch transformers accelerate
pip install sentencepiece tokenizers
pip install datasets peft bitsandbytes
pip install trl  # For SFTTrainer
pip install autoawq  # For quantization
pip install vllm  # For deployment

# Bilingual alignment
pip install wechsel
```

---

## Training Configuration

### Stage 1-5 (Embedding Training)
- Optimizer: AdamW
- Learning rate: 1e-4 → 5e-5
- Batch size: 8-16 (gradient accumulation)
- Data: Korean medical LM corpus

### Stage 6-7 (LoRA Training)
- LoRA rank: 64
- LoRA alpha: 128
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Quantization: 4-bit NF4

---

## Notes

- Using MedGemma 4B IT (instruction-tuned) as base
- Target: ~10K new Korean tokens
- Hybrid embedding init: Subword decomposition + WECHSEL alignment
- Total estimated training: 7 stages + instruction tuning
