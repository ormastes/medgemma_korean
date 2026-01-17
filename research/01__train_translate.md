# Translation Training Research: Optimal Data Size, Epochs, and Convergence

## Overview

This document summarizes research findings on how much training is needed for translation fine-tuning of LLMs, with specific focus on Korean-English translation.

---

## Key Research Papers

### 1. "How Much Data is Enough Data?" (arXiv 2024)
**Source:** [arXiv:2409.03454](https://arxiv.org/html/2409.03454v1)

Fine-tuning Llama 3 8B Instruct for translation with varying dataset sizes.

#### Key Findings:

| Dataset Size | BLEU Δ | COMET Δ | Recommendation |
|-------------|--------|---------|----------------|
| 1k-2k | -2 to -5 | -10 to -15 | ❌ Performance degradation |
| 5k | +1 to +3 | +5 to +8 | ⚠️ Minimum threshold |
| 10k-15k | +4.8 | +16.9 | ✅ Good starting point |
| 100k+ | +13.7 | +25.0 | ✅ Optimal for production |

#### Language-Specific Results:

| Language | COMET Improvement | Notes |
|----------|------------------|-------|
| PT-BR (high resource) | +46% | Baseline already strong |
| German (high resource) | +42% | Steady improvement |
| **Korean (low resource)** | **+130%** | **Largest gain** |

**Critical Finding:** Korean showed 130% COMET improvement (baseline → 100k+), nearly 3x the average gain of other languages. Low-resource languages benefit disproportionately from larger fine-tuning datasets.

---

### 2. "Fine-Tuning LLMs to Translate" (EMNLP 2024)
**Source:** [ACL Anthology](https://aclanthology.org/2024.emnlp-main.24.pdf)

#### Minimal Data Fine-tuning:

| Training Examples | Performance | Method |
|------------------|-------------|--------|
| 1-3 | Baseline | In-context learning (ICL) |
| 32 | Above baseline | SFT outperforms ICL |
| 64-128 | Good | Consistent improvements |
| 256+ | Strong | Near full fine-tuning quality |

**Key Insight:** With only **32 training examples**, Llama-2 outperforms general-purpose instruction-tuned baselines on translation tasks.

---

### 3. "When Scaling Meets LLM Finetuning" (ICLR 2024)
**Source:** [OpenReview](https://openreview.net/pdf?id=5HCnKDeTws)

#### Scaling Laws for Translation:

```
Performance ∝ (Model_Size)^αm × (Data_Size)^αd
```

Where:
- αm (model scaling exponent) > αd (data scaling exponent)
- **Implication:** Larger model helps more than more data for same compute budget

#### WMT Translation Results:

| Setting | Finding |
|---------|---------|
| En-De | Fitted multiplicative scaling law |
| En-Zh | Similar patterns, language-dependent |
| Recommendation | Balance model size and data size based on task |

---

### 4. Korean-English Translation Studies

#### Stanford CS224N Study
**Source:** [Stanford](https://web.stanford.edu/class/cs224n/final-reports/256985783.pdf)

- **Training iterations:** 20,000+ needed for meaningful convergence
- **Observation:** Small loss doesn't guarantee good translation
- **Issue:** Output can be grammatically correct but semantically wrong

#### Korean NMT Tokenization Study
**Source:** [Korea Science](https://koreascience.or.kr/article/JAKO202111037333482.page)

- **Best results:** 50,000 epochs with BPE tokenization
- **BLEU progression:**
  - 20,000 epochs: 34.50 BLEU
  - 50,000 epochs: 35.73 BLEU
- **Recommendation:** Korean needs morpheme-aware tokenization

---

## Optimal Hyperparameters for Translation LoRA

### Learning Rate

| Scenario | Learning Rate | Notes |
|----------|--------------|-------|
| LoRA (rank 8-64) | 2e-4 | Starting point |
| LoRA (high rank 128+) | 1e-4 | More stable |
| Full fine-tuning | 2e-5 | 10x lower than LoRA |
| Translation-specific | 1e-4 to 5e-5 | Conservative for quality |

**Source:** [Unsloth LoRA Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)

### LoRA Rank and Alpha

| Rank (r) | Alpha (α) | Use Case |
|----------|-----------|----------|
| 8 | 16 | Quick experiments |
| 16 | 32 | Standard training |
| 32-64 | 64-128 | Complex tasks (translation) |
| 128+ | 256+ | Near full fine-tuning |

**Best Practice:** α = 2 × r (empirically optimal ratio)

**Source:** [Databricks LoRA Guide](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms)

### Epochs

| Dataset Size | Recommended Epochs | Reasoning |
|-------------|-------------------|-----------|
| < 5k | 3-5 | Need multiple passes |
| 5k-20k | 2-3 | Standard range |
| 20k-100k | 1-2 | Avoid overfitting |
| 100k+ | 1 | Single pass sufficient |

**Warning:** LLMs can overfit quickly. Early stopping recommended.

**Source:** [SuperAnnotate Fine-tuning Guide](https://www.superannotate.com/blog/llm-fine-tuning)

### Batch Size

| GPU Memory | Batch Size | Gradient Accumulation | Effective Batch |
|-----------|------------|----------------------|-----------------|
| 24GB | 2-4 | 8-16 | 32-64 |
| 48GB | 4-8 | 4-8 | 32-64 |
| 80GB | 8-16 | 2-4 | 32-64 |

**Note:** Smaller batch sizes may improve LoRA convergence.

---

## Recommended Training Configuration for Korean-English

### Phase 1: Minimal Viable Training
```python
# For quick validation
DATA_SIZE = 5_000          # Minimum threshold
EPOCHS = 3
LEARNING_RATE = 2e-4
LORA_RANK = 32
LORA_ALPHA = 64
BATCH_SIZE = 4
GRAD_ACCUM = 8
```

### Phase 2: Standard Training
```python
# For production-ready model
DATA_SIZE = 20_000         # Good balance
EPOCHS = 2
LEARNING_RATE = 1e-4
LORA_RANK = 64
LORA_ALPHA = 128
BATCH_SIZE = 4
GRAD_ACCUM = 8
```

### Phase 3: Optimal Training (Korean)
```python
# For best Korean translation quality
DATA_SIZE = 100_000+       # Korean benefits most
EPOCHS = 1
LEARNING_RATE = 5e-5
LORA_RANK = 64
LORA_ALPHA = 128
BATCH_SIZE = 4
GRAD_ACCUM = 16
```

---

## Training Steps Calculation

### Formula:
```
Total Steps = (Dataset Size × Epochs) / (Batch Size × Gradient Accumulation)
```

### Example Calculations:

| Data Size | Epochs | Batch | Grad Accum | Total Steps |
|-----------|--------|-------|------------|-------------|
| 5,000 | 3 | 4 | 8 | 469 |
| 20,000 | 2 | 4 | 8 | 1,250 |
| 100,000 | 1 | 4 | 16 | 1,563 |

### Recommended Checkpoints:

| Total Steps | Checkpoint Every | Eval Every |
|-------------|-----------------|------------|
| < 500 | 100 | 50 |
| 500-2000 | 250 | 100 |
| 2000+ | 500 | 200 |

---

## Convergence Monitoring

### Signs of Good Convergence:
1. **Loss curve:** Smooth decrease, no spikes
2. **Validation BLEU:** Steady increase
3. **Translation samples:** Improving fluency

### Signs of Overfitting:
1. **Training loss:** Still decreasing
2. **Validation loss:** Increasing or plateau
3. **Translations:** Memorizing training data

### Early Stopping Criteria:
```python
# Stop if validation loss increases for N evaluations
PATIENCE = 3
MIN_DELTA = 0.01  # Minimum improvement required
```

---

## Korean-Specific Considerations

### 1. Tokenization
- Use subword tokenization (BPE/SentencePiece)
- Consider Korean morpheme boundaries
- Extended vocabulary helps (our 23,699 new tokens)

### 2. Data Quality > Quantity
- Clean parallel pairs essential
- Domain-specific data (medical) more valuable
- Remove misaligned translations

### 3. Bidirectional Training
- Train both en→ko and ko→en
- Improves overall understanding
- Doubles effective training data

### 4. Low-Resource Boost
- Korean benefits 130% vs 46% average (COMET)
- Invest in larger datasets for Korean
- Quality human translations preferred

---

## Training Schedule Recommendation

### For MedGemma Korean Translation:

```
┌─────────────────────────────────────────────────────────────┐
│                 RECOMMENDED TRAINING PLAN                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1: Warmup (Steps 0-200)                              │
│    - LR warmup: 0 → 1e-4                                    │
│    - Monitor for stability                                   │
│                                                              │
│  Phase 2: Main Training (Steps 200-1500)                    │
│    - Constant LR: 1e-4                                      │
│    - Eval every 100 steps                                   │
│    - Save checkpoint every 250 steps                        │
│                                                              │
│  Phase 3: Cooldown (Steps 1500-2000)                        │
│    - LR decay: 1e-4 → 1e-5                                  │
│    - Fine-grained evaluation                                │
│                                                              │
│  Early Stop: If val_loss increases 3 consecutive evals      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary: Quick Reference

| Parameter | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **Data Size** | 5,000 | 20,000 | 100,000+ |
| **Epochs** | 1 | 2 | 1-2 |
| **Learning Rate** | 5e-5 | 1e-4 | 2e-4 |
| **LoRA Rank** | 16 | 32-64 | 64 |
| **LoRA Alpha** | 32 | 64-128 | 128 |
| **Batch Size** | 2 | 4 | 4-8 |
| **Grad Accum** | 4 | 8 | 8-16 |
| **Total Steps** | ~500 | ~1,500 | ~2,000 |

---

## References

1. [How Much Data is Enough Data? (arXiv 2024)](https://arxiv.org/html/2409.03454v1)
2. [Fine-Tuning LLMs to Translate (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.24.pdf)
3. [When Scaling Meets LLM Finetuning (ICLR 2024)](https://openreview.net/pdf?id=5HCnKDeTws)
4. [Teaching LLMs to Translate with Comparison (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29920/31609)
5. [Korean-English NMT with Multiple Tokenization (arXiv 2021)](https://arxiv.org/abs/2105.14274)
6. [Unsloth LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
7. [Databricks LoRA Guide](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms)
8. [SuperAnnotate Fine-tuning Guide](https://www.superannotate.com/blog/llm-fine-tuning)
