# Korean Language Adaptation Strategies for MedGemma

## Comparison Table

| Strategy | Complexity | Training Data Required | VRAM (4B Model) | Token Efficiency | English Retention | Best For |
|----------|------------|----------------------|-----------------|------------------|-------------------|----------|
| **1. Translation Layer** | Low | None | ~4-8 GB | N/A | 100% | Quick deployment, prototyping |
| **2. Vocab Expansion + Continued Pretraining** | High | 200M-2B tokens | 10-18 GB | 3x improvement | High (with mixing) | Production, high-quality |
| **3. LoRA/Adapters Only** | Low | 10M-100M tokens | ~4.5 GB | Poor (3-26x overhead) | Very High | Task-specific, limited data |
| **4. EEVE-Style (Vocab + Staged Training)** | Medium-High | 2B tokens | 15-18 GB | 3x improvement | Very High | Recommended balance |
| **5. Embedding Alignment (WECHSEL)** | Medium | 50M-200M tokens | ~10 GB | 3x improvement | High | Limited training data |
| **6. Prompt Tuning/In-Context** | Very Low | None | ~4-8 GB | N/A | 100% | API-only access, quick tests |

---

## Detailed Strategy Breakdown

### 1. Front-End Translation Layer

| Aspect | Details |
|--------|---------|
| **Process** | Input (Korean) → Translate to English → LLM → Translate back to Korean |
| **Pros** | No model modification, fast deployment, preserves all English capabilities |
| **Cons** | Translation errors propagate, added latency (2 extra steps), domain terms may translate poorly |
| **Implementation** | Use MarianMT, M2M100, or commercial APIs (Google/DeepL) |
| **When to Use** | Prototyping, when model modification is not possible |

### 2. Vocabulary Expansion + Continued Pretraining

| Aspect | Details |
|--------|---------|
| **Process** | Train Korean tokenizer → Merge vocab → Initialize embeddings → Continued pretraining |
| **Pros** | Native Korean understanding, efficient tokenization, maintains bilingual capability |
| **Cons** | Requires substantial compute and data, risk of catastrophic forgetting |
| **Data Needed** | 200M-2B+ tokens of Korean text |
| **Key Steps** | 1) Collect corpus, 2) Train tokenizer, 3) Merge vocab, 4) Initialize embeddings, 5) Pretrain |

### 3. LoRA/Adapters Only (No Tokenizer Change)

| Aspect | Details |
|--------|---------|
| **Process** | Apply LoRA/Adapters to frozen base model, fine-tune on Korean data |
| **Pros** | Minimal compute, preserves English perfectly, small adapter files |
| **Cons** | Inefficient tokenization (3-26x longer), limited Korean fluency |
| **VRAM** | ~4.5 GB (4-bit), ~10 GB (16-bit) |
| **When to Use** | Quick experiments, task-specific Korean adaptation |

### 4. EEVE-Style (Recommended)

| Aspect | Details |
|--------|---------|
| **Process** | Add ~9K Korean tokens → Subword-based embedding init → 7-stage progressive unfreezing |
| **Pros** | SOTA results with only 2B tokens, preserves English, efficient inference |
| **Cons** | Complex multi-stage training process |
| **Stages** | 1) New input embeds, 2) New output embeds, 3) Both, 4) All output, 5) New input + all output, 6) QLoRA all layers, 7) Cool-down |
| **Key Insight** | Proper initialization enables efficient adaptation |

### 5. Embedding Alignment (WECHSEL/CLP)

| Aspect | Details |
|--------|---------|
| **Process** | Use bilingual dictionaries to initialize Korean embeddings aligned with English |
| **Pros** | Faster convergence, leverages existing English knowledge |
| **Cons** | Requires quality bilingual lexicons, still needs some fine-tuning |
| **Best For** | When training data is limited but good dictionaries exist |

### 6. Prompt Tuning / In-Context Learning

| Aspect | Details |
|--------|---------|
| **Process** | Learn soft prompts or use few-shot examples to guide Korean processing |
| **Pros** | No model changes, immediate deployment, preserves all capabilities |
| **Cons** | Limited Korean fluency, depends on model's latent multilingual ability |
| **When to Use** | API-only access, quick testing, temporary solution |

---

## Hardware Requirements Summary (MedGemma 4B)

| Method | Training VRAM | Inference VRAM | Training Time (A5000) |
|--------|---------------|----------------|----------------------|
| Translation Layer | N/A | 4-8 GB | N/A |
| QLoRA Only | ~4.5 GB | ~4.5 GB | 1-2 days |
| Vocab Expansion + Embedding Training | ~10 GB | ~6 GB | 2-3 days |
| Full EEVE (7 stages) | 15-18 GB | ~6 GB | 5-7 days |
| Full Fine-tune (no quantization) | ~32 GB | ~16 GB | 3-5 days |

---

## Recommendation by Use Case

| Use Case | Recommended Strategy | Rationale |
|----------|---------------------|-----------|
| **Quick Prototype** | Translation Layer or Prompt Tuning | No training needed |
| **Limited Data (<100MB)** | LoRA/Adapters | Parameter-efficient, works with small data |
| **Production Korean Medical LLM** | EEVE-Style (Vocab Expansion + Staged Training) | Best quality-efficiency balance |
| **Maximum Quality** | Full Vocabulary Expansion + Continued Pretraining | Highest fluency, needs most resources |
| **Edge Deployment** | Knowledge Distillation | Create smaller specialized models |

---

## Critical Success Factors

1. **Embedding Initialization**: Never use random init - use subword decomposition or bilingual alignment
2. **Data Mixing**: Include 10% English data to prevent catastrophic forgetting
3. **Token Efficiency**: Vocabulary expansion reduces Korean tokens by ~3x
4. **Staged Training**: Progressive unfreezing stabilizes training
5. **Evaluation**: Use KorMedMCQA for Korean medical benchmarking
