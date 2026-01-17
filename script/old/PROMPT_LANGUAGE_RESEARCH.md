# Prompt Language Research for Korean LLM Training

## Research Question
When fine-tuning an LLM for Korean medical content, should prompts/instructions be in Korean (target language) or English (original/dominant language)?

---

## Key Research Findings

### 1. Native vs Non-Native Prompting Study (2024)
**Source:** [Native vs Non-Native Language Prompting: A Comparative Analysis](https://arxiv.org/html/2409.07054v1)

**Finding:** English prompts outperformed native (Arabic) prompts across all models tested, including Arabic-centric LLMs.

| Prompt Type | Performance |
|-------------|-------------|
| English (non-native) | Best |
| Mixed (native instruction + English labels) | Second |
| Native | Lowest |

**Why:** Models trained predominantly on English have stronger capabilities in English, even when adapted to other languages.

---

### 2. Korean Instruction Toolkit (KIT-19)
**Source:** [KIT-19: Korean Instruction Toolkit](https://arxiv.org/html/2403.16444v1)

**Finding:** Native Korean instruction data is superior to machine-translated English datasets.

**Key Points:**
- Translated datasets lose Korean cultural and linguistic nuances
- KIT-19 created 19 native Korean NLP tasks (5,000 examples each)
- Native instructions capture cultural features better

---

### 3. Content-Prompt Language Matching Study (2025)
**Source:** [Why Your LLM Prompts Should Match Your Content Language](https://ryanstenhouse.dev/why-your-llm-prompts-should-match-your-content-language/)

**Finding:** Matching prompt language to content language consistently outperforms "translate to English" approach.

> "The assumption that using English prompts is better for extracting data in any language isn't just wrong; it's measurably worse."

**Why:** Cross-lingual mismatch between prompt and content hurts accuracy.

---

### 4. Multilingual Chain-of-Thought Study
**Source:** [Multilingual Prompt Engineering Survey](https://arxiv.org/html/2505.11665v1)

**Finding:** Native-CoT outperforms English-CoT across multiple languages.

| Method | Performance |
|--------|-------------|
| Regressive Native-CoT | Better |
| Regressive English-CoT | Worse |

---

### 5. Thunder-LLM Korean Adaptation (2025)
**Source:** [Thunder-LLM: Adapting LLMs to Korean](https://arxiv.org/html/2506.21595v1)

**Approach:** Uses bilingual data (Korean + English) for pretraining, then Korean instruction tuning.

**Key Insight:**
- Vocabulary expansion for Korean expressiveness
- High-quality Korean instruction dataset crucial
- Bilingual alignment during pretraining helps

---

## Summary Table

| Aspect | English Prompt | Korean Prompt | Winner |
|--------|---------------|---------------|--------|
| **Inference on English-centric models** | Strong | Weak | English |
| **Training data quality** | Loses nuances | Preserves culture | Korean |
| **Content in Korean** | Cross-lingual mismatch | Natural alignment | Korean |
| **Chain-of-thought reasoning** | Lower accuracy | Higher accuracy | Korean |
| **Model after Korean fine-tuning** | May cause code-switching | More consistent | Korean |

---

## Recommendation for MedGemma Korean

### Context
- **Base model:** MedGemma (English-centric, medical domain)
- **Target:** Korean medical content
- **Goal:** 90% accuracy on KorMedMCQA

### Recommended Strategy: **Hybrid Approach**

#### For Training Data (Instructions):
| Phase | Prompt Language | Ratio | Reason |
|-------|-----------------|-------|--------|
| Phase 1: Vocabulary/Embedding | N/A | - | Plain text only |
| Phase 2: Medical Dictionary | English | 100% | Simple term definitions |
| Phase 3: MCQ Reasoning | **Korean** | 95% | Content is Korean, preserve alignment |
| Phase 3: MCQ Detailed | Korean | 5% | Format instruction in target language |

#### Why Korean for MCQ Training:
1. **Content alignment:** Questions and answers are in Korean
2. **Avoid code-switching:** English prompts may cause mixed language outputs
3. **Cultural context:** Medical terms have Korean-specific meanings
4. **Native reasoning:** CoT in Korean performs better for Korean content

#### Why English for Dictionary:
1. **Simple format:** "Meaning of word X: Y" is language-agnostic
2. **Bilingual definitions:** Some terms have English equivalents
3. **Base model strength:** Leverages MedGemma's English capabilities

---

## Alternative Strategies to Consider

### Option A: Full Korean
- All prompts in Korean
- Most consistent, avoids code-switching
- May underutilize base model's English strength

### Option B: Full English
- All prompts in English
- Leverages base model capabilities
- Risk: Cross-lingual mismatch, code-switching

### Option C: Curriculum (Recommended)
- Start with English prompts (leverage base model)
- Gradually shift to Korean prompts
- Final fine-tuning with Korean-only

### Option D: Mixed Per-Sample
- Random mix of English/Korean prompts
- Teaches model to handle both
- May be confusing during training

---

## Code-Switching Prevention

Common problem: Model responds in wrong language or mixes languages.

**Solutions:**
1. Use consistent prompt language during training
2. Include language tag in prompt: `[Korean response required]`
3. Add Korean-only validation during training
4. Penalize non-Korean characters in output

---

## Final Recommendation

For your MedGemma Korean training:

```
train_01 (Dictionary): English prompts
  - "Meaning of word {term}:"
  - Simpler task, bilingual acceptable

train_02 (MCQ): Korean prompts
  - "<reasoning> 안에서 단계별로 분석하고 정답 알파벳을 답하세요."
  - Content is Korean, reasoning should match
```

This hybrid approach:
- Preserves base model English strength for simple tasks
- Uses Korean for complex reasoning where content alignment matters
- Reduces code-switching risk for medical Q&A

---

## Sources

1. [Native vs Non-Native Language Prompting](https://arxiv.org/html/2409.07054v1) - Arabic study
2. [KIT-19: Korean Instruction Toolkit](https://arxiv.org/html/2403.16444v1) - Native Korean data
3. [Thunder-LLM](https://arxiv.org/html/2506.21595v1) - Korean LLM adaptation
4. [Multilingual Prompt Engineering Survey](https://arxiv.org/html/2505.11665v1) - Cross-lingual study
5. [Why Prompts Should Match Content Language](https://ryanstenhouse.dev/why-your-llm-prompts-should-match-your-content-language/) - Practical analysis
6. [Navigating Korean LLM Research](https://huggingface.co/blog/amphora/navigating-ko-llm-research-1) - Korean LLM overview
7. [Optimizing Language Augmentation for Korean](https://arxiv.org/html/2403.10882v1) - Korean optimization study
