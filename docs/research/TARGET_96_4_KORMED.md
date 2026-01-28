# Target: 96.4% on Korean Medical Licensing Examination

## Benchmark Reference

**Source:** [Medigate News - KMed.ai Achievement](https://medigatenews.com/news/1991840877)

| Model | Score | Institution | Date |
|-------|-------|-------------|------|
| **KMed.ai** | **96.4%** | Naver + Seoul National University Hospital | 2025 |
| GPT-4 | < 96.4% | OpenAI | - |
| MedGemma Korean (Ours) | **Target: ≥96.4%** | - | - |

## KMed.ai Key Features

1. **Sovereign AI**: Independent from overseas general-purpose AI
2. **Clinical Terminology**: Reflects actual Korean clinical standards
3. **Medical Staff Participation**: SNUH doctors directly involved in development
4. **Foundation for Medical AGI**: EMR comprehension, medical reasoning, clinical judgment

## Our Strategy to Reach 96.4%

### Current Pipeline

```
Phase 0: Plain Text Pre-training (Korean language)
    ↓
Phase 1: Medical Dictionary + Translation
    ↓
Phase 2: MCQ with Reasoning (KorMedMCQA)
    ↓
Target: ≥96.4% on KMLE-equivalent test
```

### Key Differences from KMed.ai

| Aspect | KMed.ai | MedGemma Korean |
|--------|---------|-----------------|
| Base Model | Proprietary (Naver) | MedGemma (Google) |
| Training Data | SNUH Clinical Data | Public Korean Medical Datasets |
| Development | Corporate + Hospital | Open Source |
| Approach | End-to-end Training | LoRA Fine-tuning |

### Required Improvements

#### 1. Data Quality Enhancement
- [ ] Add more KMLE-style questions
- [ ] Include clinical case studies
- [ ] Improve reasoning chain quality
- [ ] Add Korean clinical terminology coverage

#### 2. Training Strategy
- [ ] Multi-stage LoRA training (current approach)
- [ ] Reasoning format optimization
- [ ] Answer extraction accuracy
- [ ] Format score + correctness balance

#### 3. Evaluation Alignment
- [ ] Use actual KMLE questions (if available)
- [ ] Match KMed.ai evaluation criteria
- [ ] Ensure fair comparison methodology

## Scoring Formula (Updated)

Current scoring in `_mcq_evaluation.py`:

```python
# 1/3 format + 1/3 valid_char + 1/3 correctness
total_score = (format_score * 1/3) + (valid_char_score * 1/3) + (correctness_score * 1/3)
```

**Components:**
- `format_score`: Reasoning format compliance (0.0-1.0)
- `valid_char_score`: Extracted answer is A/B/C/D/E (0 or 1)
- `correctness_score`: Answer matches expected (0 or 1)

**Maximum possible:** 1.0 (100%)

## Milestones

| Milestone | Accuracy | Status |
|-----------|----------|--------|
| Baseline (MedGemma) | ~60% | Measured |
| After Phase 0 | ~65% | - |
| After Phase 1 | ~75% | - |
| After Phase 2 | ~85% | - |
| Loop Training | ~90% | Target |
| **Final Target** | **≥96.4%** | **Goal** |

## Dataset Requirements

### Current: KorMedMCQA
- 604 test samples
- Medical licensing exam style
- Korean language

### Additional Data Needed
1. **KMLE Past Exams**: Actual licensing exam questions
2. **Clinical Cases**: Real-world medical scenarios
3. **Textbook QA**: Korean medical textbook content
4. **SNUH-style Data**: Clinical terminology alignment

## Technical Approach

### 1. Progressive LoRA Training
```
LoRA_0 (Plain Text) → LoRA_1 (Medical Dict) → LoRA_2 (MCQ)
```

### 2. Reasoning Enhancement
```
reasoning:
facts:
- Key medical facts from question
candidates:
- A, B, C, D, E analysis
criteria:
- Medical accuracy, clinical relevance
analysis:
- Detailed reasoning
evaluation:
- 평가기준: ...
- 점수표: A=x%, B=y%, ...
- 근거요약: ...
answer:
X
```

### 3. Loop Training Until Target
```python
while accuracy < 96.4:
    for type in [type1, type2, type3, type4]:
        train(type, epochs=1)
    accuracy = evaluate_kormedmcqa()
```

## References

1. [KMed.ai News Article](https://medigatenews.com/news/1991840877)
2. [KorMedMCQA Dataset](https://huggingface.co/datasets/sean0042/KorMedMCQA)
3. [MedGemma Model](https://huggingface.co/google/medgemma-4b-it)

## Timeline

| Phase | Duration | Target |
|-------|----------|--------|
| Data Preparation | 1 week | Complete datasets |
| Phase 0-2 Training | 2 weeks | 85% accuracy |
| Loop Training | 1-2 weeks | 90% accuracy |
| Optimization | 2 weeks | **96.4% accuracy** |

## Success Criteria

- [ ] KorMedMCQA accuracy ≥ 96.4%
- [ ] Consistent reasoning format
- [ ] Valid answer extraction rate > 99%
- [ ] Reproducible results

---

*Last updated: 2025-01-28*
