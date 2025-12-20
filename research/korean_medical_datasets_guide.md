# Korean Medical Datasets & LLM Training Guide

A comprehensive guide to public Korean medical datasets, key papers, and training strategies for Korean medical LLMs.

---

## 1. Public Korean Medical Datasets

### 1-1. QA / Exam / Reasoning Style

#### KorMedMCQA
- **Source**: [Hugging Face](https://huggingface.co/datasets/sean0042/KorMedMCQA)
- **Description**: Multiple-choice QA from Korean healthcare licensing exams (의사/간호사/약사/치과의사, 2012-2024)
- **Size**: 7,469 questions
- **License**: MIT
- **Use for**:
  - Evaluation of Korean medical LLMs (standard benchmark)
  - Supervised finetuning / CoT finetuning
- **Important**: Keep 2024 exam questions as held-out test set (not in training data)

#### ChuGyouk Korean Medical Dataset Collection
- **Source**: [Hugging Face Collection](https://huggingface.co/collections/ChuGyouk/korean-medical-dataset-66318e62b80cd6ba89629042)
- **Includes**:
  - `medical-reasoning-train-kormedmcqa` (questions + rationales)
  - Korean medical O1 reasoning datasets
- **Use for**: Reasoning-style SFT (step-by-step explanation in Korean)

#### KorMedLawQA
- **Source**: [Hugging Face](https://huggingface.co/datasets/snuh/KorMedLawQA)
- **Description**: Multiple-choice QA focusing on Korean medical law (KMLE style)
- **License**: Apache-2.0
- **Use for**:
  - Safety / legal-awareness finetuning
  - Specialized evaluation of medico-legal reasoning

#### AI-Hub Healthcare QA Data (초거대 AI 헬스케어 질의응답 데이터)
- **Source**: [AI-Hub](https://aihub.or.kr/)
- **Description**: Large-scale Korean healthcare QA (질문 + 전문가 답변, 일부는 의료 문서 기반)
- **License**: AI-Hub public data (commercial use possible, check terms)
- **Use for**: Training medical chatbots

> **실전 사용 팁:**
> - KorMedMCQA / KorMedLawQA → 고난도 시험 문제 & 법
> - AI-Hub 헬스케어 QA → 일반 진료·상담 스타일 대화
> - 둘을 섞으면 "시험 + 실제 상담" 둘 다에 강한 LLM 만들기 좋습니다.

---

### 1-2. NER / De-identification / Clinical Entity

#### KBMC - Korean Bio-Medical Corpus (Medical NER)
- **Paper**: [arXiv:2403.16158](https://arxiv.org/abs/2403.16158)
- **Description**: First open Korean medical NER dataset
  - Entity types: disease, body part, treatment (BIO tagging)
  - 6,150 sentences, 153,971 tokens
  - 4,162 disease names, 841 body parts, 396 treatments
- **Performance**: ~20% improvement vs general Korean NER data
- **Use for**:
  - De-identification pipelines
  - Training medical token-level encoders (biLSTM-CRF, BERT token classifier)

#### Korean Clinical Entity Recognition
- **Description**: Dataset from online medical Q&A site
- **Entity types**: symptom/disease, test, treatment
- **Note**: Not always fully open, but methodology is documented

#### NER-based De-identification in Korean EMR
- **Note**: EMR data not public (privacy), but methodology useful for hospital data under IRB

---

### 1-3. General Corpora / Parallel Corpora with Medical Domain

For base pretraining / continued pretraining:

#### AI-Hub Healthcare Text Collections
| Dataset | Description | Use |
|---------|-------------|-----|
| 헬스케어 필수의료 의학지식 데이터 | Curated medical knowledge, KMLE-linked | Pretraining |
| 전문 의학지식 데이터 | Professional medical knowledge | Pretraining |
| 의료 분야 음성 데이터 | Patient-doctor dialogues (audio + transcripts) | Conversational corpus |
| 비대면 진료 음성 | Telemedicine conversations | Dialogue training |
| 응급실 임상 대화 데이터 | Emergency room conversations | Clinical dialogue |

#### 전문분야 한영 말뭉치 (Professional Domain KR-EN Corpus)
- **Size**: 1.5M sentence pairs across 8 domains
- **Medical/Health portion**: ~250k pairs
- **Use for**: Bilingual pretraining, medical MT

#### Open Korean Corpora Aggregators
- [Open-korean-corpora](https://github.com/ko-nlp/Open-korean-corpora) - Aggregates many Korean corpora with license info

---

### 1-4. Preference / Alignment Datasets (for RLHF/DPO)

#### KoMeP - Korean Medical Preference Dataset
- **Paper**: [PMC12086433](https://pmc.ncbi.nlm.nih.gov/articles/PMC12086433/)
- **Description**: First publicly available preference dataset for Korean medical LLMs
- **Size**: 5,551 preference pairs
- **Method**: Built from biomedical exam questions + multiple LLM responses, filtered using DAHL hallucination score
- **Use for**: DPO / RRHF / pairwise ranking-based alignment

#### Open-Korean-Instructions
- **Source**: [GitHub](https://github.com/ko-nlp/Open-korean-corpora)
- **Description**: Various open Korean instruction datasets (general + some medical)
- **Use for**: Mix with medical-specific data for general + medical specialization

---

### 1-5. Pretrained Models to Reuse

#### KM-BERT: Pre-trained BERT for Korean Medical NLP
- **Paper**: [Scientific Reports 2022](https://www.nature.com/articles/s41598-022-17806-8)
- **Training data**: 777k Korean medical sentences (textbooks, AI-Hub)
- **Use for**:
  - Encoder initialization
  - Tokenizer/embedding layer for Korean medical LLM

---

## 2. Key Korean Papers about Medical Corpora / LLMs

### Essential Reading:

1. **Korean Bio-Medical Corpus (KBMC) for Medical NER** - LREC-COLING 2024
   - [arXiv:2403.16158](https://arxiv.org/abs/2403.16158)
   - KBMC creation, annotation scheme, ~20% NER improvement
   - ChatGPT-assisted corpus creation methodology

2. **KorMedMCQA: Multi-Choice QA Benchmark** - 2024
   - [arXiv:2403.01469](https://arxiv.org/abs/2403.01469)
   - Exam extraction 2012-2024, dataset splits, correlation with MedQA (USMLE)

3. **Korean Medical Preference Dataset Construction** - 2025
   - [PMC12086433](https://pmc.ncbi.nlm.nih.gov/articles/PMC12086433/)
   - KoMeP and DAHL hallucination metric
   - Full pipeline for automatic preference pair creation

4. **High-Quality English-Korean Medical Corpus for LLMs** - 2025
   - Bilingual corpus from four Korean hospitals
   - Markdown+JSON format for LLM pretraining

5. **Pre-trained BERT for Korean Medical NLP** - Scientific Reports 2022
   - [Nature](https://www.nature.com/articles/s41598-022-17806-8)
   - 777k sentences from 28 specialties via AI-Hub

6. **Korean Clinical Entity Recognition using BERT** - 2020
   - Medical QA-derived NER dataset
   - BERT-based NER model

### Key Takeaways from Papers:
- **Text sources**: Textbooks, guidelines, AI-Hub, exams, clinical notes
- **Annotation methods**: NER, QA, preference, safety
- **Evaluation**: KorMedMCQA, KMLE-style exams

---

## 3. Training Pipeline for Korean Medical LLM

### 3-1. Base Corpus / Continued Pretraining

**Main Korean Medical Corpus:**
- AI-Hub 의료 텍스트 (필수/전문 의학지식, QA 원천 텍스트, 문서 기반 건강상식)
- 전문분야 한영 말뭉치 의료/보건 portion (~250k pairs)
- Additional EN-KO medical parallel corpora

**General Korean (for language ability):**
- Open Korean corpora (news, Wikipedia) from Open-korean-corpora

**Task**: Masked/Causal LM for continued pretraining of base model (LLaMA/Qwen/Gemma)

### 3-2. Supervised Medical Finetuning (SFT)

**Data Mix:**
| Type | Datasets |
|------|----------|
| QA/Exam | KorMedMCQA (exclude 2024), AI-Hub 헬스케어 QA, KorMedLawQA |
| NER/Classification | KBMC, clinical entity corpora |
| General Instructions | Open-Korean-Instructions |

**Format Conversion:**
```
Input: 질문 + 선택지
Output: 정답 + 해설 (reasoning-augmented)
```

### 3-3. Alignment (DPO / RLHF)

1. Start from SFT model
2. Use **KoMeP** (preference pairs) for DPO/IPO/RRHF
3. Include safety/legality data from KorMedLawQA distribution

---

## 4. Privacy & License Considerations

### Important Notes:

- **Real EMR/Hospital notes** (e.g., SNUH 38M-document corpus): Requires IRB + data-use agreements
- **Open corpora**: Always check:
  - License (Apache-2.0, CC-BY, AI-Hub 약관)
  - Commercial use permissions
  - Attribution requirements

### License Quick Reference:

| Dataset | License | Commercial |
|---------|---------|------------|
| KorMedMCQA | MIT | Yes |
| KorMedLawQA | Apache-2.0 | Yes |
| KBMC | Check paper | Check |
| AI-Hub data | AI-Hub terms | Check each |
| KoMeP | Check paper | Check |

---

## 5. Recommended Data Splits

### For Training Korean Medical LLM:

**Evaluation (Hold-out):**
- KorMedMCQA 2024 exam questions (all configs)
- Subset of KorMedLawQA test

**Training:**
- KorMedMCQA 2012-2023
- All other datasets

**Validation:**
- 5-10% of training data
- Small portion of KorMedMCQA 2023

---

## 6. Dataset Download Status (This Project)

**Total Downloaded: 13.4M+ examples**

### Medical-Specific Datasets (886K examples):
- [x] KorMedMCQA (ALL configs: doctor/dentist/nurse/pharm) - 7,489 examples
  - Training (2012-2023): 6,525 examples
  - **Evaluation holdout (2024): 964 examples** - DO NOT use in training
- [x] KorMedLawQA (medical law QA) - 13,388 examples
- [x] Medical Reasoning KorMedMCQA (with rationales) - 8,751 examples
- [x] Medical O1 Reasoning Korean - 25,682 examples
- [x] KMMLU medical subjects - 31,469 examples
- [x] Chinese Medical Dialogue - 799,743 examples (reference)
- [x] Korean Medical Dispute Cases - 5 examples

### Korean General Datasets (12.5M+ examples):
- [x] Korean Wikipedia - 500K articles
- [x] NamuWiki - 6.2M articles
- [x] Korean Textbooks - 4.4M examples
- [x] Open-Korean-Instructions - 375K examples
- [x] KoAlpaca variants - 70K examples
- [x] KULLM v2 - 152K examples
- [x] UltraChat 200K - 515K examples
- [x] KLUE (Korean NLU benchmark) - 205K examples

### Preference/DPO Datasets:
- [x] Korean Ultrafeedback - 61,966 examples (prompt/chosen/rejected)

### Pending:
- [ ] OSCAR Korean (requires HuggingFace access approval)
- [ ] AI-Hub datasets (requires AI-Hub account registration)
- [ ] KoMeP preference dataset (may need direct download from paper)

### Model:
- [x] MedGemma-4b-it downloaded (~8GB)
- [x] MedGemma-27b-it available (for inference with high VRAM)

---

## References

1. KorMedMCQA - https://huggingface.co/datasets/sean0042/KorMedMCQA
2. KBMC - https://arxiv.org/abs/2403.16158
3. KM-BERT - https://www.nature.com/articles/s41598-022-17806-8
4. KoMeP - https://pmc.ncbi.nlm.nih.gov/articles/PMC12086433/
5. Open-korean-corpora - https://github.com/ko-nlp/Open-korean-corpora
6. AI-Hub - https://aihub.or.kr/
