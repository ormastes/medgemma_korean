# Korean Medical Data Sources Research

## Overview

This document catalogs available Korean medical datasets for training MedGemma Korean, organized by source and data type.

## Data Categories

### 1. MCQ (Multiple Choice Question) Datasets

| Dataset | HuggingFace ID | Size | Source Type | Use Case |
|---------|----------------|------|-------------|----------|
| KorMedMCQA | `sean0042/KorMedMCQA` | 7,469 | Original Korean | Evaluation benchmark |
| MedQA Korean | `ChuGyouk/MedQA` | 22.9K | Translation (solar-1-mini) | MCQ training |
| MedQA Evol Korean | `ChuGyouk/MedQA-Evol-Korean` | 51.8K | Translation | MCQ training |
| KorMedLawQA | `snuh/KorMedLawQA` | - | Synthetic (GPT-4o-mini) | Medical law MCQ |
| MedExpQA Korean | `ChuGyouk/MedExpQA-Kor` | 497 | Translation | MCQ with explanation |

### 2. Instruction/QA Datasets

| Dataset | HuggingFace ID | Size | Source Type | Use Case |
|---------|----------------|------|-------------|----------|
| KoMedInstruct-52k | `ChuGyouk/KoMedInstruct-52k` | 52K | Translation | Instruction tuning |
| GenMedGPT-5k-ko | `ChuGyouk/GenMedGPT-5k-ko` | 5.45K | Translation | General medical |
| HealthSearchQA-ko | `ChuGyouk/HealthSearchQA-ko` | 3.17K | Synthetic (GPT-4o) | Health search |
| AI Healthcare QA | `ChuGyouk/AI_healthcare_QA` | 12.1K | Synthetic (GPT-4o) | Healthcare QA |
| KorMedConceptsQA | `ChuGyouk/KorMedConceptsQA` | 73.2K | Curated | Medical concepts |

### 3. Reasoning Datasets

| Dataset | HuggingFace ID | Size | Source Type | Use Case |
|---------|----------------|------|-------------|----------|
| Medical O1 Reasoning Ko | `ChuGyouk/medical-o1-reasoning-SFT-Ko` | 25.7K | Synthetic | Chain-of-thought |
| Medical Reasoning KorMedMCQA | `ChuGyouk/medical-reasoning-train-kormedmcqa` | 8.75K | Synthetic (Gemini) | CoT training |
| ChainofDiagnosis-Ko | `ChuGyouk/ChainofDiagnosis-Ko` | 39.1K | Translation | Diagnosis reasoning |

### 4. Domain-Specific Datasets

| Dataset | HuggingFace ID | Size | Source Type | Use Case |
|---------|----------------|------|-------------|----------|
| Asan AMC Healthinfo | `ChuGyouk/Asan-AMC-Healthinfo` | 19.2K | Original Korean | Hospital health info |
| PubMedQA-test-Ko | `ChuGyouk/PubMedQA-test-Ko` | 500 | Translation | Biomedical QA |
| Chest Radiology EnKo | `ChuGyouk/chest_radiology_enko` | 244 | Synthetic (Gemini) | Radiology |
| Medical Question Pairs Ko | `ChuGyouk/medical_questions_pairs_ko` | 3.05K | Translation | Question similarity |

### 5. Multimodal Datasets

| Dataset | HuggingFace ID | Size | Source Type | Use Case |
|---------|----------------|------|-------------|----------|
| PubMedVision-EnKo | `ChuGyouk/PubMedVision-EnKo` | 1.29M | Translation | Medical images |

---

## Source Type Classification

### Original Korean
- Created from Korean sources (exams, hospitals, etc.)
- Highest quality for Korean medical context
- Examples: KorMedMCQA, Asan AMC Healthinfo

### Translation (Simple)
- Translated from English using MT systems
- May have translation artifacts
- Examples: MedQA Korean, KoMedInstruct-52k

### Synthetic (LLM Generated)
- Generated using GPT-4o, Gemini, Claude
- Variable quality, good for augmentation
- Examples: HealthSearchQA-ko, AI Healthcare QA

### Curated
- Human-edited or manually curated
- High quality, smaller scale
- Examples: KorMedConceptsQA

---

## Data Format Types

### MCQ Format
```json
{
  "question": "다음 중 고혈압의 위험 요인이 아닌 것은?",
  "options": {
    "A": "비만",
    "B": "운동",
    "C": "흡연",
    "D": "과도한 음주",
    "E": "고염식"
  },
  "answer": "B"
}
```

### Instruction Format
```json
{
  "instruction": "당뇨병의 주요 증상을 설명해주세요.",
  "input": "",
  "output": "당뇨병의 주요 증상으로는..."
}
```

### QA Format
```json
{
  "question": "고혈압 환자의 식이요법은?",
  "answer": "저염식이를 권장하며..."
}
```

### Reasoning Format
```json
{
  "question": "환자가 호흡곤란을 호소합니다...",
  "reasoning": "1. 증상 분석: ...\n2. 감별진단: ...",
  "answer": "폐렴을 의심해볼 수 있습니다."
}
```

---

## Current Data in Project

### Already Downloaded

| Source | Train | Verification | Type |
|--------|-------|--------------|------|
| KoMedInstruct_52k | 39,035 | 12,964 | Instruction |
| MedQA_Korean | 8,636 | 2,813 | MCQ |
| MedQA_Evol_Korean | 38,695 | 13,114 | MCQ |
| Medical_O1_Reasoning_Ko | 19,261 | 6,402 | Reasoning |
| Asan_AMC_Healthinfo | 14,314 | 4,636 | QA |
| HealthSearchQA_ko | 2,336 | 805 | QA |
| KorMedMCQA | 1,845 | 641 | MCQ (eval) |
| **Total** | **124,123** | **41,375** | - |

### Filtered MCQ Data

| Split | Samples | Notes |
|-------|---------|-------|
| Train | 54,023 | Valid A-E answers only |
| Verification | 18,199 | Valid A-E answers only |

---

## Recommended Data Organization

```
data/
├── by_source/
│   ├── kormedmcqa/           # Original Korean exam
│   ├── medqa_korean/         # Translated MedQA
│   ├── komedinstruct/        # Instruction tuning
│   ├── asan_amc/             # Hospital health info
│   ├── medical_o1_reasoning/ # Chain-of-thought
│   └── healthsearchqa/       # Health search
├── by_type/
│   ├── mcq/                  # All MCQ format
│   ├── instruction/          # All instruction format
│   ├── qa/                   # All QA format
│   └── reasoning/            # All reasoning format
└── by_quality/
    ├── original_korean/      # Native Korean sources
    ├── translation/          # Translated datasets
    └── synthetic/            # LLM-generated
```

---

## References

- [KorMedMCQA Paper](https://arxiv.org/abs/2403.01469)
- [ChuGyouk Korean Medical Dataset Collection](https://huggingface.co/collections/ChuGyouk/korean-medical-dataset-66bc742e78a0239bbfd56216)
- [KorMedLawQA](https://huggingface.co/datasets/snuh/KorMedLawQA)
