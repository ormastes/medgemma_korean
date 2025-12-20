# Korean Medical Data Format & Training Guide

## 1. Data Sources Overview

| Dataset | HuggingFace ID | Size | Source Quality | Priority |
|---------|----------------|------|----------------|----------|
| KorMedMCQA | `sean0042/KorMedMCQA` | 7.5K | Original Korean | ⭐⭐⭐ |
| MedQA Korean | `ChuGyouk/MedQA` | 22.9K | Translation | ⭐⭐ |
| MedQA Evol Korean | `ChuGyouk/MedQA-Evol-Korean` | 51.8K | Translation | ⭐⭐ |
| KorMedLawQA | `snuh/KorMedLawQA` | - | Synthetic (GPT-4o) | ⭐⭐ |
| KoMedInstruct-52k | `ChuGyouk/KoMedInstruct-52k` | 52K | Translation | ⭐⭐ |
| GenMedGPT-5k-ko | `ChuGyouk/GenMedGPT-5k-ko` | 5.5K | Translation | ⭐ |
| HealthSearchQA-ko | `ChuGyouk/HealthSearchQA-ko` | 3.2K | Synthetic (GPT-4o) | ⭐⭐ |
| AI Healthcare QA | `ChuGyouk/AI_healthcare_QA` | 12.1K | Synthetic (GPT-4o) | ⭐⭐ |
| Asan AMC Healthinfo | `ChuGyouk/Asan-AMC-Healthinfo` | 19.2K | Original Korean | ⭐⭐⭐ |
| KorMedConceptsQA | `ChuGyouk/KorMedConceptsQA` | 73.2K | Curated | ⭐⭐⭐ |
| Medical O1 Reasoning Ko | `ChuGyouk/medical-o1-reasoning-SFT-Ko` | 25.7K | Synthetic | ⭐⭐ |
| Medical Reasoning KorMedMCQA | `ChuGyouk/medical-reasoning-train-kormedmcqa` | 8.8K | Synthetic (Gemini) | ⭐⭐ |
| ChainofDiagnosis-Ko | `ChuGyouk/ChainofDiagnosis-Ko` | 39.1K | Translation | ⭐⭐ |

---

## 2. Data Types & Raw Formats

### 2.1 MCQ (Multiple Choice Question)

**Datasets:** KorMedMCQA, MedQA Korean, MedQA Evol Korean, KorMedLawQA

**Raw Format:**
```json
{
  "question": "다음 중 심근경색의 초기 증상이 아닌 것은?",
  "options": {
    "A": "흉통",
    "B": "호흡곤란",
    "C": "좌측 팔 통증",
    "D": "두통",
    "E": "발한"
  },
  "answer": "D"
}
```

**Alternative Format (KorMedMCQA):**
```json
{
  "question": "...",
  "option_a": "흉통",
  "option_b": "호흡곤란",
  "option_c": "좌측 팔 통증",
  "option_d": "두통",
  "option_e": "발한",
  "answer": "D"
}
```

---

### 2.2 Instruction

**Datasets:** KoMedInstruct-52k, GenMedGPT-5k-ko

**Raw Format:**
```json
{
  "instruction": "당뇨병의 주요 증상을 설명해주세요.",
  "input": "",
  "output": "당뇨병의 주요 증상으로는 다갈증(갈증 증가), 다뇨증(소변량 증가), 다식증(식욕 증가), 체중 감소, 피로감 등이 있습니다..."
}
```

---

### 2.3 QA (Question-Answer)

**Datasets:** HealthSearchQA-ko, AI Healthcare QA, Asan AMC Healthinfo, KorMedConceptsQA

**Raw Format:**
```json
{
  "question": "고혈압 환자가 피해야 할 음식은?",
  "answer": "고혈압 환자는 나트륨 함량이 높은 음식을 피해야 합니다. 특히 절임류, 가공식품, 라면, 젓갈 등을 줄이고..."
}
```

---

### 2.4 Reasoning (Chain-of-Thought)

**Datasets:** Medical O1 Reasoning Ko, Medical Reasoning KorMedMCQA, ChainofDiagnosis-Ko

**Raw Format:**
```json
{
  "question": "65세 남성이 가슴 통증과 호흡곤란을 호소합니다. ECG에서 ST 분절 상승이 관찰됩니다. 가장 가능성 높은 진단은?",
  "reasoning": "1. 환자 정보 분석: 65세 남성, 심혈관 위험군\n2. 증상 분석: 가슴 통증 + 호흡곤란 = 심장 관련 가능성\n3. ECG 소견: ST 분절 상승 = 급성 심근경색의 전형적 소견\n4. 감별진단: STEMI vs NSTEMI vs 불안정 협심증",
  "answer": "ST분절 상승 심근경색 (STEMI)"
}
```

---

## 3. Refined Format (Training Format)

All data types are converted to **ChatML format** for training:

### 3.1 MCQ Refined Format

```
<|im_start|>system
당신은 한국어 의료 전문 AI 어시스턴트입니다.
<|im_end|>
<|im_start|>user
다음 중 심근경색의 초기 증상이 아닌 것은?

A) 흉통
B) 호흡곤란
C) 좌측 팔 통증
D) 두통
E) 발한
<|im_end|>
<|im_start|>assistant
정답은 D입니다.
<|im_end|>
```

**Stored Fields:**
| Field | Description |
|-------|-------------|
| `text` | Full ChatML formatted text |
| `question` | Original question |
| `correct_answer` | Answer letter (A-E) |
| `source` | Dataset name |
| `type` | "mcq" |

---

### 3.2 Instruction Refined Format

```
<|im_start|>system
당신은 한국어 의료 전문 AI 어시스턴트입니다.
<|im_end|>
<|im_start|>user
당뇨병의 주요 증상을 설명해주세요.
<|im_end|>
<|im_start|>assistant
당뇨병의 주요 증상으로는 다갈증(갈증 증가), 다뇨증(소변량 증가), 다식증(식욕 증가), 체중 감소, 피로감 등이 있습니다...
<|im_end|>
```

**Stored Fields:**
| Field | Description |
|-------|-------------|
| `text` | Full ChatML formatted text |
| `instruction` | Original instruction |
| `output` | Response |
| `source` | Dataset name |
| `type` | "instruction" |

---

### 3.3 QA Refined Format

```
<|im_start|>system
당신은 한국어 의료 전문 AI 어시스턴트입니다.
<|im_end|>
<|im_start|>user
고혈압 환자가 피해야 할 음식은?
<|im_end|>
<|im_start|>assistant
고혈압 환자는 나트륨 함량이 높은 음식을 피해야 합니다...
<|im_end|>
```

**Stored Fields:**
| Field | Description |
|-------|-------------|
| `text` | Full ChatML formatted text |
| `question` | Original question |
| `answer` | Response |
| `source` | Dataset name |
| `type` | "qa" |

---

### 3.4 Reasoning Refined Format

```
<|im_start|>system
당신은 한국어 의료 전문 AI 어시스턴트입니다. 단계별로 추론하여 답변하세요.
<|im_end|>
<|im_start|>user
65세 남성이 가슴 통증과 호흡곤란을 호소합니다. ECG에서 ST 분절 상승이 관찰됩니다. 가장 가능성 높은 진단은?
<|im_end|>
<|im_start|>assistant
1. 환자 정보 분석: 65세 남성, 심혈관 위험군
2. 증상 분석: 가슴 통증 + 호흡곤란 = 심장 관련 가능성
3. ECG 소견: ST 분절 상승 = 급성 심근경색의 전형적 소견
4. 감별진단: STEMI vs NSTEMI vs 불안정 협심증

따라서, ST분절 상승 심근경색 (STEMI)
<|im_end|>
```

**Stored Fields:**
| Field | Description |
|-------|-------------|
| `text` | Full ChatML formatted text |
| `question` | Original question |
| `reasoning` | Step-by-step reasoning |
| `source` | Dataset name |
| `type` | "reasoning" |

---

## 4. Training Methods by Data Type

| Data Type | Training Method | Loss Function | Evaluation Metric | Target |
|-----------|-----------------|---------------|-------------------|--------|
| **MCQ** | SFT + Direct Answer | Cross-entropy | Accuracy (%) | 90% |
| **Instruction** | SFT | Cross-entropy | Perplexity | < 3.0 |
| **QA** | SFT | Cross-entropy | Perplexity | < 3.0 |
| **Reasoning** | SFT + CoT | Cross-entropy | Reasoning Quality | - |

---

## 5. Training Configuration by Type

### 5.1 MCQ Training

```python
config = {
    "target": "Direct answer prediction (A/B/C/D/E)",
    "system_prompt": "당신은 한국어 의료 전문 AI 어시스턴트입니다.",
    "evaluation": "Extract first letter, compare with correct_answer",
    "stop_condition": "accuracy >= 90%",
    "hyperparams": {
        "lr": 1e-4,  # Higher for direct answer
        "epochs": 10,
        "eval_steps": 500,
    }
}
```

### 5.2 Instruction Training

```python
config = {
    "target": "Follow instructions accurately",
    "system_prompt": "당신은 한국어 의료 전문 AI 어시스턴트입니다.",
    "evaluation": "Perplexity on validation set",
    "hyperparams": {
        "lr": 2e-5,  # Lower for generation
        "epochs": 3,
    }
}
```

### 5.3 QA Training

```python
config = {
    "target": "Accurate medical information",
    "system_prompt": "당신은 한국어 의료 전문 AI 어시스턴트입니다.",
    "evaluation": "Perplexity + manual review",
    "hyperparams": {
        "lr": 2e-5,
        "epochs": 3,
    }
}
```

### 5.4 Reasoning Training

```python
config = {
    "target": "Step-by-step medical reasoning",
    "system_prompt": "당신은 한국어 의료 전문 AI 어시스턴트입니다. 단계별로 추론하여 답변하세요.",
    "evaluation": "Reasoning coherence + final answer accuracy",
    "hyperparams": {
        "lr": 5e-5,
        "epochs": 5,
    }
}
```

---

## 6. Data Quality Levels

| Level | Source | Description | Weight |
|-------|--------|-------------|--------|
| **raw** | Downloaded | Original from HuggingFace | 1.0 |
| **refined** | Rule-based | Filtered + formatted | 1.2 |
| **reviewed** | DeepSeek LLM | Quality scored + filtered | 1.5 |

---

## 7. Pipeline Summary

```
┌─────────────────┐
│  HuggingFace    │
│  Datasets (13)  │
└────────┬────────┘
         │ download_all_datasets.py
         ▼
┌─────────────────┐
│   data/raw/     │
│   by_source/    │
└────────┬────────┘
         │ refine_all.py
         ▼
┌─────────────────┐
│  data/refined/  │
│  ├── mcq/       │
│  ├── qa/        │
│  ├── instruction│
│  └── reasoning/ │
└────────┬────────┘
         │ review_with_deepseek.py (optional)
         ▼
┌─────────────────┐
│  data/reviewed/ │
│  (Quality 30+)  │
└────────┬────────┘
         │ train_korean_medical.py
         ▼
┌─────────────────┐
│  Trained Model  │
│  (MedGemma-Ko)  │
└─────────────────┘
```

---

## 8. Recommended Training Order

| Phase | Data Type | Epochs | Purpose |
|-------|-----------|--------|---------|
| 1 | combined | 1-2 | General medical knowledge |
| 2 | qa + instruction | 2-3 | Response quality |
| 3 | reasoning | 3-5 | Chain-of-thought |
| 4 | mcq | 5-10 | Target 90% accuracy |

---

## 9. File Locations

```
medgemma_korean/
├── data/
│   ├── data_config.json          # Dataset configurations
│   ├── raw/by_source/            # Downloaded datasets
│   ├── refined/                  # Refined data by type
│   │   ├── mcq/train/
│   │   ├── qa/train/
│   │   ├── instruction/train/
│   │   ├── reasoning/train/
│   │   └── combined/train/
│   └── reviewed/                 # DeepSeek reviewed
├── refine_scripts/
│   ├── refine_all.py             # Rule-based refinement
│   ├── refine_mcq.py             # MCQ specific
│   └── review_with_deepseek.py   # LLM review
├── scripts/
│   ├── download_all_datasets.py  # Download datasets
│   ├── train_korean_medical.py   # Main training script
│   └── build_korean_medical_tokenizer.py
└── research/
    └── data_format_training_guide.md  # This document
```
