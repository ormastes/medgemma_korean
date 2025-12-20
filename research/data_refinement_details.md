# Data Refinement & Training Details

## Complete Data Flow

```
Raw Data (HuggingFace)
        ↓
    Download
        ↓
data/raw/by_source/{dataset}/
        ↓
    Refine (refine_by_category.py)
        ↓
data/refined/
├── category_a_token/     → train_category_a_token.py
└── category_b_generation/ → train_category_b_generation.py
```

---

## 1. MCQ Datasets → Category A (Token Answer)

### 1.1 KorMedMCQA (Original Korean)

**Source:** `sean0042/KorMedMCQA`
**Size:** 7,469 samples
**Quality:** ⭐⭐⭐ (Original Korean medical exam)

**Raw Format:**
```json
{
  "question": "65세 남성이 3일 전부터 발생한 흉통을 주소로 내원하였다. 심전도에서 V1-V4 유도에서 ST분절 상승이 관찰된다. 가장 적절한 진단은?",
  "option_a": "불안정 협심증",
  "option_b": "전벽 심근경색",
  "option_c": "하벽 심근경색",
  "option_d": "심낭염",
  "option_e": "폐색전증",
  "answer": "B"
}
```

**Refined Format (Category A):**
```json
{
  "prompt": "<|im_start|>system\n의료 객관식 문제입니다. 정답 알파벳만 답하세요 (A, B, C, D, E 중 하나).\n<|im_end|>\n<|im_start|>user\n65세 남성이 3일 전부터 발생한 흉통을 주소로 내원하였다. 심전도에서 V1-V4 유도에서 ST분절 상승이 관찰된다. 가장 적절한 진단은?\n\nA) 불안정 협심증\nB) 전벽 심근경색\nC) 하벽 심근경색\nD) 심낭염\nE) 폐색전증\n<|im_end|>\n<|im_start|>assistant\n",
  "completion": "B",
  "text": "...(prompt + completion)...",
  "answer": "B",
  "source": "kormedmcqa",
  "category": "A",
  "type": "mcq_token"
}
```

**Training Target:** Model outputs `B` (single letter)

---

### 1.2 MedQA Korean (Translation)

**Source:** `ChuGyouk/MedQA`
**Size:** 22,900 samples
**Quality:** ⭐⭐ (Translated from English)

**Raw Format:**
```json
{
  "question": "28세 여성이 복통과 설사를 호소합니다. 대장내시경에서 직장부터 시작하여 연속적인 점막 염증이 관찰됩니다. 가장 가능성 있는 진단은?",
  "options": {
    "A": "크론병",
    "B": "궤양성 대장염",
    "C": "감염성 대장염",
    "D": "허혈성 대장염",
    "E": "방사선 대장염"
  },
  "answer": "B"
}
```

**Refined Format (Category A):**
```json
{
  "prompt": "<|im_start|>system\n의료 객관식 문제입니다. 정답 알파벳만 답하세요 (A, B, C, D, E 중 하나).\n<|im_end|>\n<|im_start|>user\n28세 여성이 복통과 설사를 호소합니다...\n\nA) 크론병\nB) 궤양성 대장염\nC) 감염성 대장염\nD) 허혈성 대장염\nE) 방사선 대장염\n<|im_end|>\n<|im_start|>assistant\n",
  "completion": "B",
  "answer": "B",
  "source": "medqa_korean",
  "category": "A",
  "type": "mcq_token"
}
```

---

### 1.3 MedQA Evol Korean (Translation)

**Source:** `ChuGyouk/MedQA-Evol-Korean`
**Size:** 51,800 samples
**Quality:** ⭐⭐ (Evolved/augmented translation)

**Raw → Refined:** Same as MedQA Korean

---

### 1.4 KorMedLawQA (Synthetic)

**Source:** `snuh/KorMedLawQA`
**Size:** ~5,000 samples
**Quality:** ⭐⭐ (GPT-4o generated)

**Raw Format:**
```json
{
  "question": "의료법에 따르면 의료인이 진료기록부를 보존해야 하는 기간은?",
  "options": ["3년", "5년", "10년", "15년", "영구"],
  "answer": "C",
  "reasoning": "의료법 제22조에 따르면..."
}
```

**Refined Format (Category A):**
```json
{
  "prompt": "...<|im_start|>user\n의료법에 따르면 의료인이 진료기록부를 보존해야 하는 기간은?\n\nA) 3년\nB) 5년\nC) 10년\nD) 15년\nE) 영구\n<|im_end|>\n<|im_start|>assistant\n",
  "completion": "C",
  "answer": "C",
  "source": "kormedlawqa",
  "category": "A",
  "type": "mcq_token"
}
```

---

## 2. QA Datasets → Category B (Generation)

### 2.1 Asan AMC Healthinfo (Original Korean)

**Source:** `ChuGyouk/Asan-AMC-Healthinfo`
**Size:** 19,200 samples
**Quality:** ⭐⭐⭐ (Real hospital health info)

**Raw Format:**
```json
{
  "question": "당뇨병 환자의 식이요법은 어떻게 해야 하나요?",
  "answer": "당뇨병 환자의 식이요법은 다음과 같습니다:\n\n1. 규칙적인 식사: 하루 3끼를 일정한 시간에\n2. 탄수화물 조절: 흰 쌀밥보다 잡곡밥\n3. 섬유질 섭취: 채소와 과일\n4. 단 음식 제한: 설탕, 꿀, 음료수\n5. 적절한 단백질: 살코기, 생선, 두부\n\n자세한 내용은 담당 의료진과 상담하세요."
}
```

**Refined Format (Category B):**
```json
{
  "prompt": "<|im_start|>system\n당신은 한국어 의료 전문 AI입니다. 정확하고 상세하게 답변하세요.\n<|im_end|>\n<|im_start|>user\n당뇨병 환자의 식이요법은 어떻게 해야 하나요?\n<|im_end|>\n<|im_start|>assistant\n",
  "completion": "당뇨병 환자의 식이요법은 다음과 같습니다:\n\n1. 규칙적인 식사: 하루 3끼를 일정한 시간에\n2. 탄수화물 조절: 흰 쌀밥보다 잡곡밥\n3. 섬유질 섭취: 채소와 과일\n4. 단 음식 제한: 설탕, 꿀, 음료수\n5. 적절한 단백질: 살코기, 생선, 두부\n\n자세한 내용은 담당 의료진과 상담하세요.",
  "text": "...(prompt + completion + <|im_end|>)...",
  "source": "asan_amc_healthinfo",
  "category": "B",
  "type": "qa_generation"
}
```

**Training Target:** Model generates full text response

---

### 2.2 HealthSearchQA Korean (Synthetic)

**Source:** `ChuGyouk/HealthSearchQA-ko`
**Size:** 3,170 samples
**Quality:** ⭐⭐ (GPT-4o generated)

**Raw Format:**
```json
{
  "question": "감기와 독감의 차이점은 무엇인가요?",
  "answer": "감기와 독감은 모두 호흡기 질환이지만 원인과 증상에 차이가 있습니다.\n\n감기:\n- 원인: 리노바이러스 등 200종 이상\n- 증상: 콧물, 재채기, 가벼운 기침\n- 경과: 1주일 내 자연 호전\n\n독감(인플루엔자):\n- 원인: 인플루엔자 바이러스\n- 증상: 고열, 심한 근육통, 두통\n- 경과: 1-2주, 합병증 위험"
}
```

**Refined Format (Category B):** Same structure as Asan AMC

---

### 2.3 AI Healthcare QA (Synthetic)

**Source:** `ChuGyouk/AI_healthcare_QA`
**Size:** 12,100 samples
**Quality:** ⭐⭐ (GPT-4o generated)

**Raw → Refined:** Same as QA format

---

### 2.4 KorMedConceptsQA (Curated)

**Source:** `ChuGyouk/KorMedConceptsQA`
**Size:** 73,200 samples
**Quality:** ⭐⭐⭐ (Human curated)

**Raw → Refined:** Same as QA format

---

## 3. Instruction Datasets → Category B (Generation)

### 3.1 KoMedInstruct-52k (Translation)

**Source:** `ChuGyouk/KoMedInstruct-52k`
**Size:** 52,000 samples
**Quality:** ⭐⭐ (Translated instruction data)

**Raw Format:**
```json
{
  "instruction": "고혈압의 원인과 예방법을 설명하세요.",
  "input": "",
  "output": "고혈압의 원인:\n\n1. 유전적 요인: 가족력이 있는 경우 위험 증가\n2. 생활습관: 고염식, 비만, 운동 부족\n3. 스트레스: 만성 스트레스로 인한 혈압 상승\n4. 나이: 나이가 들수록 혈관 탄력 감소\n\n예방법:\n\n1. 저염식: 하루 소금 6g 이하\n2. 규칙적 운동: 주 150분 유산소 운동\n3. 적정 체중 유지: BMI 18.5-24.9\n4. 금연 및 절주\n5. 스트레스 관리"
}
```

**Refined Format (Category B):**
```json
{
  "prompt": "<|im_start|>system\n당신은 한국어 의료 전문 AI입니다. 지시에 따라 답변하세요.\n<|im_end|>\n<|im_start|>user\n고혈압의 원인과 예방법을 설명하세요.\n<|im_end|>\n<|im_start|>assistant\n",
  "completion": "고혈압의 원인:\n\n1. 유전적 요인...\n\n예방법:\n\n1. 저염식...",
  "text": "...(prompt + completion + <|im_end|>)...",
  "source": "komedinstruct_52k",
  "category": "B",
  "type": "instruction_generation"
}
```

---

### 3.2 GenMedGPT-5k-ko (Translation)

**Source:** `ChuGyouk/GenMedGPT-5k-ko`
**Size:** 5,450 samples
**Quality:** ⭐ (Simple translation)

**Raw → Refined:** Same as instruction format

---

## 4. Reasoning Datasets → Category B (Generation)

### 4.1 Medical O1 Reasoning Korean (Synthetic)

**Source:** `ChuGyouk/medical-o1-reasoning-SFT-Ko`
**Size:** 25,700 samples
**Quality:** ⭐⭐ (Synthetic reasoning)

**Raw Format:**
```json
{
  "question": "45세 여성이 2주간의 피로감과 체중 증가를 호소합니다. 혈액검사에서 TSH 상승, T4 감소가 관찰됩니다. 진단과 치료는?",
  "reasoning": "단계별 분석:\n\n1. 환자 정보: 45세 여성, 중년\n2. 주증상: 피로감, 체중 증가 (대사 저하 의심)\n3. 검사 소견:\n   - TSH 상승: 뇌하수체가 갑상선 자극 증가\n   - T4 감소: 갑상선 호르몬 생산 저하\n4. 이 패턴은 일차성 갑상선기능저하증의 전형적 소견\n5. 가장 흔한 원인: 하시모토 갑상선염 (자가면역)",
  "answer": "진단: 일차성 갑상선기능저하증 (하시모토 갑상선염 가능성)\n치료: 레보티록신 (갑상선 호르몬 보충)"
}
```

**Refined Format (Category B):**
```json
{
  "prompt": "<|im_start|>system\n당신은 의료 전문 AI입니다. 단계별로 추론하여 답변하세요.\n<|im_end|>\n<|im_start|>user\n45세 여성이 2주간의 피로감과 체중 증가를 호소합니다...\n<|im_end|>\n<|im_start|>assistant\n",
  "completion": "단계별 분석:\n\n1. 환자 정보: 45세 여성, 중년\n2. 주증상: 피로감, 체중 증가...\n\n결론: 진단: 일차성 갑상선기능저하증...",
  "text": "...(prompt + completion + <|im_end|>)...",
  "source": "medical_o1_reasoning_ko",
  "category": "B",
  "type": "reasoning_generation"
}
```

---

### 4.2 ChainofDiagnosis Korean (Translation)

**Source:** `ChuGyouk/ChainofDiagnosis-Ko`
**Size:** 39,100 samples
**Quality:** ⭐⭐ (Translated reasoning)

**Raw Format:**
```json
{
  "question": "58세 남성, 갑작스러운 심한 두통, 구토, 의식 저하. CT에서 지주막하 출혈 소견.",
  "chain": "1. 주증상 분석: 갑작스러운 심한 두통 ('내 생애 최악의 두통') - 혈관성 원인 의심\n2. 동반 증상: 구토, 의식 저하 - 뇌압 상승 시사\n3. CT 소견: 지주막하 출혈 확인\n4. 가장 흔한 원인: 뇌동맥류 파열 (85%)\n5. 긴급성: 응급 상황, 재출혈 위험 높음",
  "diagnosis": "뇌동맥류 파열에 의한 지주막하 출혈"
}
```

**Refined Format (Category B):**
```json
{
  "prompt": "<|im_start|>system\n당신은 의료 전문 AI입니다. 단계별로 추론하여 답변하세요.\n<|im_end|>\n<|im_start|>user\n58세 남성, 갑작스러운 심한 두통, 구토, 의식 저하...\n<|im_end|>\n<|im_start|>assistant\n",
  "completion": "1. 주증상 분석: 갑작스러운 심한 두통...\n\n결론: 뇌동맥류 파열에 의한 지주막하 출혈",
  "source": "chainofdiagnosis_ko",
  "category": "B",
  "type": "reasoning_generation"
}
```

---

## 5. Training Configuration Summary

### Category A: Token Answer Training

| Parameter | Value | Reason |
|-----------|-------|--------|
| **System Prompt** | "정답 알파벳만 답하세요" | Force single letter |
| **max_new_tokens** | 3 | Only need 1 letter |
| **do_sample** | False | Greedy decoding |
| **Learning Rate** | 1e-4 | Higher for classification |
| **Batch Size** | 4 | More samples per step |
| **max_length** | 512 | Short context |
| **Evaluation** | Exact match accuracy | Compare predicted vs expected |
| **Target** | ≥90% | High accuracy |

### Category B: Generation Training

| Parameter | Value | Reason |
|-----------|-------|--------|
| **System Prompt** | "상세하게 답변하세요" | Full response |
| **max_new_tokens** | 512 | Long generation |
| **do_sample** | True | Sampling for diversity |
| **Learning Rate** | 2e-5 | Lower for generation |
| **Batch Size** | 1-2 | Longer sequences |
| **max_length** | 1024 | Longer context |
| **Evaluation** | Perplexity | Lower is better |
| **Target** | <3.0 | Good fluency |

---

## 6. Complete Dataset Summary Table

| Dataset | Size | Category | Output Type | System Prompt |
|---------|------|----------|-------------|---------------|
| KorMedMCQA | 7.5K | A | Letter | "정답 알파벳만" |
| MedQA Korean | 22.9K | A | Letter | "정답 알파벳만" |
| MedQA Evol | 51.8K | A | Letter | "정답 알파벳만" |
| KorMedLawQA | 5K | A | Letter | "정답 알파벳만" |
| **Category A Total** | **~87K** | - | - | - |
| Asan AMC | 19.2K | B | Full text | "상세하게 답변" |
| HealthSearchQA | 3.2K | B | Full text | "상세하게 답변" |
| AI Healthcare QA | 12.1K | B | Full text | "상세하게 답변" |
| KorMedConceptsQA | 73.2K | B | Full text | "상세하게 답변" |
| KoMedInstruct | 52K | B | Full text | "지시에 따라" |
| GenMedGPT | 5.5K | B | Full text | "지시에 따라" |
| Medical O1 | 25.7K | B | Reasoning | "단계별 추론" |
| ChainofDiagnosis | 39.1K | B | Reasoning | "단계별 추론" |
| Med Reasoning | 8.8K | B | Reasoning | "단계별 추론" |
| **Category B Total** | **~239K** | - | - | - |

---

## 7. Training Pipeline Commands

```bash
# Step 1: Download all datasets
python scripts/download_all_datasets.py

# Step 2: Refine into categories
python refine_scripts/refine_by_category.py

# Step 3: Train Category B (Generation) - Foundation
python scripts/train_category_b_generation.py \
    --model medgemma-27b \
    --epochs 3 \
    --target-perplexity 3.0

# Step 4: Train Category A (Token) - Fine-tune for accuracy
python scripts/train_category_a_token.py \
    --model medgemma-27b \
    --epochs 10 \
    --target-accuracy 90.0
```
