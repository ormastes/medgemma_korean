# Korean Medical Data Type Definition

## 1. Two Main Data Categories

### Category A: Token/Letter Answer (Classification)
- **Output:** Single letter (A-E) or single word
- **Training:** Direct prediction, no generation
- **Evaluation:** Exact match accuracy

### Category B: Conversation/Generation
- **Output:** Full text response
- **Training:** Sequence generation
- **Evaluation:** Perplexity, BLEU, human eval

---

## 2. Category A: Token Answer Data

### 2.1 MCQ Direct (Letter Answer)

**Input Format:**
```
<|im_start|>system
의료 객관식 문제입니다. 정답 알파벳만 답하세요.
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
```

**Output (Target):**
```
D
```

**Datasets:**
| Dataset | Samples | Answer Type |
|---------|---------|-------------|
| KorMedMCQA | 7.5K | Letter (A-E) |
| MedQA Korean | 22.9K | Letter (A-E) |
| MedQA Evol Korean | 51.8K | Letter (A-E) |
| KorMedLawQA | ~5K | Letter (A-E) |

---

### 2.2 Yes/No Classification

**Input Format:**
```
<|im_start|>system
의료 질문입니다. 예 또는 아니오로만 답하세요.
<|im_end|>
<|im_start|>user
당뇨병 환자가 인슐린 주사를 맞아야 하나요?
<|im_end|>
<|im_start|>assistant
```

**Output (Target):**
```
예
```

---

### 2.3 Single Word Classification

**Input Format:**
```
<|im_start|>system
의료 분류 질문입니다. 한 단어로만 답하세요.
<|im_end|>
<|im_start|>user
다음 증상의 원인 장기는? 증상: 황달, 복수, 간비대
<|im_end|>
<|im_start|>assistant
```

**Output (Target):**
```
간
```

---

## 3. Category B: Conversation/Generation Data

### 3.1 QA Generation

**Input Format:**
```
<|im_start|>system
당신은 한국어 의료 전문 AI입니다. 정확하고 상세하게 답변하세요.
<|im_end|>
<|im_start|>user
고혈압 환자가 피해야 할 음식은?
<|im_end|>
<|im_start|>assistant
```

**Output (Target):**
```
고혈압 환자는 다음 음식을 피해야 합니다:

1. 고나트륨 식품: 절임류, 젓갈, 라면
2. 가공식품: 햄, 소시지, 통조림
3. 알코올: 혈압 상승 유발
4. 카페인: 과다 섭취 시 혈압 상승

대신 신선한 과일, 채소, 저염식을 권장합니다.
```

**Datasets:**
| Dataset | Samples | Output Type |
|---------|---------|-------------|
| Asan AMC | 19.2K | Full text |
| HealthSearchQA | 3.2K | Full text |
| AI Healthcare QA | 12.1K | Full text |
| KorMedConceptsQA | 73.2K | Full text |

---

### 3.2 Instruction Following

**Input Format:**
```
<|im_start|>system
당신은 한국어 의료 전문 AI입니다. 지시에 따라 답변하세요.
<|im_end|>
<|im_start|>user
당뇨병의 주요 증상 5가지를 나열하세요.
<|im_end|>
<|im_start|>assistant
```

**Output (Target):**
```
당뇨병의 주요 증상 5가지:

1. 다갈증 (Polydipsia) - 갈증 증가
2. 다뇨증 (Polyuria) - 소변량 증가
3. 다식증 (Polyphagia) - 식욕 증가
4. 체중 감소 - 인슐린 저항성으로 인한
5. 피로감 - 에너지 대사 장애
```

**Datasets:**
| Dataset | Samples | Output Type |
|---------|---------|-------------|
| KoMedInstruct-52k | 52K | Full text |
| GenMedGPT-5k-ko | 5.5K | Full text |

---

### 3.3 Reasoning (Chain-of-Thought)

**Input Format:**
```
<|im_start|>system
당신은 의료 전문 AI입니다. 단계별로 추론하여 진단하세요.
<|im_end|>
<|im_start|>user
65세 남성, 흉통, 호흡곤란, ECG에서 ST분절 상승. 진단은?
<|im_end|>
<|im_start|>assistant
```

**Output (Target):**
```
단계별 분석:

1. 환자 정보: 65세 남성 (심혈관 고위험군)
2. 증상: 흉통 + 호흡곤란 (심장 관련 의심)
3. ECG 소견: ST분절 상승 (심근경색 특징)
4. 감별진단: STEMI vs NSTEMI

결론: ST분절 상승 심근경색 (STEMI)
```

**Datasets:**
| Dataset | Samples | Output Type |
|---------|---------|-------------|
| Medical O1 Reasoning | 25.7K | Reasoning + Answer |
| ChainofDiagnosis | 39.1K | Reasoning + Answer |
| Medical Reasoning KorMedMCQA | 8.8K | Reasoning + Answer |

---

## 4. Data Conversion from Existing Formats

### 4.1 Convert MCQ to Token Answer

**Before (Current):**
```json
{
  "text": "<|im_start|>...<|im_end|>\n<|im_start|>assistant\n정답은 D입니다.\n<|im_end|>",
  "correct_answer": "D"
}
```

**After (Token Answer):**
```json
{
  "prompt": "<|im_start|>system\n의료 객관식입니다. 정답 알파벳만 답하세요.\n<|im_end|>\n<|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n",
  "completion": "D",
  "answer": "D"
}
```

---

### 4.2 Convert QA to Check if Token or Generation

**Check Logic:**
```python
def classify_answer(answer_text):
    """Classify if answer should be token or generation."""
    answer = answer_text.strip()

    # Token answer cases
    if len(answer) == 1 and answer in "ABCDE":
        return "letter"
    if answer in ["예", "아니오", "네", "아니요"]:
        return "yes_no"
    if len(answer.split()) == 1 and len(answer) <= 10:
        return "single_word"

    # Generation cases
    return "generation"
```

---

## 5. Training Strategy by Category

### 5.1 Category A: Token Answer Training

| Aspect | Configuration |
|--------|---------------|
| **Objective** | Classification / Direct prediction |
| **Output Length** | max_new_tokens=5 |
| **Decoding** | do_sample=False (greedy) |
| **Evaluation** | Exact match accuracy |
| **Target** | 90%+ accuracy |
| **Loss** | Cross-entropy on answer token only |

**Training Code:**
```python
# For MCQ letter answer
sft_config = SFTConfig(
    max_length=512,
    # Only compute loss on completion (answer letter)
    dataset_kwargs={"skip_prompt": True},
)
```

---

### 5.2 Category B: Generation Training

| Aspect | Configuration |
|--------|---------------|
| **Objective** | Sequence generation |
| **Output Length** | max_new_tokens=512 |
| **Decoding** | temperature=0.7, top_p=0.9 |
| **Evaluation** | Perplexity, BLEU |
| **Target** | Perplexity < 3.0 |
| **Loss** | Cross-entropy on full response |

---

## 6. Summary Table

| Type | Category | System Prompt | Output | Evaluation |
|------|----------|---------------|--------|------------|
| MCQ | A: Token | "정답 알파벳만 답하세요" | A-E | Accuracy |
| Yes/No | A: Token | "예/아니오로만 답하세요" | 예/아니오 | Accuracy |
| Classification | A: Token | "한 단어로만 답하세요" | 단어 | Accuracy |
| QA | B: Gen | "상세하게 답변하세요" | 문장 | Perplexity |
| Instruction | B: Gen | "지시에 따라 답변하세요" | 문장 | Perplexity |
| Reasoning | B: Gen | "단계별로 추론하세요" | 추론+답 | Quality |

---

## 7. Training Scripts

### Category A: Token Answer
```bash
# Train MCQ letter prediction
python scripts/train_category_a_token.py \
    --model medgemma-27b \
    --epochs 10 \
    --target-accuracy 90.0

# Output: Single letter (A-E)
# Evaluation: Exact match accuracy
```

### Category B: Generation
```bash
# Train text generation
python scripts/train_category_b_generation.py \
    --model medgemma-27b \
    --epochs 3 \
    --target-perplexity 3.0

# Output: Full text response
# Evaluation: Perplexity
```

---

## 8. Recommended Training Order

```
Phase 1: Category B - Generation (Foundation)
├── Script: train_category_b_generation.py
├── Data: category_b_generation/
├── Epochs: 2-3
├── Output: Full text responses
└── Target: Perplexity < 3.0

Phase 2: Category A - Token (Fine-tune)
├── Script: train_category_a_token.py
├── Data: category_a_token/
├── Epochs: 5-10
├── Output: Letter (A-E)
└── Target: Accuracy ≥ 90%
```

---

## 9. Data Format Comparison

| Aspect | Category A | Category B |
|--------|------------|------------|
| **Output** | Single letter | Full text |
| **System Prompt** | "정답 알파벳만 답하세요" | "상세하게 답변하세요" |
| **max_new_tokens** | 3 | 512 |
| **Decoding** | Greedy | Sampling |
| **Evaluation** | Accuracy | Perplexity |
| **Learning Rate** | 1e-4 | 2e-5 |

---

## 10. File Locations

```
data/refined/
├── category_a_token/      # MCQ → Letter
│   ├── train/
│   └── validation/
└── category_b_generation/ # QA/Instruction → Text
    ├── train/
    └── validation/

scripts/
├── train_category_a_token.py      # Token prediction
├── train_category_b_generation.py # Text generation
└── download_all_datasets.py       # Data download

refine_scripts/
└── refine_by_category.py          # Data preparation
```
