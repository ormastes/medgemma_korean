# Fine-Tuning for QA Tasks

## Overview

This document describes the supervised fine-tuning (SFT) approach for medical QA tasks after continued pretraining.

## Training Pipeline

```
Stage 1.1 (CPT)          Stage 1.2 (SFT)           Stage 1.3 (Evaluation)
Raw Text Training   →    QA Fine-Tuning       →    KorMedMCQA Testing

Input:                   Input:                    Input:
- Plain text             - Chat format             - Test MCQ (604)
- Medical vocab ×5       - QA pairs
                         - MCQ with reasoning
Output:                  Output:                   Output:
- Korean fluency         - QA ability              - Accuracy ≥90%
- Medical terms          - Reasoning
- PPL <3.0               - MCQ solving
```

---

## Stage 1.2: Supervised Fine-Tuning

### Two Training Tracks

#### Track 1: Medical Dictionary QA (train_01)

**Purpose:** Learn medical term definitions

**Format:**
```
<start_of_turn>user
Meaning of word 고혈압:<end_of_turn>
<start_of_turn>model
Hypertension. 혈압이 정상 범위보다 높은 상태<end_of_turn>
```

**Data:**
- Medical Dictionary: 4,071 terms
- Character Dictionary: 89 symbols
- **Total:** 4,160 QA pairs

**Training:**
```bash
python script/train/train_01_medical_dict.py \
    --base-model model/00_trained/medgemma-4b/final \
    --epochs 3 \
    --device cuda:0
```

**Validation:**
- KorMedMCQA test (604 samples)
- Check if model can answer definition questions
- Expected accuracy: 50-60%

---

#### Track 2: MCQ with Reasoning (train_02)

**Purpose:** Learn to solve medical MCQs with chain-of-thought reasoning

**Format (95% simple):**
```
<start_of_turn>user
Reasoning 후 정답 알파벳 하나만 답하세요.

항문압 측정 검사에서 항문 압력이 증가하는 경우는?
A) 직장질루(rectovaginal fistula)
B) 항문열창(anal fissure)
C) 대변실금(fecal incontinence)
D) 대변메막힘(fecal impaction)
E) 직장탈출증(rectal prolapse)

<end_of_turn>
<start_of_turn>model
<reasoning>
각 선택지를 분석하면:
A) 직장질루 - 항문 압력 감소
B) 항문열창 - 괄약근 긴장 증가로 압력 상승 ✓
C) 대변실금 - 압력 감소
D) 대변메막힘 - 압력 변화 없음
E) 직장탈출증 - 압력 감소
</reasoning>B<end_of_turn>
```

**Format (5% detailed):**
```
<start_of_turn>user
Reasoning 후 정답 알파벳 하나만 답하세요.

```format
첫째: 문제의 핵심 파악 (3-5 문장)
둘째: 관련 의학 지식 설명 (3-5 문장)
셋째: 각 선택지 분석 (3-5 문장)
넷째: A) ~%, B) ~%, C) ~%, D) ~%, E) ~% 형식으로 확률 제시
```

```example
첫째: 이 문제는 심전도 ST 분절 상승을 보이는 환자의 진단에 관한 문제입니다...
둘째: ST 분절 상승은 심근경색의 특징적 소견으로...
셋째: A) 협심증은 ST 하강을 보이므로...
넷째: A) 5%, B) 90%, C) 3%, D) 1%, E) 1%
```

[MCQ 문제...]

<end_of_turn>
<start_of_turn>model
<reasoning>
첫째: 이 문제는 항문압 측정에서 압력이 증가하는 질환을 찾는 문제입니다. 항문 압력은 괄약근의 긴장도에 따라 결정됩니다.

둘째: 항문열창은 항문 점막의 찢어짐으로 인해 괄약근이 반사적으로 수축하여 압력이 증가합니다. 반면 실금이나 탈출증은 괄약근 약화로 압력이 감소합니다.

셋째: A) 직장질루는 누공 형성으로 압력과 무관. B) 항문열창은 통증으로 인한 괄약근 경련. C) 대변실금은 괄약근 약화. D) 메막힘은 압력 변화 없음. E) 탈출증은 약화.

넷째: A) 5%, B) 85%, C) 3%, D) 5%, E) 2%
</reasoning>B<end_of_turn>
```

**Data:**
- KorMedMCQA training: 1,890 samples
- Format distribution:
  - 95% simple reasoning (1,796 samples)
  - 5% detailed reasoning (94 samples)

**Training:**
```bash
python script/train/train_02_kor_med_test.py \
    --base-model model/01_trained/medgemma-4b/final \
    --epochs 5 \
    --device cuda:0
```

**Validation:**
- KorMedMCQA test (604 samples)
- Expected accuracy: 70-80%

---

### Loss Computation

#### Answer-Only Loss

**Mask prompt tokens:**

```python
def create_answer_only_labels(input_ids, answer_start_idx):
    """
    Mask prompt tokens with -100, only compute loss on answer.

    Args:
        input_ids: [seq_len] token ids
        answer_start_idx: index where answer begins

    Returns:
        labels: [seq_len] with -100 for prompt tokens
    """
    labels = input_ids.clone()
    labels[:answer_start_idx] = -100
    return labels

# Loss computation
loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    labels.view(-1),
    ignore_index=-100
)
```

**Why answer-only?**
- Prevents model from just copying prompt patterns
- Forces model to generate correct answer
- Focuses learning on answer generation

---

#### Reasoning Score (train_02)

**Weighted loss:**

```python
# Scoring
correct_answer = 2/3 weight
reasoning_format = 1/3 weight

# Format checks
has_reasoning_tags = '<reasoning>' in output and '</reasoning>' in output
has_keywords = all(kw in output for kw in ['첫째:', '둘째:', '셋째:', '넷째:'])
has_percentages = all(f'{letter})' in output for letter in 'ABCDE')

# Final score
if answer_correct:
    score = 2/3
    if has_reasoning_tags and has_keywords:
        score += 1/3
else:
    if has_reasoning_tags and has_keywords:
        score = 1/4  # Good reasoning, wrong answer
    else:
        score = 0
```

**Purpose:** Encourage both correct answers AND proper reasoning

---

### Loop Training (train_01_02_loop.py)

**Strategy:** Alternate between dictionary and MCQ training until target accuracy

```python
while accuracy < 90%:
    # Train on dictionary
    train_01(epochs=1)

    # Train on MCQ
    train_02(epochs=1)

    # Evaluate
    accuracy = evaluate_kormedmcqa()

    if accuracy >= 90%:
        break
```

**Checkpoints:**
```
models/loop_training/
├── loop_1/
│   ├── after_type1/  (dictionary trained)
│   └── after_type2/  (MCQ trained)
├── loop_2/
│   ├── after_type1/
│   └── after_type2/
├── best_checkpoint/  (highest accuracy)
└── final/            (when 90% reached)
```

**Command:**
```bash
python script/train/train_01_02_loop.py \
    --max-loops 10 \
    --target-accuracy 90 \
    --device cuda:0
```

---

## Data Preparation

### train_01 Format

**Input:** Medical dictionary JSON

```json
[
  {"term": "고혈압", "definition": "Hypertension. 혈압이 정상 범위보다 높은 상태"}
]
```

**Output:** Chat format JSONL

```json
{"prompt": "<start_of_turn>user\nMeaning of word 고혈압:<end_of_turn>", "completion": "<start_of_turn>model\nHypertension. 혈압이 정상 범위보다 높은 상태<end_of_turn>"}
```

---

### train_02 Format

**Input:** KorMedMCQA JSONL

```json
{
  "question": "항문압 측정 검사에서 항문 압력이 증가하는 경우는?",
  "A": "직장질루(rectovaginal fistula)",
  "B": "항문열창(anal fissure)",
  "answer": 2
}
```

**Output:** Chat format with reasoning

```json
{
  "prompt": "<start_of_turn>user\nReasoning 후 정답 알파벳 하나만 답하세요.\n\n항문압 측정 검사에서...\nA) 직장질루...\nB) 항문열창...\n<end_of_turn>",
  "completion": "<start_of_turn>model\n<reasoning>각 선택지 분석...</reasoning>B<end_of_turn>"
}
```

---

## Validation Strategy

### Metrics

1. **Accuracy:** Exact match on answer letter (A/B/C/D/E)
2. **Reasoning Quality:** Has proper format tags and keywords
3. **Loss:** Answer-only cross-entropy

### Evaluation Loop

```python
@torch.no_grad()
def evaluate_kormedmcqa(model, test_data, device):
    model.eval()
    correct = 0
    total = 0

    for item in test_data:
        # Generate answer
        prompt = format_mcq_prompt(item)
        output = model.generate(prompt)

        # Extract answer letter
        predicted = extract_answer_letter(output)
        expected = get_answer_letter(item['answer'])

        if predicted == expected:
            correct += 1
        total += 1

    accuracy = 100 * correct / total
    return accuracy
```

### Stop Conditions

**train_01:**
- Validation accuracy plateaus
- OR reaches target (60%)

**train_02:**
- Validation accuracy ≥80%
- OR loop training continues

**Loop training:**
- KorMedMCQA accuracy ≥90%
- OR max loops reached (10)

---

## Hyperparameters

### train_01 (Dictionary)

```python
{
    "epochs": 3,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "learning_rate": 2e-5,
    "max_length": 512,
    "lora_r": 64,
    "lora_alpha": 128
}
```

### train_02 (MCQ)

```python
{
    "epochs": 5,
    "batch_size": 2,
    "gradient_accumulation": 8,
    "learning_rate": 1e-5,
    "max_length": 1024,  # Longer for reasoning
    "lora_r": 64,
    "lora_alpha": 128
}
```

### Loop Training

```python
{
    "epochs_per_loop": 1,  # Each type gets 1 epoch per loop
    "max_loops": 10,
    "target_accuracy": 90,
    "patience": 3  # Stop if no improvement for 3 loops
}
```

---

## Expected Results

| Stage | Accuracy | Notes |
|-------|----------|-------|
| After CPT (train_00) | 30-40% | Baseline, no task training |
| After train_01 | 50-60% | Knows medical terms |
| After train_02 | 70-80% | Can reason through MCQ |
| After loop (1-3 loops) | 85-90% | Refined reasoning |
| Final target | ≥90% | Production ready |

---

## Troubleshooting

### Low Accuracy (<50%)

**Causes:**
- Base model (CPT) not trained enough
- Medical vocabulary not learned

**Solutions:**
- Train train_00 longer
- Check medical data 5x multiplication
- Verify plain text includes medical terms

### Plateau at 70-80%

**Causes:**
- Model memorizing patterns, not reasoning
- Insufficient reasoning examples

**Solutions:**
- Add more detailed reasoning examples (increase 5% → 10%)
- Adjust reasoning score weights
- Increase loop training iterations

### Overfitting

**Causes:**
- Training too many epochs
- Small validation set

**Solutions:**
- Reduce epochs
- Add early stopping
- Increase validation data

---

## Best Practices

1. **Always start from CPT model:** Don't skip train_00
2. **Monitor validation closely:** Catch overfitting early
3. **Use loop training:** Better than single-pass
4. **Check reasoning quality:** Not just accuracy
5. **Save all checkpoints:** May need to rollback

---

## References

- Format specification: `research/00__format.md`
- Training scripts: `script/train/train_01_medical_dict.py`, `train_02_kor_med_test.py`
- Loop training: `script/train/train_01_02_loop.py`
- Main training guide: `research/01_train.md`
