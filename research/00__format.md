# Data Format Specification

## Overview

This document specifies the data formats used throughout the training pipeline.

## Format Types

### 1. Raw Text Format (Stage 1.1 - Continued Pretraining)

**Purpose:** Language learning without instruction following

**Format:**
```json
{"text": "plain Korean text without special tokens"}
```

**Characteristics:**
- No special tokens
- No chat templates
- Natural Korean language
- Used for continued pretraining (CPT)

**Examples:**

#### Korean Plain Text
```json
{"text": "당뇨병은 혈당 조절 장애로 인한 대사 질환이다. 인슐린 분비나 작용의 이상으로 발생한다."}
```

#### MCQ as Plain Text
```json
{"text": "항문압 측정 검사에서 항문 압력이 증가하는 경우는?\n선택지: 직장질루(rectovaginal fistula), 항문열창(anal fissure), 대변실금(fecal incontinence), 대변메막힘(fecal impaction), 직장탈출증(rectal prolapse)\n정답: 항문열창\n\n항문열창(anal fissure)은 의학적으로 중요한 개념이다."}
```

#### Medical Dictionary as Plain Text
```json
{"text": "고혈압(Hypertension)은 혈압이 정상 범위보다 높은 상태를 의미한다."}
```

#### Book Corpus
```json
{"text": "마틴 가드너 저, 공창식 역, 쉽게 배우는 미적분학, 홍릉과학출판사, 2004년"}
```

---

### 2. Chat Format (Stage 1.2+ - Supervised Fine-Tuning)

**Purpose:** Instruction following and task-specific training

**Format:**
```
<start_of_turn>user
[prompt]<end_of_turn>
<start_of_turn>model
[response]<end_of_turn>
```

**Characteristics:**
- Special tokens: `<start_of_turn>`, `<end_of_turn>`
- User-model dialogue structure
- Answer-only loss (prompt masked)
- Used for SFT (Supervised Fine-Tuning)

**Examples:**

#### Medical Dictionary QA
```
<start_of_turn>user
Meaning of word 고혈압:<end_of_turn>
<start_of_turn>model
Hypertension. 혈압이 정상 범위보다 높은 상태<end_of_turn>
```

#### MCQ Simple Format (95%)
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
B) 항문열창 - 괄약근 긴장 증가로 압력 상승 ✓
</reasoning>B<end_of_turn>
```

#### MCQ Detailed Format (5%)
```
<start_of_turn>user
Reasoning 후 정답 알파벳 하나만 답하세요.

```format
첫째: 문제의 핵심 파악 (3-5 문장)
둘째: 관련 의학 지식 설명 (3-5 문장)
셋째: 각 선택지 분석 (3-5 문장)
넷째: A) ~%, B) ~%, C) ~%, D) ~%, E) ~% 형식으로 확률 제시
```

[MCQ 문제...]

<end_of_turn>
<start_of_turn>model
<reasoning>
첫째: 이 문제는 항문압 측정에서 압력이 증가하는 질환을 찾는 문제입니다.
둘째: 항문열창은 괄약근의 반사적 수축을 유발합니다.
셋째: A) 5%, B) 85%, C) 3%, D) 5%, E) 2%
넷째: B가 정답입니다.
</reasoning>B<end_of_turn>
```

---

## Format Evolution Through Training Stages

| Stage | Format | Special Tokens | Purpose | Example |
|-------|--------|----------------|---------|---------|
| **1.1 CPT** | Raw text | None | Language learning | `{"text": "당뇨병은..."}` |
| **1.2 SFT** | Chat | `<start_of_turn>` | Instruction following | `<start_of_turn>user\n...<end_of_turn>` |

---

## Data Sources & Formats

### Raw Data → Plain Text

| Source | Raw Format | Plain Text Output | Multiplication |
|--------|-----------|-------------------|----------------|
| Namu Wiki | Arrow dataset | `{"text": "..."}` | 100K sampled |
| Wikipedia | Arrow dataset | `{"text": "..."}` | Included |
| C4 Korean | Arrow dataset | `{"text": "..."}` | Included |
| Book Corpus | ZIP (JSON) | `{"text": "..."}` | ALL |
| KorMedMCQA | JSONL | `{"text": "Q\n선택지\n정답\n설명"}` | **×5** |
| Medical Dict | JSON | `{"text": "용어(English)은 정의"}` | **×5** |
| Char Dict | JSON | `{"text": "의학 기호 ↑는..."}` | **×5** |

**Medical Data Multiplication:** MCQ, Medical Dictionary, and Character Dictionary are multiplied 5x before mixing to ensure adequate medical vocabulary exposure during CPT.

---

## Plain Text Training Strategy

### Data Composition

```
Plain Text Training Data:
├── Korean General Text (100K samples)
│   ├── Namu Wiki
│   ├── Wikipedia
│   └── C4 Korean
├── Korean Book Corpus (ALL samples, ~millions)
└── Medical Data (×5 multiplication)
    ├── KorMedMCQA: 1,890 → 9,450 samples
    ├── Medical Dict: 4,071 → 20,355 samples
    └── Char Dict: 89 → 445 samples
```

**Rationale:**
- Medical data is only ~6K samples (1,890 + 4,071 + 89)
- Book corpus provides millions of general Korean samples
- 5x multiplication ensures medical terms aren't drowned out
- Model learns both Korean language AND medical vocabulary

### Random Mixing

After multiplication, all sources are:
1. Combined into a single pool
2. Shuffled randomly
3. Written to output file

**Result:** Each training batch contains diverse content (general + medical)

---

## Loss Computation

### Stage 1.1 (Raw Text CPT)

**Loss:** Cross-entropy over full sequence

```python
# No masking - predict every token
loss = cross_entropy(logits, labels)
```

**Packing:**
```
[text1] <end_of_turn> [text2] <end_of_turn> [text3] <end_of_turn>
```

### Stage 1.2 (Chat Format SFT)

**Loss:** Cross-entropy on answer tokens only

```python
# Mask prompt tokens with -100
labels = input_ids.clone()
labels[:answer_start_idx] = -100

loss = cross_entropy(logits, labels, ignore_index=-100)
```

**No packing** - one conversation per sequence

---

## Tokenization

### Plain Text (CPT)
```python
# Simple tokenization
tokens = tokenizer(text, truncation=False, padding=False)
```

### Chat Format (SFT)
```python
# Apply chat template
formatted = tokenizer.apply_chat_template(
    [{"role": "user", "content": user_msg},
     {"role": "assistant", "content": assistant_msg}],
    tokenize=True
)
```

---

## Validation Data Format

**Same as training format** for each stage:

- **CPT validation:** Raw text `{"text": "..."}`
- **SFT validation:** Chat format with `<start_of_turn>`

---

## Special Tokens

### Gemma Tokenizer

| Token | ID | Usage |
|-------|-----|-------|
| `<bos>` | 2 | Beginning of sequence |
| `<eos>` | 1 | End of sequence |
| `<pad>` | 0 | Padding |
| `<start_of_turn>` | 106 | Start of turn (chat) |
| `<end_of_turn>` | 107 | End of turn (chat) |

### Reasoning Tags (Custom)

| Token | Usage |
|-------|-------|
| `<reasoning>` | Start of reasoning block |
| `</reasoning>` | End of reasoning block |

**Note:** Reasoning tags are part of text content, not special tokens in tokenizer.

---

## File Formats

### JSONL (JSON Lines)

One JSON object per line:

```json
{"text": "first sample"}
{"text": "second sample"}
{"text": "third sample"}
```

**Usage:** All output files (plain text, mixed datasets)

### JSON Array

Array of objects:

```json
[
  {"term": "고혈압", "definition": "Hypertension..."},
  {"term": "당뇨병", "definition": "Diabetes..."}
]
```

**Usage:** Input dictionaries

---

## Quality Filters

### Plain Text Filters

```python
# Minimum length
if len(text) < 100:
    skip

# Korean ratio
korean_chars = len(re.findall(r'[\uac00-\ud7af]', text))
korean_ratio = korean_chars / len(text)
if korean_ratio < 0.1:
    skip
```

### Medical Data Filters

```python
# Must have medical term
if not has_medical_term(text):
    skip

# Must have Korean-English pair
if not has_korean_english_pair(text):
    skip
```

---

## Summary

**Raw Text Format (CPT):**
- Purpose: Language + medical vocabulary learning
- Format: `{"text": "plain text"}`
- Medical data: Multiplied 5x
- Loss: Full sequence CE

**Chat Format (SFT):**
- Purpose: Instruction following + reasoning
- Format: `<start_of_turn>user\n...<end_of_turn>`
- Medical data: Task-specific formatting
- Loss: Answer-only CE

**Progression:** Raw text → Chat format → Task performance
