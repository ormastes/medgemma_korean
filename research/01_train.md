# 1 Train
After LoRA setup.

## 1.1 Train Raw Korean Texts (Continued Pretraining)

Use cross-entropy loss on sequentially packed token data.

### Data Format: Raw Text String

All data sources are converted to **plain text strings** for continued pretraining:

```json
{"text": "plain Korean text without any special formatting"}
```

---

### Raw Data Sources (What Goes Into Plain Text)

Before conversion, raw data exists in these locations:

#### Source 1: Korean Plain Text Datasets
**Location:** `data/01_raw/00_korean/`

**Structure:**
```
data/01_raw/00_korean/
├── namu_wiki/
│   └── train/
│       ├── data-00000-of-00038.arrow    (583 MB)
│       ├── data-00001-of-00038.arrow    (584 MB)
│       └── ... (38 arrow files, ~18 GB total)
├── wikipedia-korean/
│   ├── data-00000-of-00007.arrow        (~3 GB total)
│   └── ...
├── c4_korean/
│   ├── data-00000-of-00007.arrow        (~3 GB total)
│   └── ...
└── korean/
    └── raw_text_mixed.jsonl             (443 MB - OUTPUT)
```

**Raw Format Example (from Arrow dataset):**
```python
# Reading from dataset
from datasets import load_from_disk
ds = load_from_disk("data/01_raw/00_korean/namu_wiki")

# Sample record:
{
  "namespace": 0,
  "title": "!!아앗!!",
  "text": "[목차] '''{{{+1 ！！ああっと！！}}}''' == 개요 == ...",
  "contributors": ["110.46.34.123", "kirby10", "max0243"],
  "id": "000002-1"
}
```

**Total:** ~6.2M articles in Namu Wiki, ~1M in Wikipedia, ~1M in C4

**Note:** There is also a Korean book corpus (`029.대규모_구매도서_기반_한국어_말뭉치_데이터/`, ~17 GB) in the `korean/` directory, but it is **NOT currently used** in the conversion. The conversion script only uses the arrow datasets above.

---

#### Source 2: KorMedMCQA Dataset
**Location:** `data/01_raw/02_kor_med_test/`

**Files:**
```
data/01_raw/02_kor_med_test/
├── train.jsonl          (1,890 samples)
├── test.jsonl           (604 samples)
└── dataset_info.json
```

**Raw Format Example:**
```json
{
  "subject": "doctor",
  "year": 2012,
  "period": 1,
  "q_number": 1,
  "question": "항문압 측정 검사에서 항문 압력이 증가하는 경우는?",
  "A": "직장질루(rectovaginal fistula)",
  "B": "항문열창(anal fissure)",
  "C": "대변실금(fecal incontinence)",
  "D": "대변메막힘(fecal impaction)",
  "E": "직장탈출증(rectal prolapse)",
  "answer": 2,
  "cot": "",
  "exam_type": "doctor"
}
```

**Total:** 1,890 training samples, 604 test samples

---

#### Source 3: Medical Dictionary
**Location:** `data/02_refined/01_medical_dict.json`

**Raw Format Example:**
```json
[
  {
    "term": "약",
    "definition": "medicine"
  },
  {
    "term": "고혈압",
    "definition": "Hypertension. 혈압이 정상 범위보다 높은 상태"
  },
  {
    "term": "당뇨병",
    "definition": "Diabetes Mellitus. 인슐린 분비나 작용의 이상"
  }
]
```

**Total:** 4,071 medical terms

---

#### Source 4: Character Dictionary
**Location:** `data/02_refined/02_char_dict.json`

**Raw Format Example:**
```json
[
  {
    "term": "↑",
    "definition": "increased, elevated, or upward trend"
  },
  {
    "term": "↓",
    "definition": "decreased, reduced, or downward trend"
  },
  {
    "term": "±",
    "definition": "plus-minus sign"
  }
]
```

**Total:** 89 medical symbols

---

### Conversion Process

**Script:** `script/create_raw_text_dataset.py`

**Process:**
1. Load Korean datasets from arrow files
2. Load MCQ from JSONL
3. Load dictionaries from JSON
4. Convert each to plain text format
5. Mix randomly with 5x sampling
6. Write to `data/01_raw/00_korean/korean/raw_text_mixed.jsonl`

**Command:**
```bash
python script/create_raw_text_dataset.py

# Output: 530,140 samples, 443 MB
```

---

### Data Sources (After Conversion - Mixed 5x Random Sampling)

#### 1. Korean Plain Text (~100K samples → 500K after 5x)

**Raw Data Source:** `data/01_raw/00_korean/{namu_wiki,wikipedia-korean,c4_korean}/`

**Original Format (Dataset - Arrow Files):**
```json
{
  "namespace": 0,
  "title": "당뇨병",
  "text": "[목차] '''당뇨병'''은 [[혈당]] 조절 장애로 인한 [[대사 질환]]이다. 인슐린 분비나 작용의 이상으로 발생한다.",
  "contributors": ["user1", "user2"],
  "id": "123456"
}
```

**After Conversion (Raw Text):**
```json
{"text": "당뇨병은 혈당 조절 장애로 인한 대사 질환이다. 인슐린 분비나 작용의 이상으로 발생한다."}
```

**Changes:**
- Remove wiki markup: `'''`, `[[]]`, `[목차]`
- Remove metadata: namespace, title, contributors, id
- Keep only cleaned plain text

---

#### 2. KorMedMCQA (1,890 samples → 9,506 after 5x)

**Raw Data Source:** `data/01_raw/02_kor_med_test/train.jsonl`

**Original Format (JSONL - JSON Lines):**
```json
{
  "subject": "doctor",
  "year": 2012,
  "q_number": 1,
  "question": "항문압 측정 검사에서 항문 압력이 증가하는 경우는?",
  "A": "직장질루(rectovaginal fistula)",
  "B": "항문열창(anal fissure)",
  "C": "대변실금(fecal incontinence)",
  "D": "대변메막힘(fecal impaction)",
  "E": "직장탈출증(rectal prolapse)",
  "answer": 2,
  "cot": "",
  "exam_type": "doctor"
}
```

**After Conversion (Raw Text):**
```json
{"text": "항문압 측정 검사에서 항문 압력이 증가하는 경우는?\n선택지: 직장질루(rectovaginal fistula), 항문열창(anal fissure), 대변실금(fecal incontinence), 대변메막힘(fecal impaction), 직장탈출증(rectal prolapse)\n정답: 항문열창\n\n항문열창(anal fissure)은 의학적으로 중요한 개념이다."}
```

**Changes:**
- Remove metadata: subject, year, q_number, exam_type
- Format as Q&A text with Korean medical terms
- Add brief explanation using answer text

---

#### 3. Medical Dictionary (4,071 samples → 20,572 after 5x)

**Raw Data Source:** `data/02_refined/01_medical_dict.json`

**Original Format (JSON Array):**
```json
{
  "term": "고혈압",
  "definition": "Hypertension. 혈압이 정상 범위보다 높은 상태"
}
```

**After Conversion (Raw Text):**
```json
{"text": "고혈압(Hypertension)은 혈압이 정상 범위보다 높은 상태를 의미한다."}
```

**Original Format (Simple Term):**
```json
{
  "term": "약",
  "definition": "medicine"
}
```

**After Conversion (Raw Text):**
```json
{"text": "약은 medicine을 의미하는 의학 용어이다."}
```

**Changes:**
- Remove JSON structure
- Create natural Korean sentence
- Combine Korean term + English definition

---

#### 4. Character Dictionary (67 samples → 327 after 5x)

**Raw Data Source:** `data/02_refined/02_char_dict.json`

**Original Format (JSON Array):**
```json
{
  "term": "↑",
  "definition": "increased, elevated, or upward trend"
}
```

**After Conversion (Raw Text):**
```json
{"text": "의학 기호 ↑는 increased, elevated, or upward trend를 나타낸다."}
```

**Original Format (Another Symbol):**
```json
{
  "term": "±",
  "definition": "plus-minus sign"
}
```

**After Conversion (Raw Text):**
```json
{"text": "의학 기호 ±는 plus-minus sign를 나타낸다."}
```

**Changes:**
- Remove JSON structure
- Create explanation sentence in Korean
- Link symbol to its meaning

---

### Mixed Dataset Output

**File:** `data/01_raw/00_korean/korean/raw_text_mixed.jsonl`

**Format:** One JSON object per line (JSONL)

**Example Lines:**
```json
{"text": "당뇨병은 혈당 조절 장애로 인한 대사 질환이다..."}
{"text": "항문압 측정 검사에서 항문 압력이 증가하는 경우는?\n선택지: ..."}
{"text": "고혈압(Hypertension)은 혈압이 정상 범위보다 높은 상태를 의미한다."}
{"text": "의학 기호 ↑는 increased, elevated, or upward trend를 나타낸다."}
{"text": "스사노오 10식과의 차이점은 색상이 어둡고..."}
```

**Distribution after 5x random sampling:**
- Korean Plain: 94.3% (499,735 samples)
- MCQ: 1.8% (9,506 samples)
- Medical Dict: 3.9% (20,572 samples)
- Char Dict: 0.1% (327 samples)

**Total:** 530,140 samples, 443 MB

---

### Training Process

**Input:** Raw text strings (no special tokens, no formatting)

```python
# Each sample is just plain text
sample1 = "당뇨병은 혈당 조절 장애로 인한 대사 질환이다..."
sample2 = "항문압 측정 검사에서 항문 압력이 증가하는 경우는?..."
sample3 = "고혈압(Hypertension)은 혈압이..."
```

**Loss:** Cross-entropy on next-token prediction

```python
# Tokenize and pack sequences
tokens = tokenizer(texts, truncation=False, padding=False)
packed = pack_sequences(tokens, max_length=512, eos_token="<end_of_turn>")

# Train with CE loss
loss = cross_entropy(model(input_ids), labels)
```

**Packing:** Multiple texts packed into single sequence with EOS

```
[text1] <end_of_turn> [text2] <end_of_turn> [text3] <end_of_turn> ...
```

---

### Validation

**Metrics:**
- Validation loss (held-out Korean text)
- Validation perplexity = exp(loss)
- Tokens/sec (data pipeline health)

**Stop conditions:**
- Val loss stops improving (<0.01 improvement)
- Patience: 5-10 evaluations
- Target perplexity: <3.0

**Implementation:**
```python
# See research/01__train_raw_text.md for full pseudo code
if val_loss < best_val - 1e-4:
    best_val = val_loss
    bad_evals = 0
else:
    bad_evals += 1
    if bad_evals >= patience:
        print("Early stop: raw Korean val stopped improving")
```

---

### Key Points

1. **No special formatting** - Just plain Korean text
2. **No chat templates** - No `<start_of_turn>`, `<end_of_turn>` in text content
3. **Natural language** - Reads like normal Korean prose
4. **Medical terms preserved** - Korean(English) pairs maintained
5. **Random mixing** - Each batch has diverse content

**Why raw text?**
- Continued pretraining learns language patterns
- No instruction-following needed yet
- Focus on Korean fluency + medical vocabulary
- Chat formatting comes in later stages (train_01, train_02)

---

## 1.2 Train Exam Prompt (Staged SFT)

After raw text pretraining (1.1), switch to **chat format** with special tokens.

### Format Change: Raw Text → Chat Format

**Stage 1.1 (Raw Text CPT):**
```json
{"text": "고혈압(Hypertension)은 혈압이 정상 범위보다 높은 상태를 의미한다."}
```

**Stage 1.2+ (Chat Format SFT):**
```
<start_of_turn>user
Meaning of word 고혈압:<end_of_turn>
<start_of_turn>model
Hypertension. 혈압이 정상 범위보다 높은 상태<end_of_turn>
```

---

### Step 1: QA + Example

**Goal:** Train on QA format, measure QA loss

**Data:**
- Medical dictionary (chat format)
- MCQ examples (chat format)

**Loss:** Answer-only loss (mask prompt tokens)

**Stop when:**
- QA validation loss < 0.3
- Raw Korean ppl < 1.03 × baseline

---

### Step 2: QA + Example (More Data)

**Goal:** Expand to more QA examples

**Data:**
- All medical QA
- All MCQ with reasoning

**Stop when:**
- QA validation loss < 0.2
- Raw Korean ppl maintained

---

### Step 3: QA + Example + Dict

**Goal:** Add dictionary alongside QA

**Data:**
- QA (70%)
- Dictionary (30%)

**Stop when:**
- QA loss < 0.2
- Dict loss < 0.3
- Raw Korean ppl maintained

---

### Step 4: QA + Example + Dict + Raw Text

**Goal:** Mix instruction data with raw Korean text to prevent forgetting

**Data:**
- QA (40%)
- Dictionary (20%)
- Raw Korean text (40%)

**Stop when:**
- QA loss < 0.2
- Raw Korean ppl ≈ 1.1 baseline (same as step 1)
- Both QA and raw Korean stable for 5+ evaluations

**This ensures:**
- QA ability maintained (loss ~0)
- Korean fluency maintained (ppl similar to CPT)

---

## Summary: Data Format by Stage

| Stage | Format | Example | Special Tokens |
|-------|--------|---------|----------------|
| 1.1 CPT | Raw text | `"당뇨병은 혈당 조절 장애..."` | No |

**Key difference:**
- CPT uses **plain text** for language learning

---

## References

- Raw text format details: `research/01__train_raw_text.md`
- Dataset creation: `script/create_raw_text_dataset.py`
- Training scripts: `script/train/train_00_plain_text.py`
