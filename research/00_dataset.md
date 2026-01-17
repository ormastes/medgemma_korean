# Dataset Requirement for Training LLM in Other Languages

## Data Size Requirements by Training Approach

| Criterion | Tokens | Notes |
|-----------|--------|-------|
| Minimum viable | 2B | With vocabulary expansion (EEVE approach) |
| Effective training | 10-50B | Good balance of cost/performance |
| Optimal | 50-100B | Best results for continual pretraining |

---

## Dataset Sources

| Source | Size | Estimated Tokens | Purpose |
|--------|------|------------------|---------|
| C4 Korean | ~3GB | ~1B tokens | General Korean |
| NamuWiki | ~18GB | ~5-6B tokens | General Korean |
| Korean Wikipedia | ~3GB | ~1B tokens | General Korean (high quality) |
| AI Hub data | ~10GB | ~10-15B tokens | Various domains |
| **Total** | **~34GB** | **~17-23B tokens** | |

---

## Raw Data Formats (data/01_raw/)

### Type 0: Plain Text (Korean Corpus)
**Source:** NamuWiki, Wikipedia, C4 Korean
**Format:** Arrow files (HuggingFace datasets)
```python
# Arrow file structure
{
    "text": "한국어 텍스트 내용..."
}
```

### Type 1: Medical Dictionary
**Source:** `data/01_raw/01_medical_dict/korean_medical_dict.jsonl`
**Format:**
```json
{"term": "(Lactic dehydrogenase LDH)", "definition": "LD\n 세부설명 : 유산탈수소효소 락트산탈수소효소"}
{"term": "(골)수강", "definition": "Medullary Cavity\n 세부설명 : 뼈의 구조물의 일부분..."}
```

### Type 2: Medical MCQ (KorMedMCQA)
**Source:** `data/01_raw/02_kor_med_test/train.jsonl`
**Format:**
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

---

## Refined Data Formats (data/02_refined/)

### Type 0: Plain Text → Continued Pretraining Format
**Output:** `data/02_refined/00_plain_text/train.jsonl`
**Format:**
```json
{"text": "한국어 텍스트 내용..."}
```

### Type 1: Medical Dictionary → Instruction Format
**Output:** `data/02_refined/01_medical_dict.json`
**Format:**
```json
[
    {"term": "고혈압", "definition": "Hypertension. 혈압이 정상 범위보다 높은 상태..."},
    {"term": "당뇨병", "definition": "Diabetes mellitus. 인슐린 분비 또는 작용 장애..."}
]
```

### Type 2: Medical MCQ → Simplified Format
**Output:** `data/02_refined/02_kor_med_test/train.jsonl`
**Format:**
```json
{
    "question": "항문압 측정 검사에서 항문 압력이 증가하는 경우는?",
    "A": "직장질루(rectovaginal fistula)",
    "B": "항문열창(anal fissure)",
    "C": "대변실금(fecal incontinence)",
    "D": "대변메막힘(fecal impaction)",
    "E": "직장탈출증(rectal prolapse)",
    "answer": "B"
}
```

---

## Data Refinement Pipeline (Pseudocode)

### Step 1: Plain Text Refinement

```python
def refine_plain_text(raw_data_path, output_path):
    """
    Refine plain text for continued pretraining.
    Input: Arrow files from HuggingFace (NamuWiki, Wikipedia, C4)
    Output: JSONL with clean text
    """

    # 1. Load raw data
    dataset = load_dataset(raw_data_path)

    refined = []
    for item in dataset:
        text = item["text"]

        # 2. Quality filtering
        if len(text) < 100:  # Too short
            continue
        if detect_language(text) != "korean":  # Not Korean
            continue
        if is_advertisement(text):  # Remove ads
            continue
        if is_duplicate(text, seen_hashes):  # Deduplication
            continue

        # 3. Text cleaning
        text = remove_html_tags(text)
        text = normalize_whitespace(text)
        text = remove_special_patterns(text)  # URLs, emails, etc.

        # 4. Add to refined
        refined.append({"text": text})

    # 5. Save as JSONL
    save_jsonl(refined, output_path)

    return refined
```

### Step 2: Medical Dictionary Refinement

```python
def refine_medical_dict(raw_dict_path, output_path):
    """
    Refine medical dictionary for instruction tuning.
    Input: JSONL with term/definition pairs
    Output: JSON array with clean term/definition
    """

    # 1. Load raw data
    raw_dict = load_jsonl(raw_dict_path)

    refined = []
    for item in raw_dict:
        term = item["term"]
        definition = item["definition"]

        # 2. Clean term
        term = term.strip()
        term = remove_numbering(term)  # "1. 조 치조 2. 폐포" → "조/치조/폐포"

        # 3. Clean definition
        definition = definition.replace("\n 세부설명 : ", " - ")
        definition = normalize_whitespace(definition)

        # 4. Skip invalid entries
        if len(term) < 2 or len(definition) < 10:
            continue

        refined.append({
            "term": term,
            "definition": definition
        })

    # 5. Save as JSON
    save_json(refined, output_path)

    return refined
```

### Step 3: Medical MCQ Refinement

```python
def refine_medical_mcq(raw_mcq_path, output_path):
    """
    Refine medical MCQ for instruction tuning.
    Input: JSONL with full metadata
    Output: JSONL with simplified format
    """

    # 1. Load raw data
    raw_mcq = load_jsonl(raw_mcq_path)

    refined = []
    for item in raw_mcq:
        # 2. Extract essential fields
        question = item["question"]
        choices = {
            "A": item["A"],
            "B": item["B"],
            "C": item["C"],
            "D": item["D"],
            "E": item["E"]
        }

        # 3. Convert numeric answer to letter
        answer_num = item["answer"]  # 1-5
        answer_letter = chr(ord('A') + answer_num - 1)  # "A"-"E"

        # 4. Clean question text
        question = normalize_whitespace(question)

        # 5. Create refined entry
        refined.append({
            "question": question,
            **choices,
            "answer": answer_letter
        })

    # 6. Save as JSONL
    save_jsonl(refined, output_path)

    return refined
```

---

## Training Format for Decoder-Based LLM (MedGemma)

MedGemma uses Gemma's chat template with special tokens:

```
<bos><start_of_turn>user
{user_message}<end_of_turn>
<start_of_turn>model
{model_response}<end_of_turn><eos>
```

### Format Conversion Functions

```python
def format_for_training(data_type, item):
    """
    Convert refined data to MedGemma training format.
    """

    if data_type == "plain_text":
        return format_plain_text(item)
    elif data_type == "medical_dict":
        return format_medical_dict(item)
    elif data_type == "medical_mcq":
        return format_medical_mcq(item)


def format_plain_text(item):
    """
    Plain text → Continued pretraining format.
    No special structure, just raw text for language modeling.
    """
    return {
        "text": item["text"]
    }


def format_medical_dict(item):
    """
    Medical dictionary → Instruction format.
    """
    user_prompt = f"Meaning of word {item['term']}:"
    model_response = item["definition"]

    return {
        "text": f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n{model_response}<end_of_turn>"
    }


def format_medical_mcq(item, include_reasoning=True):
    """
    Medical MCQ → Instruction format with optional reasoning.
    """
    # Build question with choices
    question_text = item["question"]
    choices_text = f"""A) {item['A']}
B) {item['B']}
C) {item['C']}
D) {item['D']}
E) {item['E']}"""

    if include_reasoning:
        system_prompt = "Reasoning 후 정답 알파벳 하나만 답하세요."
        user_prompt = f"{system_prompt}\n\n{question_text}\n{choices_text}"

        # Model response with reasoning
        model_response = f"""<reasoning>
첫째: 문제 분석...
둘째: 선택지 검토...
셋째: 핵심 개념 적용...
넷째: A) ...xx%, B) ...yy%, C) ...zz%, D) ...ww%, E) ...vv%
</reasoning>{item['answer']}"""
    else:
        system_prompt = "정답 알파벳만 답하세요 (A, B, C, D, E 중 하나)."
        user_prompt = f"{system_prompt}\n\n{question_text}\n{choices_text}"
        model_response = item['answer']

    return {
        "text": f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n{model_response}<end_of_turn>"
    }
```

---

## Training Pseudocode for Decoder-Based MedGemma

### Overview: 3-Stage Training Pipeline

```
Stage 0: Continued Pretraining (Plain Text)
    → Learn Korean language patterns
    → Train: input_embeddings + output_embeddings

Stage 1: Medical Dictionary (Instruction Tuning)
    → Learn medical terminology
    → Train: LoRA on attention layers

Stage 2: Medical MCQ (Task-Specific Tuning)
    → Learn medical reasoning
    → Train: LoRA on attention layers
```

### Stage 0: Continued Pretraining

```python
def train_stage0_plain_text(base_model_path, data_path, output_path):
    """
    Continued pretraining on Korean plain text.
    Goal: Teach the model Korean language patterns.
    """

    # 1. Load model with 8-bit quantization
    model = load_model_8bit(base_model_path)
    tokenizer = load_tokenizer(base_model_path)

    # 2. Extend tokenizer with Korean tokens (optional)
    new_korean_tokens = load_korean_tokens()  # ~5000-10000 tokens
    tokenizer.add_tokens(new_korean_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # 3. Create LoRA config (include embeddings)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        modules_to_save=["embed_tokens", "lm_head"],  # Train embeddings
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True,  # Rank-stabilized LoRA
    )
    model = get_peft_model(model, lora_config)

    # 4. Load and prepare data
    dataset = load_jsonl(data_path)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=512,
            padding=False,
        )

    tokenized_dataset = dataset.map(tokenize)

    # 5. Training config
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=10,
    )

    # 6. Create trainer (standard language modeling)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Causal LM, not masked LM
        ),
    )

    # 7. Train
    trainer.train()

    # 8. Save
    model.save_pretrained(output_path + "/final")
    tokenizer.save_pretrained(output_path + "/final")
```

### Stage 1: Medical Dictionary Training

```python
def train_stage1_medical_dict(base_model_path, data_path, output_path):
    """
    Instruction tuning on medical dictionary.
    Goal: Teach medical terminology.
    """

    # 1. Load model (from Stage 0 output or base)
    model = load_peft_model(base_model_path, is_trainable=True)
    tokenizer = load_tokenizer(base_model_path)

    # 2. Load and format data
    raw_data = load_json(data_path)

    formatted_data = []
    for item in raw_data:
        formatted = format_medical_dict(item)
        formatted_data.append(formatted)

    # 3. Tokenize with proper padding
    def tokenize(example):
        # Use left padding for generation
        tokenizer.padding_side = "left"

        encoded = tokenizer(
            example["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

        # Set labels (same as input_ids for causal LM)
        encoded["labels"] = encoded["input_ids"].copy()

        # Mask padding tokens in labels (-100 = ignore)
        for i, token_id in enumerate(encoded["input_ids"]):
            if token_id == tokenizer.pad_token_id:
                encoded["labels"][i] = -100

        return encoded

    dataset = Dataset.from_list(formatted_data)
    tokenized_dataset = dataset.map(tokenize)

    # 4. Training with SFTTrainer
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        bf16=True,
        save_strategy="epoch",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # 5. Train with periodic evaluation
    trainer.train()

    # 6. Save
    model.save_pretrained(output_path + "/final")
    tokenizer.save_pretrained(output_path + "/final")
```

### Stage 2: Medical MCQ Training

```python
def train_stage2_medical_mcq(base_model_path, data_path, output_path):
    """
    Task-specific tuning on medical MCQ.
    Goal: Teach medical reasoning and answer selection.
    """

    # 1. Load model (from Stage 1 output)
    model = load_peft_model(base_model_path, is_trainable=True)
    tokenizer = load_tokenizer(base_model_path)

    # 2. Load and format data
    raw_data = load_jsonl(data_path)

    formatted_data = []
    for i, item in enumerate(raw_data):
        # 95% simple format, 5% detailed format
        include_detailed = (i % 20 == 0)
        formatted = format_medical_mcq(item, include_reasoning=True)
        formatted_data.append(formatted)

    # 3. Custom loss function for MCQ
    class MCQTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(**inputs)
            loss = outputs.loss

            # Optional: Weight correct answer tokens higher
            # This helps model focus on producing correct answer

            return (loss, outputs) if return_outputs else loss

    # 4. Tokenize
    def tokenize(example):
        tokenizer.padding_side = "left"

        encoded = tokenizer(
            example["text"],
            truncation=True,
            max_length=1024,  # Longer for reasoning
            padding="max_length",
        )

        encoded["labels"] = encoded["input_ids"].copy()

        # Mask user prompt (only train on model response)
        # Find position of "<start_of_turn>model"
        model_start = find_substring_position(
            encoded["input_ids"],
            tokenizer.encode("<start_of_turn>model")
        )

        for i in range(model_start):
            encoded["labels"][i] = -100

        return encoded

    dataset = Dataset.from_list(formatted_data)
    tokenized_dataset = dataset.map(tokenize)

    # 5. Training
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,  # Lower LR for fine-tuning
        warmup_ratio=0.1,
        bf16=True,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
    )

    trainer = MCQTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # 6. Train
    trainer.train()

    # 7. Evaluate on test set
    accuracy = evaluate_kormedmcqa(model, tokenizer, test_data_path)
    print(f"KorMedMCQA Accuracy: {accuracy:.2f}%")

    # 8. Save
    model.save_pretrained(output_path + "/final")
    tokenizer.save_pretrained(output_path + "/final")
```

### Evaluation Function

```python
def evaluate_kormedmcqa(model, tokenizer, test_path):
    """
    Evaluate model on KorMedMCQA test set.
    """
    test_data = load_jsonl(test_path)

    correct = 0
    total = 0

    model.eval()

    for item in test_data:
        # Format question
        prompt = format_mcq_for_inference(item)

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.encode("<end_of_turn>")[0],
            )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract answer (last character before <end_of_turn>)
        predicted = extract_answer(response)
        expected = item["answer"]

        if predicted == expected:
            correct += 1
        total += 1

    accuracy = (correct / total) * 100
    return accuracy


def extract_answer(response):
    """
    Extract answer letter from model response.
    Handles both direct answer and reasoning format.
    """
    # Find content after last </reasoning> or after <start_of_turn>model
    if "</reasoning>" in response:
        answer_part = response.split("</reasoning>")[-1]
    else:
        answer_part = response.split("<start_of_turn>model")[-1]

    # Find first A-E letter
    for char in answer_part:
        if char in "ABCDE":
            return char

    return None
```

---

## Complete Training Pipeline

```python
def train_full_pipeline(
    base_model="google/medgemma-4b-it",
    data_dir="data/02_refined",
    output_dir="model",
    target_accuracy=90.0
):
    """
    Complete training pipeline for Korean MedGemma.
    """

    # Stage 0: Continued Pretraining
    print("=" * 50)
    print("Stage 0: Continued Pretraining on Korean Text")
    print("=" * 50)

    train_stage0_plain_text(
        base_model_path=base_model,
        data_path=f"{data_dir}/00_plain_text/train.jsonl",
        output_path=f"{output_dir}/00_trained"
    )

    # Stage 1: Medical Dictionary
    print("=" * 50)
    print("Stage 1: Medical Dictionary Training")
    print("=" * 50)

    train_stage1_medical_dict(
        base_model_path=f"{output_dir}/00_trained/final",
        data_path=f"{data_dir}/01_medical_dict.json",
        output_path=f"{output_dir}/01_trained"
    )

    # Stage 2: Medical MCQ (loop until target accuracy)
    print("=" * 50)
    print("Stage 2: Medical MCQ Training")
    print("=" * 50)

    current_accuracy = 0.0
    loop = 0
    current_model = f"{output_dir}/01_trained/final"

    while current_accuracy < target_accuracy:
        loop += 1
        print(f"\n--- Loop {loop} ---")

        train_stage2_medical_mcq(
            base_model_path=current_model,
            data_path=f"{data_dir}/02_kor_med_test/train.jsonl",
            output_path=f"{output_dir}/02_trained/loop_{loop}"
        )

        # Evaluate
        model = load_peft_model(f"{output_dir}/02_trained/loop_{loop}/final")
        tokenizer = load_tokenizer(f"{output_dir}/02_trained/loop_{loop}/final")

        current_accuracy = evaluate_kormedmcqa(
            model, tokenizer,
            f"{data_dir}/02_kor_med_test/test.jsonl"
        )

        print(f"Loop {loop} Accuracy: {current_accuracy:.2f}%")

        current_model = f"{output_dir}/02_trained/loop_{loop}/final"

        if loop >= 10:  # Max loops
            print("Max loops reached")
            break

    print("=" * 50)
    print(f"Training complete! Final accuracy: {current_accuracy:.2f}%")
    print("=" * 50)
```

---

## Key Training Concepts for Decoder-Based LLMs

### 1. Causal Language Modeling (CLM)
```
Input:  [token1, token2, token3, token4]
Target: [token2, token3, token4, token5]

Loss = CrossEntropy(predicted, target)
```
Model learns to predict next token given previous tokens.

### 2. Instruction Tuning Format
```
<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
{response}<end_of_turn>
```
Model learns to follow instructions in user turn.

### 3. Label Masking
```python
# Only compute loss on model response, not user prompt
labels = input_ids.copy()
labels[:user_prompt_end] = -100  # Ignore these tokens in loss
```

### 4. LoRA (Low-Rank Adaptation)
```python
# Instead of updating full weight matrix W
# W_new = W + ΔW

# LoRA uses low-rank decomposition
# ΔW = A × B, where A is (d, r) and B is (r, k), r << d, k

# Benefits:
# - Much fewer trainable parameters
# - Can merge back into base model
# - Memory efficient
```

### 5. Gradient Checkpointing
```python
# Trade compute for memory
# Instead of storing all activations, recompute during backward pass
model.gradient_checkpointing_enable()
```

---

## Summary: Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAW DATA (data/01_raw/)                     │
├─────────────────────────────────────────────────────────────────┤
│  00_korean/           → Arrow files (NamuWiki, Wiki, C4)        │
│  01_medical_dict/     → JSONL {term, definition}                │
│  02_kor_med_test/     → JSONL {question, A-E, answer(num)}      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                     Refinement Scripts
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  REFINED DATA (data/02_refined/)                │
├─────────────────────────────────────────────────────────────────┤
│  00_plain_text/       → JSONL {text}                            │
│  01_medical_dict.json → JSON [{term, definition}]               │
│  02_kor_med_test/     → JSONL {question, A-E, answer(letter)}   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                     Format Conversion
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING FORMAT                              │
├─────────────────────────────────────────────────────────────────┤
│  Stage 0: {text: "Korean plain text..."}                        │
│  Stage 1: {text: "<start_of_turn>user\nMeaning..."}             │
│  Stage 2: {text: "<start_of_turn>user\nReasoning..."}           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                     Training Pipeline
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT MODELS (model/)                       │
├─────────────────────────────────────────────────────────────────┤
│  00_trained/          → Stage 0 output (Korean language)        │
│  01_trained/          → Stage 1 output (Medical dictionary)     │
│  02_trained/          → Stage 2 output (Medical MCQ)            │
└─────────────────────────────────────────────────────────────────┘
```
