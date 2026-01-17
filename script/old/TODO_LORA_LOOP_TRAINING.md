# TODO: LoRA Loop Training with Reasoning

## Overview
Add LoRA adapter, update training scripts with reasoning prompts, and run loop training.

---

## 1. Add LoRA Adapter
- [ ] Run `add_lora_adapter.py` on trained model from train_00
- [ ] Output: `models/train_00_with_adapter/`

---

## 2. Update train/train_01_medical_dict.py

### Data Sources
- [ ] Include `02_refined/01_medical_dict.json` (medical terms)
- [ ] Include `02_refined/02_char_dict.json` (symbol dictionary)

### Prompt Format
```
<start_of_turn>user
Meaning of word {term}: <end_of_turn>
<start_of_turn>model
{definition}<end_of_turn>
```

### Training
- [ ] Same sample count as train_02 for balanced training

---

## 3. Update train/train_02_kor_med_test.py

### Training Prompt Format
```
<start_of_turn>user
Choose one alphabet after chain of thought in <reasoning>
First: rewrite the question.
Second: list related medical information.
Third: list why each choice is close to answer or not and likelihood percent.
A) why close or not xx%
B) why close or not yy%
C) why close or not zz%
D) why close or not qq%
E) why close or not rr%
</reasoning>(ONE ALPHABET RIGHT AFTER REASONING BLOCK)

{question}
A) {A}
B) {B}
C) {C}
D) {D}
E) {E}
<end_of_turn>
<start_of_turn>model
<reasoning>
First: {rewritten_question}
Second: {medical_info}
Third: {analysis}
A) {analysis_A} xx%
B) {analysis_B} yy%
C) {analysis_C} zz%
D) {analysis_D} qq%
E) {analysis_E} rr%
</reasoning>{answer}<end_of_turn>
```

### Validation Prompt Format
```
<start_of_turn>user
Answer one alphabet with reasoning block

{question}
A) {A}
B) {B}
C) {C}
D) {D}
E) {E}
<end_of_turn>
<start_of_turn>model
```

### Reward System (Custom Loss)

#### When Answer is CORRECT:
- **2/3 weight**: Correct answer match
- **1/3 weight**: Reasoning format check
  - Has `First:` with 3+ tokens
  - Has `Second:` with 3+ tokens
  - Has `Third:` with 3+ tokens
  - Has `A)` with 3+ tokens and percent (xx%)
  - Has `B)` with 3+ tokens and percent (yy%)
  - Has `C)` with 3+ tokens and percent (zz%)
  - Has `D)` with 3+ tokens and percent (qq%)
  - Has `E)` with 3+ tokens and percent (rr%)

#### When Answer is WRONG:
- **1/4 weight**: For proper reasoning format only
  - Same format checks as above

---

## 4. Candidate Prompts (Choose One)

### Option A: Detailed Instruction (Current)
System instructs exact reasoning format with First/Second/Third structure.

### Option B: Minimal Instruction
```
<start_of_turn>user
{question}
A) {A} B) {B} C) {C} D) {D} E) {E}

Think step by step in <reasoning>...</reasoning>, then answer.<end_of_turn>
```

### Option C: Korean Instruction
```
<start_of_turn>user
{question}
A) {A} B) {B} C) {C} D) {D} E) {E}

<reasoning> 안에서 단계별로 분석하고 정답 알파벳을 답하세요.<end_of_turn>
```

---

## 5. Run Loop Training

```bash
python train/train_01_02_loop.py \
    --model medgemma-4b \
    --base-model models/train_00_with_adapter \
    --total-epochs 5 \
    --samples-per-epoch 1000
```

### Validation
- Run KorMedMCQA validation after each epoch
- Show sample prompts periodically during training
- Track accuracy changes

---

## Files to Modify
1. `add_lora_adapter.py` - Already exists, check if works
2. `train/train_01_medical_dict.py` - Add symbol dict, new prompt format
3. `train/train_02_kor_med_test.py` - Add reasoning reward system
4. `train/train_01_02_loop.py` - Update for balanced samples
5. `validation_kor_med_test.py` - Use new validation prompt

---

## Expected Output Structure
```
models/
├── train_00_plain_text/medgemma-4b/final/   (done)
├── train_00_with_adapter/                    (step 1)
└── loop_training/
    ├── epoch_1/
    │   ├── after_01/
    │   └── after_02/
    ├── epoch_2/
    ├── ...
    ├── best_checkpoint/
    └── training_log.json
```
