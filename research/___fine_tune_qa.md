Below is a **fine-tuning QA set** to teach the **format rules**. Each sample contains **only**:

* `question:\n`
* `short_answer:\n`

(As requested, no other sections appear in these training examples.)

---

### 1

```
question:\n
What sections can contain Korean?
short_answer:\n
question:\n evaluation:\n answer:\n
```

### 2

```
question:\n
What must be included in translate?
short_answer:\n
English translation of the whole question and all options if options exist.
```

### 3

```
question:\n
What must be included in evaluation?
short_answer:\n
평가기준, 후보/점수(또는 점수표), 근거요약
```

### 4

```
question:\n
Which answer types require a one-character answer?
short_answer:\n
select_1_5, select_A_E
```

### 5

```
question:\n
What answer types exist?
short_answer:\n
select_1_5, select_A_E, word, sentence
```

### 6

```
question:\n
What items must exist in a full QA record?
short_answer:\n
question:\n translate:\n answer_type:\n remind:\n reasoning:\n facts:\n candidates:\n criteria:\n analysis:\n evaluation:\n answer:\n
```

### 7

```
question:\n
What language must translate be written in?
short_answer:\n
English only.
```

### 8

```
question:\n
What language must reasoning/facts/candidates/criteria/analysis be written in?
short_answer:\n
English only.
```

### 9

```
question:\n
What language must evaluation be written in?
short_answer:\n
Korean only.
```

### 10

```
question:\n
What language must answer be written in?
short_answer:\n
Korean by policy, except select types must be exactly one character.
```

### 11

```
question:\n
For answer_type select_A_E, what is the allowed output in answer:\n?
short_answer:\n
Exactly one character: A or B or C or D or E.
```

### 12

```
question:\n
For answer_type select_1_5, what is the allowed output in answer:\n?
short_answer:\n
Exactly one character: 1 or 2 or 3 or 4 or 5.
```

### 13

```
question:\n
For answer_type word, what is the allowed output in answer:\n?
short_answer:\n
Exactly one Korean word with no spaces.
```

### 14

```
question:\n
For answer_type sentence, what is the allowed output in answer:\n?
short_answer:\n
Exactly one Korean sentence.
```

### 15

```
question:\n
Where should explanations for multiple-choice answers be placed?
short_answer:\n
evaluation:\n (Korean), not answer:\n
```

### 16

```
question:\n
What must evaluation include for select_A_E?
short_answer:\n
All A-E options with points and a Korean justification summary.
```

### 17

```
question:\n
What must evaluation include for select_1_5?
short_answer:\n
All 1-5 options with points and a Korean justification summary.
```

### 18

```
question:\n
What must evaluation include for word type?
short_answer:\n
Multiple candidate Korean words with points and Korean justification summary.
```

### 19

```
question:\n
What must evaluation include for sentence type?
short_answer:\n
Required content elements (후보 항목) with points and Korean justification summary.
```

### 20

```
question:\n
What must candidates:\n contain for select_A_E?
short_answer:\n
English list describing A, B, C, D, E options (labels + brief meaning).
```

### 21

```
question:\n
What must candidates:\n contain for select_1_5?
short_answer:\n
English list describing 1,2,3,4,5 meanings (scale definitions).
```

### 22

```
question:\n
What must facts:\n contain?
short_answer:\n
English extracted key facts from the question only (no new hallucinated facts).
```

### 23

```
question:\n
What must criteria:\n contain?
short_answer:\n
English scoring rubric used to compare candidates.
```

### 24

```
question:\n
What must analysis:\n contain?
short_answer:\n
English reasoning that applies criteria to candidates and selects the best.
```

### 25

```
question:\n
What must remind:\n contain?
short_answer:\n
English constraints for output format (length, charset, one-char rule, one-word rule).
```

### 26

```
question:\n
Where should the Korean translation of the question go?
short_answer:\n
question:\n (Korean already) and translate:\n is English translation, not Korean.
```

### 27

```
question:\n
If question is already Korean, what should translate:\n contain?
short_answer:\n
English translation of the Korean question and options.
```

### 28

```
question:\n
If a question has options, what must translate:\n include?
short_answer:\n
English translation of the stem plus all options.
```

### 29

```
question:\n
In full QA format, should headers include a trailing newline?
short_answer:\n
Yes, headers are atomic tokens like "section_name:\n".
```

### 30

```
question:\n
Which headers must be atomic tokens with "\n"?
short_answer:\n
question:\n translate:\n answer_type:\n remind:\n reasoning:\n facts:\n candidates:\n criteria:\n analysis:\n evaluation:\n answer:\n short_answer:\n
```

### 31

```
question:\n
Should answer:\n ever include extra explanation text for select types?
short_answer:\n
No, select types must be one character only.
```

### 32

```
question:\n
Where should Korean justification be placed when answer is one character?
short_answer:\n
evaluation:\n
```

### 33

```
question:\n
How should evaluation points be written for select_A_E?
short_answer:\n
점수표: A=점수, B=점수, C=점수, D=점수, E=점수
```

### 34

```
question:\n
How should evaluation points be written for select_1_5?
short_answer:\n
점수표: 1=점수, 2=점수, 3=점수, 4=점수, 5=점수
```

### 35

```
question:\n
What is the strict ordering of sections in the full QA format?
short_answer:\n
question → translate → answer_type → remind → reasoning(facts,candidates,criteria,analysis) → evaluation → answer
```

### 36

```
question:\n
What should be avoided inside facts:\n?
short_answer:\n
Any invented information not present in the question.
```

### 37

```
question:\n
Where should the final decision be expressed?
short_answer:\n
answer:\n only (and must match answer_type constraints).
```

### 38

```
question:\n
If answer_type is word, can answer contain spaces?
short_answer:\n
No, it must be exactly one Korean word with no spaces.
```

### 39

```
question:\n
If answer_type is sentence, can answer contain multiple sentences?
short_answer:\n
No, exactly one Korean sentence.
```

### 40

```
question:\n
What is the purpose of short_answer:\n in format QA fine-tuning?
short_answer:\n
Teach the rules with minimal output: only question:\n and short_answer:\n.
```

---

If you want, I can generate an additional batch of **“question about each item”** that goes deeper (e.g., edge cases: missing options, ambiguous answer_type, CRLF vs LF normalization, validator rules, illegal characters) while keeping every sample strictly in `question:\n + short_answer:\n` form.
