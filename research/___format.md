Understood. `evaluation:\n` (Korean) will explicitly contain **all possible answers** with a **point/score**, and (optionally) a short Korean rationale per option. This keeps your one-character / one-word answer constraint intact, because the justification and scoring live in `evaluation:\n`, not in `answer:\n`.

Below is a precise template and examples for all answer types.

---

## 1) Required `evaluation:\n` format (Korean)

### For `select_A_E`

```
evaluation:\n
- 평가기준: <짧은 기준 2~4개>
- 점수표: A=<0~N>, B=<0~N>, C=<0~N>, D=<0~N>, E=<0~N>
- 근거요약: A(...); B(...); C(...); D(...); E(...)
```

### For `select_1_5`

```
evaluation:\n
- 평가기준: ...
- 점수표: 1=<0~N>, 2=<0~N>, 3=<0~N>, 4=<0~N>, 5=<0~N>
- 근거요약: 1(...); 2(...); 3(...); 4(...); 5(...)
```

### For `word`

You still need “possible answers with points,” so define a candidate list:

```
evaluation:\n
- 평가기준: ...
- 후보/점수: 후보1=<0~N>, 후보2=<0~N>, 후보3=<0~N>
- 근거요약: 후보1(...); 후보2(...); 후보3(...)
```

### For `sentence`

Same as `word`—score candidate sentence intents/contents, not literal full sentences:

```
evaluation:\n
- 평가기준: ...
- 후보/점수: 핵심요소1=<0~N>, 핵심요소2=<0~N>, 핵심요소3=<0~N>
- 근거요약: ...
```

---

## 2) Examples

### A) select_A_E (answer is 1 char)

```
question:\n
다음 중 당뇨병성 케톤산증(DKA) 초기 처치로 가장 적절한 것은?
A) 포도당 수액을 먼저 투여한다
B) 인슐린 정주를 먼저 시작하고 수액은 나중에 한다
C) 0.9% 생리식염수로 수액을 먼저 시작한다
D) 중탄산나트륨을 모든 경우에 즉시 투여한다
E) 칼륨은 절대 보충하지 않는다
translate:\n
Which initial management is most appropriate for DKA?
answer_type:\n
select_A_E
remind:\n
Answer must be exactly one character: A, B, C, D, or E.
reasoning:\n
facts:\n
- Initial DKA management prioritizes isotonic fluid resuscitation.
candidates:\n
- A, B, C, D, E
criteria:\n
- Initial priority (hemodynamics)
- Guideline consistency
- Avoid unsafe blanket rules
analysis:\n
Fluids first with 0.9% NS is the standard initial step.
evaluation:\n
- 평가기준: 초기 우선순위(수액), 안전성, 가이드라인 부합
- 점수표: A=0, B=0, C=3, D=0, E=0
- 근거요약: A(초기 우선순위 아님); B(수액 없이 인슐린은 위험); C(표준 초기 수액); D(중탄산은 제한적); E(K+는 모니터링 후 필요 시 보충)
answer:\n
C
```

---

### B) select_1_5 (answer is 1 char)

```
question:\n
다음 임상 상황의 위험도를 1~5로 평가하시오: 경미한 증상, 활력징후 안정, 경고증상 없음, 일상생활 가능.
translate:\n
Rate risk from 1 to 5: mild symptoms, stable vitals, no red flags, able to perform daily activities.
answer_type:\n
select_1_5
remind:\n
Answer must be exactly one character: 1, 2, 3, 4, or 5.
reasoning:\n
facts:\n
- No red flags and stable vitals indicate minimal risk.
candidates:\n
- 1, 2, 3, 4, 5
criteria:\n
- Red flags
- Vital instability
- Functional impairment
analysis:\n
This scenario matches the lowest risk category.
evaluation:\n
- 평가기준: 경고증상, 활력징후 불안정, 기능저하
- 점수표: 1=3, 2=1, 3=0, 4=0, 5=0
- 근거요약: 1(경고증상 없음/안정/일상 가능); 2(약한 위험 가정 필요); 3~5(조건 불충족)
answer:\n
1
```

---

### C) word (answer is one or few Korean words)

```
question:\n
우하복부 통증과 반발통이 있는 급성 복증에서 가장 가능성이 높은 진단은?
translate:\n
In acute abdomen with RLQ pain and rebound tenderness, what is the most likely diagnosis?
answer_type:\n
word
remind:\n
Answer must be exactly one or few Korean words.
reasoning:\n
facts:\n
- RLQ pain with rebound suggests localized peritoneal irritation.
candidates:\n
- Appendicitis, Gastroenteritis, Renal colic
criteria:\n
- Explains RLQ localization
- Explains rebound tenderness
analysis:\n
Appendicitis best fits the pattern.
evaluation:\n
- 평가기준: 우하복부 국소화, 반발통(복막자극) 설명력
- 후보/점수: 충수염=3, 장염=0, 요관결석=1
- 근거요약: 충수염(국소 복막자극과 일치); 장염(국소 반발통 설명 약함); 요관결석(통증은 가능하나 반발통은 비전형적)
answer:\n
충수염
```

---

### D) sentence (answer is one Korean sentence)

```
question:\n
크룹(croup)이 의심되는 소아에서 초기 치료의 핵심을 한 문장으로 설명하시오.
translate:\n
In suspected croup, describe the key initial treatment in one sentence.
answer_type:\n
sentence
remind:\n
Answer must be Korean and exactly one sentence.
reasoning:\n
facts:\n
- Steroids reduce upper-airway edema; nebulized epinephrine is used for moderate-to-severe cases.
candidates:\n
- Core elements: steroid, epinephrine-if-severe, avoid wrong first-line antibiotics
criteria:\n
- Must include steroid
- Must mention epinephrine for moderate-to-severe cases
- Must remain one sentence
analysis:\n
A correct one-sentence answer should cover steroid + severity-based epinephrine.
evaluation:\n
- 평가기준: 스테로이드 포함, 중등도 이상 시 에피네프린 언급, 한 문장
- 후보/점수: 스테로이드=3, 에피네프린(중등도↑)=3, 한문장=3, 항생제우선배제=1
- 근거요약: 핵심은 기도부종 감소(스테로이드) + 중증 시 즉각적 증상완화(에피네프린)이며, 불필요한 치료(항생제 우선)는 피한다
answer:\n
크룹이 의심되면 덱사메타손을 투여하고 증상이 중등도 이상이면 네뷸라이저 에피네프린을 병행한다.
```

---

If you want, I can give you a **strict validator** (regex + scoring-table parser rules) for `evaluation:\n` so your dataset builder can automatically reject any sample missing (a) all options, (b) points, or (c) Korean-only evaluation text.
