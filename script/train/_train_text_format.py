#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Text Format Templates

Centralized template definitions for all training scripts.
"""

# =============================================================================
# ANSWER TYPE DEFINITIONS
# =============================================================================

ANSWER_TYPES = {
    "select_A_E": {
        "remind": "Answer must be exactly one character: A, B, C, D, or E.",
        "valid_answers": ["A", "B", "C", "D", "E"],
    },
    "select_1_5": {
        "remind": "Answer must be exactly one character: 1, 2, 3, 4, or 5.",
        "valid_answers": ["1", "2", "3", "4", "5"],
    },
    "word": {
        "remind": "Answer must be exactly one or few Korean words.",
        "valid_answers": None,
    },
    "sentence": {
        "remind": "Answer must be Korean and exactly one sentence.",
        "valid_answers": None,
    },
}

# =============================================================================
# TRAIN_01: DICTIONARY TEMPLATE
# =============================================================================

DICT_TRAIN_TEMPLATE = """<start_of_turn>user
word(s) question:
{term}
short_answer:
<end_of_turn>
<start_of_turn>model
{definition}<end_of_turn>"""

# =============================================================================
# TRAIN_02: MCQ TEMPLATES (select_A_E)
# =============================================================================

# MCQ_TRAIN_TEMPLATE has translate field for training with translation
MCQ_TRAIN_TEMPLATE = """<start_of_turn>user
question:
다음 중 당뇨병성 케톤산증(DKA) 초기 처치로 가장 적절한 것은?
A) 포도당 수액을 먼저 투여한다
B) 인슐린 정주를 먼저 시작하고 수액은 나중에 한다
C) 0.9% 생리식염수로 수액을 먼저 시작한다
D) 중탄산나트륨을 모든 경우에 즉시 투여한다
E) 칼륨은 절대 보충하지 않는다
answer_type:
select_A_E
remind:
Answer must be exactly one character: A, B, C, D, or E.
translate:
<end_of_turn>
<start_of_turn>model
Which initial management is most appropriate for DKA?
A) Administer glucose solution first
B) Start insulin IV first, fluids later
C) Start fluids with 0.9% normal saline first
D) Give sodium bicarbonate immediately in all cases
E) Never supplement potassium
reasoning:
facts:
- Initial DKA management prioritizes isotonic fluid resuscitation.
candidates:
- A, B, C, D, E
criteria:
- Initial priority (hemodynamics)
- Guideline consistency
- Avoid unsafe blanket rules
analysis:
Fluids first with 0.9% NS is the standard initial step.
evaluation:
- 평가기준: 초기 우선순위(수액), 안전성, 가이드라인 부합
- 점수표: A=0, B=0, C=3, D=0, E=0
- 근거요약: A(초기 우선순위 아님); B(수액 없이 인슐린은 위험); C(표준 초기 수액); D(중탄산은 제한적); E(K+는 모니터링 후 필요 시 보충)
answer:
C<end_of_turn>
<start_of_turn>user
The answer is C. Start 0.9% NS fluid first.
question:
{question}
A) {A}
B) {B}
C) {C}
D) {D}
E) {E}
answer_type:
select_A_E
remind:
Answer must be exactly one character: A, B, C, D, or E.
translate:
{translate_question}
A) {translate_A}
B) {translate_B}
C) {translate_C}
D) {translate_D}
E) {translate_E}
<end_of_turn>
<start_of_turn>model
{answer}<end_of_turn>
"""

MCQ_VALIDATE_TEMPLATE = """<start_of_turn>user
question:
다음 중 당뇨병성 케톤산증(DKA) 초기 처치로 가장 적절한 것은?
A) 포도당 수액을 먼저 투여한다
B) 인슐린 정주를 먼저 시작하고 수액은 나중에 한다
C) 0.9% 생리식염수로 수액을 먼저 시작한다
D) 중탄산나트륨을 모든 경우에 즉시 투여한다
E) 칼륨은 절대 보충하지 않는다
answer_type:
select_A_E
remind:
Answer must be exactly one character: A, B, C, D, or E.
translate:
<end_of_turn>
<start_of_turn>model
Which initial management is most appropriate for DKA?
A) Administer glucose solution first
B) Start insulin IV first, fluids later
C) Start fluids with 0.9% normal saline first
D) Give sodium bicarbonate immediately in all cases
E) Never supplement potassium
reasoning:
facts:
- Initial DKA management prioritizes isotonic fluid resuscitation.
candidates:
- A, B, C, D, E
criteria:
- Initial priority (hemodynamics)
- Guideline consistency
- Avoid unsafe blanket rules
analysis:
Fluids first with 0.9% NS is the standard initial step.
evaluation:
- 평가기준: 초기 우선순위(수액), 안전성, 가이드라인 부합
- 점수표: A=0, B=0, C=3, D=0, E=0
- 근거요약: A(초기 우선순위 아님); B(수액 없이 인슐린은 위험); C(표준 초기 수액); D(중탄산은 제한적); E(K+는 모니터링 후 필요 시 보충)
answer:
C<end_of_turn>
<start_of_turn>user
The answer is C. Start 0.9% NS fluid first.
question:
{question}
A) {A}
B) {B}
C) {C}
D) {D}
E) {E}
answer_type:
select_A_E
remind:
Answer must be exactly one character: A, B, C, D, or E.
translate:
<end_of_turn>
<start_of_turn>model
"""


# =============================================================================
# TRANSLATION TEMPLATES (English-Korean)
# =============================================================================

# -----------------------------------------------------------------------------
# SIMPLE TEMPLATES (no examples, direct translation)
# -----------------------------------------------------------------------------

# Training template: translate English to Korean (simple)
TRANSLATION_EN_TO_KO_SIMPLE = """<start_of_turn>user
translate question:
{english}
translate:
<end_of_turn>
<start_of_turn>model
{korean}<end_of_turn>"""

# Training template: translate Korean to English (simple)
TRANSLATION_KO_TO_EN_SIMPLE = """<start_of_turn>user
korean translate question:
{korean}
korean translate:
<end_of_turn>
<start_of_turn>model
{english}<end_of_turn>"""

# -----------------------------------------------------------------------------
# EXAMPLE-BASED TEMPLATES (with 1-shot example, like MCQ format)
# These teach the model the expected translation format
# -----------------------------------------------------------------------------

# English to Korean with example
TRANSLATION_EN_TO_KO_TEMPLATE = """<start_of_turn>user
translate question:
The patient presented with chest pain radiating to the left arm.
translate:
<end_of_turn>
<start_of_turn>model
환자는 왼쪽 팔로 방사되는 흉통을 호소하며 내원하였다.<end_of_turn>
<start_of_turn>user
translate question:
{english}
translate:
<end_of_turn>
<start_of_turn>model
{korean}<end_of_turn>"""

# Korean to English with example
TRANSLATION_KO_TO_EN_TEMPLATE = """<start_of_turn>user
korean translate question:
환자는 왼쪽 팔로 방사되는 흉통을 호소하며 내원하였다.
korean translate:
<end_of_turn>
<start_of_turn>model
The patient presented with chest pain radiating to the left arm.<end_of_turn>
<start_of_turn>user
korean translate question:
{korean}
korean translate:
<end_of_turn>
<start_of_turn>model
{english}<end_of_turn>"""


# -----------------------------------------------------------------------------
# VALIDATION TEMPLATES (no answer provided)
# -----------------------------------------------------------------------------

# Validation: English to Korean (with example)
TRANSLATION_VALIDATE_EN_TO_KO_TEMPLATE = """<start_of_turn>user
translate question:
The patient presented with chest pain radiating to the left arm.
translate:
<end_of_turn>
<start_of_turn>model
환자는 왼쪽 팔로 방사되는 흉통을 호소하며 내원하였다.<end_of_turn>
<start_of_turn>user
translate question:
{english}
translate:
<end_of_turn>
<start_of_turn>model
"""

# Validation: Korean to English (with example)
TRANSLATION_VALIDATE_KO_TO_EN_TEMPLATE = """<start_of_turn>user
korean translate question:
환자는 왼쪽 팔로 방사되는 흉통을 호소하며 내원하였다.
korean translate:
<end_of_turn>
<start_of_turn>model
The patient presented with chest pain radiating to the left arm.<end_of_turn>
<start_of_turn>user
korean translate question:
{korean}
korean translate:
<end_of_turn>
<start_of_turn>model
"""

# Validation: Simple (no example)
TRANSLATION_VALIDATE_EN_TO_KO_SIMPLE = """<start_of_turn>user
translate question:
{english}
translate:
<end_of_turn>
<start_of_turn>model
"""

TRANSLATION_VALIDATE_KO_TO_EN_SIMPLE = """<start_of_turn>user
korean translate question:
{korean}
korean translate:
<end_of_turn>
<start_of_turn>model
"""


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# For train_01_medical_dict.py
PROMPT_TEMPLATE = DICT_TRAIN_TEMPLATE

# For train_02_kor_med_test.py
SIMPLE_PROMPT_TEMPLATE = MCQ_TRAIN_TEMPLATE
DETAILED_PROMPT_TEMPLATE = MCQ_TRAIN_TEMPLATE
TRAINING_PROMPT_TEMPLATE = MCQ_TRAIN_TEMPLATE
VALIDATION_PROMPT_TEMPLATE = MCQ_VALIDATE_TEMPLATE

# For train_01_english_korean.py (with examples)
TRANSLATION_TRAIN_TEMPLATE = TRANSLATION_EN_TO_KO_TEMPLATE
TRANSLATION_VALIDATION_TEMPLATE = TRANSLATION_VALIDATE_EN_TO_KO_TEMPLATE

# Simple aliases (no examples)
TRANSLATION_EN_TO_KO = TRANSLATION_EN_TO_KO_SIMPLE
TRANSLATION_KO_TO_EN = TRANSLATION_KO_TO_EN_SIMPLE
