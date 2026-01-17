Short answer first, then justification.

## **How many new tokens are typically added for Korean?**

**Practical range (most common):**

* **≈ 5,000 – 15,000 new tokens**

**Aggressive / Korean-first models:**

* **≈ 20,000 – 30,000 new tokens**

**Minimal / domain-only extension:**

* **≈ 500 – 3,000 tokens**

---

## Why these numbers (grounded in existing practice)

### 1) Empirical behavior of English-centric tokenizers

* English-centric BPE/SentencePiece tokenizers fragment Korean **3–5× more** than English.
* This usually means the base vocab is missing **several thousand high-frequency Korean subwords**.

To materially reduce fragmentation, you must cover:

* common syllable blocks (음절 단위)
* frequent morphemes (조사, 어미, 접사)
* common named entities

That empirically lands in the **low-thousands to low-tens-of-thousands** range.

---

### 2) What existing Korean adaptation papers actually do (implicitly)

While papers rarely say “we added exactly X tokens” in the abstract, tokenizer analyses and released configs show patterns:

* **Tokenizer extension (English → Korean bilingual)**
  → usually **+8k ~ +16k tokens**

* **Balanced multilingual tokenizer (English + CJK)**
  → Korean share is typically **~5k–10k tokens**

* **Korean-centric tokenizer (trained from scratch or heavily extended)**
  → Korean coverage often **20k+ tokens**

This aligns with what Thunder-LLM-style pipelines and LLM-jp / Swallow-style CPT models do in practice.

---

### 3) Rule-of-thumb decision table (use this)

| Your goal                                              | New tokens to add |
| ------------------------------------------------------ | ----------------- |
| Only must-have terms, markers, product names           | **500 – 2k**      |
| Improve Korean token efficiency, keep English dominant | **5k – 10k**      |
| Strong Korean performance, bilingual model             | **10k – 15k**     |
| Korean-first / near-native fluency                     | **20k – 30k**     |

---

## Strong recommendation for your setup

Based on your earlier constraints:

* existing embedding
* small but important Korean dataset
* must-add tokens
* want comparable weight to large data

**Best starting point:**

> **Add ~8,000–12,000 Korean tokens**

Then:

1. initialize new embedding rows with **mean of old embeddings**
2. up-weight important sentences
3. warm-up embeddings first, then CPT

This gives **real gains** without destabilizing the model or bloating inference cost.

---

## One critical warning (very important)

Adding **too many tokens** without enough Korean data:

* slows inference
* wastes parameters
* causes under-trained embeddings

If your Korean corpus is:

* **< 100M tokens** → stay **≤ 10k new tokens**
* **> 300M tokens** → **15k–25k tokens** is safe

---

If you want, next I can:

* give a **script to estimate optimal token count from your corpus**
* show **how to measure token fragmentation before/after**
* or help you decide **BPE vs unigram vs syllable-aware tokenizer for Korean**

Just tell me which one.
