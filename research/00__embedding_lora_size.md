Below is what I can **verify from the papers/repos themselves** about (1) **how many tokens were added**, (2) whether **embedding dimension increased**, and (3) how much **LoRA / adapter** was added, for major “Korean adaptation” projects you referenced.

---

## Key point first: “embedding dim increased?” almost never

In these projects, “increasing embedding” typically means **increasing the embedding matrix rows** (because vocab size increased), **not** increasing the **embedding hidden size / model width**.

* **Model width (e.g., 4096, 5120, …)** stays the same.
* What changes (if tokenizer is extended):

  * `token_embeddings.weight`: shape becomes `[new_vocab, hidden_size]`
  * `lm_head.weight`: shape becomes `[new_vocab, hidden_size]` (or tied/shared equivalent)

Thunder-LLM explicitly describes this kind of change: they extend vocabulary and initialize **new token embeddings** by averaging subtoken embeddings from the original tokenizer, but do not describe changing the model hidden size. ([arXiv][1])

---

## 1) Thunder-LLM (Tokenizer extension + embedding/LM-head row growth)

**What they did**

* Baseline tokenizer: **Llama 3.1 tokenizer = 128k tokens**
* Train a **Korean-specific tokenizer = 72,000 tokens**
* Merge them → **combined vocabulary = 200,000 tokens**
* “Original tokens remain the same” (they are appended/merged; not replaced). ([arXiv][1])

**So how many new tokens were added?**

* If base is **128,000** and final is **200,000**, then **added ≈ 72,000** tokens. ([arXiv][1])

**Did embedding dimension change?**

* No indication of changing hidden size; the described operation is adding rows and initializing each new row via subtoken-embedding averaging. ([arXiv][1])

**Side-effect they call out**

* English token counts stay roughly similar, but inference can slow due to **larger LM head** from increased vocab size. ([arXiv][1])

---

## 2) RedWhale (Tokenizer adaptation; “optimal” 20k Korean vocab; merged with base)

**What they did**

* They train a SentencePiece tokenizer on Korean, try vocab sizes **8k → 40k**, and find **20,000** tokens “optimal” for their corpus. ([ar5iv][2])
* They then **combine** this new tokenizer with the base tokenizer; overlap between base and new is **2,928 tokens**. ([ar5iv][2])

**So how many new tokens were added?**
They do **not** state the final merged vocabulary size as a single number in the visible section, but they provide enough to estimate:

* They describe the base tokenizer’s token-id range as **0 to 32K** (strongly implying base vocab ≈ 32k). ([ar5iv][2])
* New tokenizer vocab: **20,000**
* Overlap: **2,928**
* Estimated merged vocab ≈ `32,000 + 20,000 − 2,928 = 49,072`

So **net new tokens vs base ≈ 20,000 − 2,928 = 17,072** (estimate; depends on exact base vocab). ([ar5iv][2])

**Did embedding dimension change?**

* No. They explicitly talk about “Ratio of Embedding Complexity” as a function of embedding parameter count, which grows when vocab grows—this is **row-count growth**, not hidden-size growth. ([ar5iv][2])

**How much LoRA was added?**

* They used a staged training strategy and then a LoRA stage; their hyperparameter table shows **LoRA stage trainable params = 4.60%** (but it does not expose LoRA rank in the visible excerpt). ([ar5iv][2])

---

## 3) KoAlpaca / K(G)OAT (No tokenizer extension; adapters only)

Here, the key is: **they do not add tokens**—they reuse **Polyglot 5.8B tokenizer**.

* “KoAlpaca 5.8b base model … Tokenizer is Polyglot5.8b Tokenizer.” ([GitHub][3])

### Adapter size (LoRA vs IA3)

The K(G)OAT repo provides explicit adapter parameter counts:

* **LoRA trainable parameters:** **3,670,016**
* **IA3 trainable parameters:** **802,816** ([GitHub][3])

So for these projects:

* **New tokens added:** **0**
* **Embedding rows unchanged** (because vocab unchanged)
* Adaptation comes from **parameter-efficient fine-tuning modules**, not tokenizer/embedding extension. ([GitHub][3])

---

# Consolidated comparison (what we can actually ground)

* **Thunder-LLM:** +~72k tokens (128k → 200k), embedding/LM-head rows grow; hidden size unchanged. ([arXiv][1])
* **RedWhale:** Korean tokenizer of 20k merged with base; overlap 2,928; implies net new ≈ 17,072 tokens (estimate); LoRA stage exists, 4.60% trainable in their table. ([ar5iv][2])
* **KoAlpaca/K(G)OAT:** tokenizer unchanged (Polyglot tokenizer); adapters only: LoRA 3,670,016 params; IA3 802,816 params. ([GitHub][3])

---

## Discussion: what you should take away for your own Korean adaptation

1. **If your goal is fewer tokens per Korean sentence (cheaper inference, better context utilization):**
   do **tokenizer extension** (Thunder-LLM / RedWhale-style). Expect:

   * better Korean tokenization efficiency
   * but bigger LM head cost (esp. smaller models) ([arXiv][1])

2. **If your goal is fast/cheap instruction tuning with limited compute:**
   reuse a Korean-friendly tokenizer (like Polyglot) and do **LoRA/IA3**. You get no tokenization efficiency gain, but adapter params are tiny (hundreds of thousands to a few million). ([GitHub][3])

---

If you tell me your **base model family** (Llama 3.x? Qwen? Mistral? Polyglot?) and your **target constraint** (training GPU/time vs inference cost vs context length), I will recommend a concrete path: “extend tokenizer” vs “no extension + LoRA/IA3”, including what vocab size range is reasonable and what adapter budget to start with.

[1]: https://arxiv.org/pdf/2506.21595 "Thunder-LLM: Efficiently Adapting LLMs to Korean with Minimal Resources"
[2]: https://ar5iv.org/pdf/2408.11294 "[2408.11294] RedWhale: An Adapted Korean LLM Through Efficient Continual Pretraining"
[3]: https://github.com/Marker-Inc-Korea/K-G-OAT "GitHub - Marker-Inc-Korea/K-G-OAT: IA3방식으로 KoAlpaca를 fine tuning한 한국어 LLM모델"
