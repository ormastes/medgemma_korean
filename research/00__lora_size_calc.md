Below are **concrete parameter counts** for **MedGemma 4B (small)** and **MedGemma 27B (big)** for two things:

1. **Embedding growth** when you add **+30,000 tokens**
2. **LoRA parameter budget** (typical “attention-only” LoRA on Q/K/V/O)

I’m using the published Hugging Face configs:

* **MedGemma 4B**: `hidden_size=2560`, `num_hidden_layers=34`, `head_dim=256`, `num_attention_heads=8`, `num_key_value_heads=4`. ([Hugging Face][1])
* **MedGemma 27B (text)**: `hidden_size=5376`, `num_hidden_layers=62`, `head_dim=128`, `num_attention_heads=32`, `num_key_value_heads=16`. ([Hugging Face][2])

---

## A) If you add **30k new tokens**: embedding parameters added

**Rule:** added embedding parameters = `K × hidden_size` where `K=30,000`.

### MedGemma 4B (hidden=2560)

* **Input embedding added params**: `30,000 × 2,560 = 76,800,000`
* **BF16/FP16 memory** (2 bytes/param): ~**153.6 MB**
* If **LM head is untied** (separate output matrix), add the same again → **~307.2 MB** total.

### MedGemma 27B (hidden=5376)

* **Input embedding added params**: `30,000 × 5,376 = 161,280,000`
* **BF16/FP16 memory**: ~**322.6 MB**
* If **LM head is untied** → **~645.1 MB** total.

> Note: Some stacks pad vocab to a multiple (e.g., 128/256) which can slightly increase the row count beyond +30k, but the above is the direct +30k calculation.

---

## B) LoRA params for MedGemma 4B vs 27B (attention Q/K/V/O)

### Assumption (standard practice)

LoRA on **Q, K, V, O** projections in every layer.
LoRA parameters per linear layer are approximately:

`params = r × (d_in + d_out)`

In GQA models, `d_out` differs for Q vs K/V:

* `Q_out = num_attention_heads × head_dim`
* `KV_out = num_key_value_heads × head_dim`
* `O_out = hidden_size`
* `d_in = hidden_size` for all of these

This matches how Gemma-family attention is typically parameterized and is consistent with the configs above. ([Hugging Face][1])

---

### MedGemma 4B LoRA params

Config pieces: hidden=2560, heads=8, head_dim=256 → `Q_out=2048`; kv_heads=4 → `KV_out=1024`; layers=34. ([Hugging Face][1])

Per layer LoRA params:

* Q: `r × (2560 + 2048) = r × 4608`
* K: `r × (2560 + 1024) = r × 3584`
* V: same `r × 3584`
* O: `r × (2560 + 2560) = r × 5120`

Total per layer = `r × 16,896`
Total model = `34 × r × 16,896 = r × 574,464`

So:

* **r=8**  → **4,595,712 params**
* **r=16** → **9,191,424 params**
* **r=32** → **18,382,848 params**

Memory (BF16/FP16): multiply by 2 bytes:

* r=16 is ~**18.4 MB** (just adapter weights)

---

### MedGemma 27B LoRA params

Config pieces: hidden=5376, heads=32, head_dim=128 → `Q_out=4096`; kv_heads=16 → `KV_out=2048`; layers=62. ([Hugging Face][2])

Per layer LoRA params:

* Q: `r × (5376 + 4096) = r × 9472`
* K: `r × (5376 + 2048) = r × 7424`
* V: same `r × 7424`
* O: `r × (5376 + 5376) = r × 10,752`

Total per layer = `r × 35,072`
Total model = `62 × r × 35,072 = r × 2,174,464`

So:

* **r=8**  → **17,395,712 params**
* **r=16** → **34,791,424 params**
* **r=32** → **69,582,848 params**

Memory (BF16/FP16):

* r=16 is ~**69.6 MB** (adapter weights only)

---

## Practical recommendation for “+30k tokens” Korean adaptation

* **MedGemma 4B**: LoRA **r=16** is a strong default (≈9.2M params) plus +30k embedding rows (≈76.8M params).
* **MedGemma 27B**: LoRA **r=8 or r=16** depending on data size; r=16 is ≈34.8M params, while +30k embedding rows cost ≈161.3M params.

If you tell me whether your MedGemma implementation uses **tied embeddings** (input embedding == lm head weight), I’ll give you the **exact total added VRAM** including optimizer states (Adam vs 8-bit Adam) for both 4B and 27B.

[1]: https://huggingface.co/mlx-community/medgemma-4b-it-bf16/blob/main/config.json?utm_source=chatgpt.com "config.json · mlx-community/medgemma-4b-it-bf16 at main"
[2]: https://huggingface.co/unsloth/medgemma-27b-text-it-bnb-4bit/blob/main/config.json?utm_source=chatgpt.com "config.json · unsloth/medgemma-27b-text-it-bnb-4bit at main"
