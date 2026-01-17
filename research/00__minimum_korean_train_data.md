# Minimum Korean Training Data Requirements for LLMs

**Research Date:** 2026-01-17
**Purpose:** Determine optimal data size and sources for Korean LLM training

---

## Executive Summary

**Korean Wikipedia alone is NOT sufficient for Korean LLM training.**

- Korean Wikipedia: ~200-700MB compressed (~2-3GB uncompressed) = **~0.5-1B tokens**
- Minimum effective: **2B tokens** (with vocabulary expansion techniques)
- Recommended: **10-100B tokens** for continual pretraining
- Optimal: **100B+ tokens** with quality filtering

---

## Data Size Requirements by Training Approach

| Approach | Data Size | Tokens | Example Models | Notes |
|----------|-----------|--------|----------------|-------|
| **Minimal continual pretraining** | ~10-20GB | 2-5B | EEVE-Korean | Requires vocabulary expansion |
| **Effective continual pretraining** | ~50-200GB | 10-50B | RedWhale | Good balance of cost/performance |
| **Optimal continual pretraining** | ~300GB+ | 50-100B | Thunder-LLM | Best results for adapted models |
| **Korean-only pretraining** | ~500GB+ | 300B+ | Polyglot-Ko | Limited by data scarcity |
| **Full-scale Korean LLM** | ~2TB+ | 1T+ | 42dot_LLM, EXAONE | State-of-the-art |

---

## Key Research Papers and Models

### 1. Thunder-LLM (2025) - Most Efficient Approach

**Paper:** [Thunder-LLM: Efficiently Adapting LLMs to Korean with Minimal Resources](https://arxiv.org/html/2506.21595v1)

**Training Data:**
- **Total:** 102B tokens (50B Korean + 50B English, 1:1 ratio)
- **Raw data collected:** 3TB
  - Naver (blogs, cafés, news, Q&A): 2.7TB
  - Daum (cafés, news): 624.8GB
  - Tistory (blogs): 16.7GB
  - AI Hub: 17.4GB
  - KISTI (scientific articles): 27.9GB

**Data Processing:**
- Rule-based preprocessing: **~45% discarded**
- Deduplication: **~10.7% removed**
- Model-based perplexity filtering (using Korean Wikipedia as reference)
- Final: ~50B Korean tokens after filtering

**Key Findings:**
- Quality filtering is critical — nearly half of raw web data is low quality
- 1:1 English-Korean ratio maintains both language capabilities
- ~50B Korean tokens sufficient for state-of-the-art Korean performance

---

### 2. EEVE-Korean (Yanolja, 2024) - Minimum Viable Approach

**Paper:** [Efficient and Effective Vocabulary Expansion](https://arxiv.org/html/2402.14714v1)
**Model:** [EEVE-Korean-10.8B-v1.0](https://huggingface.co/yanolja/EEVE-Korean-10.8B-v1.0)

**Training Data:**
- **Only 2B tokens** of Korean web-crawled data
- Used [KOREAN-WEBTEXT](https://huggingface.co/datasets/HAERAE-HUB/KOREAN-WEBTEXT) (2B tokens) as reference

**Key Techniques:**
1. **Vocabulary expansion:** Added 8,960 Korean tokens based on frequency
   - Processed 100GB Korean corpus for token frequency
   - Selected tokens appearing ≥6,000 times
2. **7-stage progressive training:**
   - Stage 1-3: Train only new embeddings (frozen base)
   - Stage 4-7: Gradually unfreeze parameters

**Key Findings:**
- Vocabulary expansion compensates for limited data
- Progressive training prevents catastrophic forgetting
- 2B tokens sufficient when combined with proper techniques
- Achieved #1 on Open Ko-LLM Leaderboard (January 2024)

---

### 3. RedWhale (2024) - Quality-Focused Approach

**Paper:** [RedWhale: An Adapted Korean LLM Through Efficient Continual Pretraining](https://arxiv.org/html/2408.11294v1)

**Training Data:**
| Source | Raw Size | Processed Size |
|--------|----------|----------------|
| AI Hub | 43.9GB | - |
| Web-crawled | 77.7GB | - |
| **Total raw** | **121.6GB** | **43.9GB** |
| **Final tokens** | - | **9.7B tokens** |

**Key Findings:**
- After quality filtering: 34.6 million documents, 9.7B tokens
- **Model showed no convergence at 9.7B tokens** — more data would help
- Quality over quantity is emphasized
- ~64% of raw data discarded through filtering

---

### 4. Polyglot-Ko (EleutherAI, 2023) - Korean-Only Baseline

**Paper:** [A Technical Report for Polyglot-Ko](https://huggingface.co/papers/2306.02254)

**Training Data:**
- **1.2TB of Korean data** (monolingual)
- Trained models: 1.3B, 3.8B, 5.8B, 12.8B parameters
- Estimated <300B tokens

**Key Findings:**
- Korean-only pretraining faces significant data scarcity
- Despite training longer than Chinchilla scaling laws suggest, underperformed compared to multilingual approaches
- Mixed corpora (Korean + English + Code) is more effective than Korean-only

---

### 5. EXAONE-3 (LG AI Research, 2024) - State-of-the-Art

**Model:** [EXAONE-3-7.8B](https://huggingface.co/blog/amphora/navigating-ko-llm-research-1)

**Training Data:**
- **8 trillion tokens** (mixed Korean, English, code)
- Proprietary data from LG

**Key Findings:**
- Massive scale achieves best results
- Proprietary data provides advantage over public-only approaches

---

## Data Size Comparison Table

| Model | Year | Tokens | Raw Data | Approach | Performance |
|-------|------|--------|----------|----------|-------------|
| EEVE-Korean | 2024 | 2B | ~20GB | Vocabulary expansion | Good |
| RedWhale | 2024 | 9.7B | 122GB→44GB | Quality filtering | Good |
| Polyglot-Ko | 2023 | <300B | 1.2TB | Korean-only | Moderate |
| Thunder-LLM | 2025 | 100B | 3TB→filtered | Mixed 1:1 | Excellent |
| 42dot_LLM | 2023 | 1T+ | - | Mixed | Excellent |
| EXAONE-3 | 2024 | 8T | - | Mixed (proprietary) | State-of-the-art |

---

## Korean Wikipedia Statistics

| Metric | Value |
|--------|-------|
| Articles | ~700,000+ |
| Total documents (incl. redirects) | ~2.7 million |
| Compressed dump size | ~200-700MB |
| Uncompressed text | ~2-3GB |
| **Estimated tokens** | **~0.5-1B** |

**Conclusion:** Korean Wikipedia alone provides only ~0.5-1B tokens, which is **insufficient** for meaningful Korean language learning.

---

## Recommended Data Sources

### Tier 1: Essential (Public)

| Source | Size | Quality | Access |
|--------|------|---------|--------|
| Korean Wikipedia | ~3GB | High | [dumps.wikimedia.org](https://dumps.wikimedia.org/kowiki/) |
| NamuWiki | ~18GB | Medium-High | Dump available |
| C4 Korean | ~3GB | Medium | HuggingFace |
| AI Hub | ~44GB | High | [aihub.or.kr](https://aihub.or.kr) |
| KOREAN-WEBTEXT | 2B tokens | Curated | [HuggingFace](https://huggingface.co/datasets/HAERAE-HUB/KOREAN-WEBTEXT) |

### Tier 2: Recommended (Web Crawl)

| Source | Estimated Size | Quality | Notes |
|--------|----------------|---------|-------|
| Naver blogs/cafés | 500GB-2TB | Varies | Largest Korean web source |
| Daum cafés/news | 200-600GB | Varies | Second largest |
| Tistory blogs | 10-50GB | Medium | Tech-focused |
| Korean news sites | 50-200GB | High | Formal language |

### Tier 3: Domain-Specific

| Source | Size | Domain |
|--------|------|--------|
| KISTI | ~28GB | Scientific articles |
| Korean medical data | Varies | Medical domain |
| Korean legal texts | Varies | Legal domain |

---

## Data Quality Filtering Pipeline

Based on Thunder-LLM's approach:

```
Raw Korean Web Data (3TB)
        ↓
1. Rule-based preprocessing (-45%)
   - Remove duplicates, ads, navigation
   - Filter by length, language detection
        ↓
2. Deduplication (-10.7%)
   - MinHash/SimHash for near-duplicates
   - Exact match removal
        ↓
3. Model-based filtering
   - Train KenLM on Korean Wikipedia
   - Filter by perplexity score
        ↓
Final: ~50B Korean tokens (~1TB)
```

**Expected yield:** ~30-40% of raw web data after filtering

---

## Recommendations by Use Case

### Case 1: Minimal Budget (Medical Domain Focus)

**Target:** 2-10B tokens
**Strategy:** Vocabulary expansion + staged training (EEVE approach)

```
Data:
- Korean Wikipedia: 1B tokens
- NamuWiki: 5-6B tokens
- Medical-specific data: 1-2B tokens
Total: ~8-9B tokens
```

**Requirements:**
- Add 5,000-10,000 Korean tokens to vocabulary
- Use 7-stage progressive training
- Freeze base model initially

---

### Case 2: Balanced Approach (Recommended)

**Target:** 20-50B tokens
**Strategy:** Quality-filtered continual pretraining

```
Data:
- Korean Wikipedia: 1B tokens
- NamuWiki: 6B tokens
- C4 Korean: 1B tokens
- AI Hub: 10-15B tokens
- Filtered web crawl: 10-20B tokens
Total: ~30-40B tokens
```

**Requirements:**
- Implement perplexity-based filtering
- Use 1:1 Korean-English ratio
- Gradient checkpointing for memory efficiency

---

### Case 3: Optimal Performance

**Target:** 50-100B tokens
**Strategy:** Thunder-LLM approach

```
Data:
- All public sources: ~20B tokens
- Large-scale web crawl (Naver, Daum): 50-80B tokens
- Quality filtering pipeline
Total: ~50-100B tokens (after filtering)
```

**Requirements:**
- Significant compute budget
- Web crawling infrastructure
- Quality filtering pipeline (KenLM, perplexity)

---

## Current Project Data Assessment

### Available Data (`data/01_raw/`)

| Source | Size | Estimated Tokens |
|--------|------|------------------|
| C4 Korean | ~3GB | ~1B tokens |
| NamuWiki | ~18GB | ~5-6B tokens |
| Korean Wikipedia | ~3GB | ~1B tokens |
| **Total** | **~24GB** | **~7-8B tokens** |

### Assessment

| Criterion | Status |
|-----------|--------|
| Minimum viable (2B) | ✅ Exceeded |
| Effective training (10-50B) | ⚠️ Below range |
| Optimal (50-100B) | ❌ Insufficient |

### Recommendation

Current data (~7-8B tokens) is in the **"marginal"** range:
- Will show meaningful improvement in Korean
- Not optimal for best performance
- Consider adding AI Hub data (+10-15B tokens)
- Medical-specific data compensates for general data limitations

---

## Key Takeaways

1. **Korean Wikipedia alone is insufficient** (~1B tokens vs. 2B+ minimum needed)

2. **Quality matters more than quantity** — 45-65% of raw web data is low quality

3. **Vocabulary expansion is crucial** — Add 5,000-10,000 Korean-specific tokens

4. **Mixed training works better** — 1:1 Korean-English ratio outperforms Korean-only

5. **Staged training prevents forgetting** — Progressive unfreezing from embeddings to full model

6. **Minimum effective sizes:**
   - With vocabulary expansion: **2B tokens**
   - Standard continual pretraining: **10B+ tokens**
   - Optimal results: **50-100B tokens**

7. **For domain-specific (medical):** General Korean foundation + domain data can compensate for smaller general corpus

---

## References

1. **Thunder-LLM** (2025): [arXiv:2506.21595](https://arxiv.org/html/2506.21595v1)
2. **EEVE-Korean** (2024): [arXiv:2402.14714](https://arxiv.org/html/2402.14714v1), [HuggingFace](https://huggingface.co/yanolja/EEVE-Korean-10.8B-v1.0)
3. **RedWhale** (2024): [arXiv:2408.11294](https://arxiv.org/html/2408.11294v1)
4. **Polyglot-Ko** (2023): [arXiv:2306.02254](https://huggingface.co/papers/2306.02254)
5. **Navigating Korean LLM Research**: [HuggingFace Blog](https://huggingface.co/blog/amphora/navigating-ko-llm-research-1)
6. **KOREAN-WEBTEXT Dataset**: [HuggingFace](https://huggingface.co/datasets/HAERAE-HUB/KOREAN-WEBTEXT)
7. **Wikipedia Statistics**: [Wikipedia:Size_of_Wikipedia](https://en.wikipedia.org/wiki/Wikipedia:Size_of_Wikipedia)
8. **Wikimedia Dumps**: [dumps.wikimedia.org/kowiki](https://dumps.wikimedia.org/kowiki/)
