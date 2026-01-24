#!/usr/bin/env python3
"""
Download English-Korean Parallel Datasets

Datasets proven by research papers:
1. CCMatrix (ACL 2021) - "CCMatrix: Mining Billions of High-Quality Parallel Sentences on the WEB"
2. korean-parallel-corpora - Community curated from multiple sources
3. TED Talks - Well-established multilingual benchmark
4. OPUS collections - Research-grade parallel corpora

References:
- CCMatrix: https://arxiv.org/abs/1911.04944
- OPUS-MT: https://arxiv.org/abs/2212.01936
- Thunder-LLM: https://arxiv.org/abs/2506.21595
"""

import os
import json
from pathlib import Path

# Output directory
OUTPUT_DIR = Path("/home/ormastes/dev/pub/medgemma_korean/data/01_raw/03_korean_english")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_korean_parallel_corpora():
    """
    Download Moo/korean-parallel-corpora from HuggingFace
    Community-curated Korean-English parallel corpus
    """
    print("\n" + "="*60)
    print("Downloading: korean-parallel-corpora")
    print("="*60)

    from datasets import load_dataset

    output_path = OUTPUT_DIR / "korean_parallel_corpora"
    output_path.mkdir(exist_ok=True)

    try:
        ds = load_dataset("Moo/korean-parallel-corpora", trust_remote_code=True)
        print(f"Dataset info: {ds}")

        # Save to jsonl
        for split in ds.keys():
            jsonl_path = output_path / f"{split}.jsonl"
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for item in ds[split]:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Saved {split}: {len(ds[split])} samples to {jsonl_path}")

        return True, len(ds.get('train', ds[list(ds.keys())[0]]))
    except Exception as e:
        print(f"Error: {e}")
        return False, 0


def download_ted_talks():
    """
    Download TED Talks Korean-English dataset
    Well-established multilingual benchmark
    """
    print("\n" + "="*60)
    print("Downloading: TED Talks Korean-English")
    print("="*60)

    from datasets import load_dataset

    output_path = OUTPUT_DIR / "ted_talks_ko_en"
    output_path.mkdir(exist_ok=True)

    try:
        ds = load_dataset("msarmi9/korean-english-multitarget-ted-talks-task", trust_remote_code=True)
        print(f"Dataset info: {ds}")

        for split in ds.keys():
            jsonl_path = output_path / f"{split}.jsonl"
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for item in ds[split]:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Saved {split}: {len(ds[split])} samples to {jsonl_path}")

        return True, len(ds.get('train', ds[list(ds.keys())[0]]))
    except Exception as e:
        print(f"Error: {e}")
        return False, 0


def download_opus_kde4():
    """
    Download OPUS KDE4 Korean-English dataset
    Software localization corpus from OPUS project
    """
    print("\n" + "="*60)
    print("Downloading: OPUS KDE4 Korean-English")
    print("="*60)

    from datasets import load_dataset

    output_path = OUTPUT_DIR / "opus_kde4_ko_en"
    output_path.mkdir(exist_ok=True)

    try:
        ds = load_dataset("opus_kde4", lang1="en", lang2="ko", trust_remote_code=True)
        print(f"Dataset info: {ds}")

        for split in ds.keys():
            jsonl_path = output_path / f"{split}.jsonl"
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for item in ds[split]:
                    # Normalize format
                    normalized = {
                        "en": item.get("translation", {}).get("en", ""),
                        "ko": item.get("translation", {}).get("ko", "")
                    }
                    f.write(json.dumps(normalized, ensure_ascii=False) + '\n')
            print(f"Saved {split}: {len(ds[split])} samples to {jsonl_path}")

        return True, len(ds.get('train', ds[list(ds.keys())[0]]))
    except Exception as e:
        print(f"Error: {e}")
        return False, 0


def download_opus_tatoeba():
    """
    Download OPUS Tatoeba Korean-English dataset
    Sentence pairs from Tatoeba project
    """
    print("\n" + "="*60)
    print("Downloading: OPUS Tatoeba Korean-English")
    print("="*60)

    from datasets import load_dataset

    output_path = OUTPUT_DIR / "opus_tatoeba_ko_en"
    output_path.mkdir(exist_ok=True)

    try:
        ds = load_dataset("tatoeba", lang1="en", lang2="ko", trust_remote_code=True)
        print(f"Dataset info: {ds}")

        for split in ds.keys():
            jsonl_path = output_path / f"{split}.jsonl"
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for item in ds[split]:
                    normalized = {
                        "en": item.get("translation", {}).get("en", ""),
                        "ko": item.get("translation", {}).get("ko", "")
                    }
                    f.write(json.dumps(normalized, ensure_ascii=False) + '\n')
            print(f"Saved {split}: {len(ds[split])} samples to {jsonl_path}")

        return True, len(ds.get('train', ds[list(ds.keys())[0]]))
    except Exception as e:
        print(f"Error: {e}")
        return False, 0


def download_ccmatrix_sample():
    """
    Download CCMatrix Korean-English (sampled)

    Paper: "CCMatrix: Mining Billions of High-Quality Parallel Sentences on the WEB"
    Published: ACL 2021

    Note: Full CCMatrix is very large (~4.5M en-ko pairs), downloading sample
    """
    print("\n" + "="*60)
    print("Downloading: CCMatrix Korean-English (sampled)")
    print("Paper: ACL 2021 - CCMatrix: Mining Billions of High-Quality Parallel Sentences")
    print("="*60)

    from datasets import load_dataset

    output_path = OUTPUT_DIR / "ccmatrix_ko_en"
    output_path.mkdir(exist_ok=True)

    try:
        # CCMatrix is very large, use streaming and sample
        ds = load_dataset(
            "yhavinga/ccmatrix",
            "en-ko",
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        # Sample 500K pairs (manageable size but still substantial)
        MAX_SAMPLES = 500000

        jsonl_path = output_path / "train.jsonl"
        count = 0

        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in ds:
                if count >= MAX_SAMPLES:
                    break
                normalized = {
                    "en": item.get("translation", {}).get("en", item.get("en", "")),
                    "ko": item.get("translation", {}).get("ko", item.get("ko", ""))
                }
                if normalized["en"] and normalized["ko"]:
                    f.write(json.dumps(normalized, ensure_ascii=False) + '\n')
                    count += 1
                    if count % 50000 == 0:
                        print(f"  Progress: {count:,} samples...")

        print(f"Saved train: {count:,} samples to {jsonl_path}")
        return True, count

    except Exception as e:
        print(f"Error: {e}")
        return False, 0


def download_aihub_sample():
    """
    Try to download AI Hub Korean-English corpus
    Source: Korean National Information Society Agency (NIA)
    Used in: MRL-2021 paper (800K parallel sentences)
    """
    print("\n" + "="*60)
    print("Checking: AI Hub / KEAT corpus availability")
    print("Paper: MRL-2021 (Emory NLP)")
    print("="*60)

    # AI Hub requires registration, check if alternative exists on HuggingFace
    from datasets import load_dataset

    output_path = OUTPUT_DIR / "aihub_news_ko_en"
    output_path.mkdir(exist_ok=True)

    try:
        # Try jungyeul/korean-parallel-corpora (GitHub mirror)
        ds = load_dataset("jungyeul/korean-parallel-corpora", trust_remote_code=True)
        print(f"Dataset info: {ds}")

        for split in ds.keys():
            jsonl_path = output_path / f"{split}.jsonl"
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for item in ds[split]:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Saved {split}: {len(ds[split])} samples to {jsonl_path}")

        return True, len(ds.get('train', ds[list(ds.keys())[0]]))
    except Exception as e:
        print(f"Not available on HuggingFace: {e}")
        print("Note: Full KEAT corpus requires AI Hub registration at https://aihub.or.kr/")
        return False, 0


def create_readme(results):
    """Create README with dataset information"""
    readme_content = """# English-Korean Parallel Datasets

## Datasets Proven by Research Papers

| Dataset | Paper | Samples | Status |
|---------|-------|---------|--------|
"""

    dataset_info = {
        "ccmatrix_ko_en": {
            "paper": "CCMatrix: Mining Billions of High-Quality Parallel Sentences (ACL 2021)",
            "url": "https://arxiv.org/abs/1911.04944"
        },
        "korean_parallel_corpora": {
            "paper": "Community curated, used in multiple Korean NLP studies",
            "url": "https://huggingface.co/datasets/Moo/korean-parallel-corpora"
        },
        "ted_talks_ko_en": {
            "paper": "TED Talks multilingual benchmark (OPUS project)",
            "url": "https://opus.nlpl.eu/"
        },
        "opus_kde4_ko_en": {
            "paper": "OPUS-MT: Democratizing Machine Translation (2022)",
            "url": "https://arxiv.org/abs/2212.01936"
        },
        "opus_tatoeba_ko_en": {
            "paper": "Tatoeba Challenge (community translations)",
            "url": "https://tatoeba.org/"
        },
        "aihub_news_ko_en": {
            "paper": "MRL-2021: Korean-English Parallel Dataset (Emory NLP)",
            "url": "https://github.com/emorynlp/MRL-2021"
        }
    }

    for name, (success, count) in results.items():
        info = dataset_info.get(name, {"paper": "Unknown", "url": ""})
        status = f"✅ {count:,}" if success else "❌ Failed"
        readme_content += f"| {name} | {info['paper']} | {status} |\n"

    readme_content += """

## Usage

```python
import json

# Load a dataset
with open("ccmatrix_ko_en/train.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        print(f"EN: {item['en']}")
        print(f"KO: {item['ko']}")
        break
```

## References

1. **CCMatrix** (ACL 2021)
   - Paper: "CCMatrix: Mining Billions of High-Quality Parallel Sentences on the WEB"
   - Authors: Holger Schwenk, Guillaume Wenzek, et al.
   - URL: https://arxiv.org/abs/1911.04944

2. **OPUS-MT** (2022)
   - Paper: "Democratizing Machine Translation with OPUS-MT"
   - URL: https://arxiv.org/abs/2212.01936

3. **Thunder-LLM** (2025)
   - Paper: "Thunder-LLM: Efficiently Adapting LLMs to Korean with Minimal Resources"
   - Used parallel data for bilingual pretraining
   - URL: https://arxiv.org/abs/2506.21595

4. **MRL-2021** (Emory NLP)
   - Paper: "English-Korean Parallel Dataset"
   - Source: Korean NIA AI Hub (KEAT corpus)
   - URL: https://github.com/emorynlp/MRL-2021
"""

    readme_path = OUTPUT_DIR / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"\nREADME saved to {readme_path}")


def main():
    print("="*60)
    print("Downloading English-Korean Parallel Datasets")
    print("Proven by research papers for LLM training")
    print("="*60)

    results = {}

    # Download datasets in order of importance
    results["korean_parallel_corpora"] = download_korean_parallel_corpora()
    results["ted_talks_ko_en"] = download_ted_talks()
    results["opus_kde4_ko_en"] = download_opus_kde4()
    results["opus_tatoeba_ko_en"] = download_opus_tatoeba()
    results["ccmatrix_ko_en"] = download_ccmatrix_sample()
    results["aihub_news_ko_en"] = download_aihub_sample()

    # Create README
    create_readme(results)

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    total_samples = 0
    for name, (success, count) in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {name}: {count:,} samples")
        if success:
            total_samples += count

    print(f"\nTotal: {total_samples:,} parallel sentence pairs")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
