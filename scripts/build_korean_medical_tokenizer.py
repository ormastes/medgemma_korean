#!/usr/bin/env python3
"""
Build Korean Medical Tokenizer

1. Download CC-100 Korean corpus
2. Extract medical terms from existing datasets
3. Expand medical dictionary
4. Train tokenizer with medical vocabulary
"""

import os
import re
import json
from pathlib import Path
from collections import Counter
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DICT_DIR = DATA_DIR / "bilingual_dict"
CORPUS_DIR = DATA_DIR / "corpus"
OUTPUT_DIR = DATA_DIR / "tokenizer_corpus"

CORPUS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Korean medical term patterns
MEDICAL_PATTERNS = [
    r'[가-힣]+증',        # ~증 (증상, 증후군)
    r'[가-힣]+염',        # ~염 (염증)
    r'[가-힣]+암',        # ~암 (암)
    r'[가-힣]+병',        # ~병 (병)
    r'[가-힣]+균',        # ~균 (세균)
    r'[가-힣]+제',        # ~제 (약제)
    r'[가-힣]+술',        # ~술 (수술)
    r'[가-힣]+선',        # ~선 (분비선)
    r'[가-힣]+막',        # ~막 (막)
    r'[가-힣]+관',        # ~관 (혈관)
    r'[가-힣]+낭',        # ~낭 (낭종)
    r'[가-힣]+종',        # ~종 (종양)
    r'[가-힣]+경',        # ~경 (내시경)
    r'[가-힣]+약',        # ~약
    r'[가-힣]+요법',      # ~요법
    r'[가-힣]+치료',      # ~치료
    r'[가-힣]+검사',      # ~검사
    r'[가-힣]+진단',      # ~진단
    r'[가-힣]+증후군',    # ~증후군
    r'[가-힣]+질환',      # ~질환
    r'[가-힣]+장애',      # ~장애
    r'[가-힣]+감염',      # ~감염
    r'[가-힣]+세포',      # ~세포
    r'[가-힣]+호르몬',    # ~호르몬
    r'[가-힣]+효소',      # ~효소
    r'[가-힣]+항체',      # ~항체
    r'[가-힣]+항원',      # ~항원
    r'[가-힣]+바이러스',  # ~바이러스
]


def download_cc100_korean(max_samples=500000):
    """Download CC-100 Korean corpus."""
    print("=" * 60)
    print("Downloading CC-100 Korean corpus...")
    print("=" * 60)

    output_path = CORPUS_DIR / "cc100_korean.txt"

    if output_path.exists():
        print(f"Already exists: {output_path}")
        return output_path

    try:
        # Try CC-100
        ds = load_dataset("statmt/cc100", "ko", split="train", streaming=True)

        texts = []
        for i, item in enumerate(tqdm(ds, desc="Downloading", total=max_samples)):
            if i >= max_samples:
                break
            texts.append(item["text"])

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(texts))

        print(f"Saved {len(texts)} texts to {output_path}")
        return output_path

    except Exception as e:
        print(f"CC-100 failed: {e}")
        print("Trying alternative: Korean Wikipedia...")

        try:
            ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train")
            texts = [item["text"][:5000] for item in tqdm(ds, desc="Wikipedia")][:max_samples]

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(texts))

            print(f"Saved {len(texts)} Wikipedia texts to {output_path}")
            return output_path

        except Exception as e2:
            print(f"Wikipedia also failed: {e2}")
            return None


def extract_medical_terms_from_datasets():
    """Extract medical terms from our Korean medical datasets."""
    print("\n" + "=" * 60)
    print("Extracting medical terms from datasets...")
    print("=" * 60)

    medical_terms = Counter()

    # Datasets to scan
    dataset_paths = [
        DATA_DIR / "processed" / "korean_medical_mcq_filtered" / "train",
        DATA_DIR / "processed" / "korean_medical_mcq_filtered" / "verification",
        DATA_DIR / "processed" / "korean_medical_lm" / "train",
    ]

    for ds_path in dataset_paths:
        if not ds_path.exists():
            print(f"  Skipping (not found): {ds_path}")
            continue

        print(f"  Scanning: {ds_path}")

        try:
            ds = load_from_disk(str(ds_path))

            for item in tqdm(ds, desc="Extracting", leave=False):
                text = item.get("text", "") or item.get("question", "") or ""

                # Extract terms matching patterns
                for pattern in MEDICAL_PATTERNS:
                    matches = re.findall(pattern, text)
                    for match in matches:
                        if len(match) >= 2:  # At least 2 chars
                            medical_terms[match] += 1

        except Exception as e:
            print(f"  Error: {e}")

    # Filter by frequency (at least 5 occurrences)
    filtered_terms = {term: count for term, count in medical_terms.items() if count >= 5}

    print(f"\nExtracted {len(filtered_terms)} medical terms (freq >= 5)")

    # Save
    terms_path = DICT_DIR / "extracted_medical_terms.json"
    with open(terms_path, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(filtered_terms.items(), key=lambda x: -x[1])), f,
                  ensure_ascii=False, indent=2)

    print(f"Saved to: {terms_path}")
    return filtered_terms


def expand_medical_dictionary(extracted_terms):
    """Expand medical dictionary with extracted terms."""
    print("\n" + "=" * 60)
    print("Expanding medical dictionary...")
    print("=" * 60)

    # Load existing dictionary
    existing_dict_path = DICT_DIR / "bilingual_medical_dict.json"
    with open(existing_dict_path, encoding="utf-8") as f:
        existing_dict = json.load(f)

    print(f"Existing dictionary: {len(existing_dict)} terms")

    # Get Korean terms from existing dict
    existing_korean = set(existing_dict.values())

    # Add new terms (Korean only, no English mapping needed for tokenizer)
    new_terms = []
    for term in extracted_terms:
        if term not in existing_korean and len(term) >= 2:
            new_terms.append(term)

    print(f"New terms to add: {len(new_terms)}")

    # Create expanded dictionary
    expanded_dict = {
        "bilingual": existing_dict,
        "korean_medical_terms": sorted(new_terms),
        "total_korean_terms": len(existing_korean) + len(new_terms)
    }

    expanded_path = DICT_DIR / "expanded_medical_dict.json"
    with open(expanded_path, "w", encoding="utf-8") as f:
        json.dump(expanded_dict, f, ensure_ascii=False, indent=2)

    print(f"Expanded dictionary saved to: {expanded_path}")
    print(f"Total Korean medical terms: {expanded_dict['total_korean_terms']}")

    return expanded_dict


def create_tokenizer_corpus(cc100_path, expanded_dict):
    """Create combined corpus for tokenizer training."""
    print("\n" + "=" * 60)
    print("Creating tokenizer corpus...")
    print("=" * 60)

    corpus_texts = []

    # 1. Add CC-100 Korean
    if cc100_path and cc100_path.exists():
        print("Adding CC-100 Korean...")
        with open(cc100_path, encoding="utf-8") as f:
            cc100_texts = f.read().split("\n")
        corpus_texts.extend(cc100_texts[:200000])  # Limit to 200K
        print(f"  Added {min(len(cc100_texts), 200000)} texts from CC-100")

    # 2. Add medical terms as repeated examples
    print("Adding medical terms...")
    korean_terms = list(expanded_dict["bilingual"].values())
    korean_terms.extend(expanded_dict["korean_medical_terms"])

    # Create sentences with medical terms
    medical_sentences = []
    for term in korean_terms:
        # Create example sentences
        medical_sentences.extend([
            f"{term}에 대해 설명해주세요.",
            f"{term}의 증상은 무엇입니까?",
            f"{term} 치료법을 알려주세요.",
            f"환자가 {term}으로 진단받았습니다.",
            f"{term}은 의료 용어입니다.",
        ])

    corpus_texts.extend(medical_sentences)
    print(f"  Added {len(medical_sentences)} medical term sentences")

    # 3. Add existing medical corpus
    existing_corpus = DATA_DIR / "raw" / "korean_corpus_for_tokenizer.txt"
    if existing_corpus.exists():
        print("Adding existing tokenizer corpus...")
        with open(existing_corpus, encoding="utf-8") as f:
            existing_texts = f.read().split("\n")
        corpus_texts.extend(existing_texts)
        print(f"  Added {len(existing_texts)} existing texts")

    # Save combined corpus
    output_path = OUTPUT_DIR / "korean_medical_tokenizer_corpus.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(corpus_texts))

    print(f"\nTotal corpus size: {len(corpus_texts)} texts")
    print(f"Saved to: {output_path}")

    # Save summary
    summary = {
        "total_texts": len(corpus_texts),
        "cc100_texts": min(len(cc100_texts) if cc100_path else 0, 200000),
        "medical_sentences": len(medical_sentences),
        "medical_terms": len(korean_terms),
        "corpus_path": str(output_path)
    }

    with open(OUTPUT_DIR / "corpus_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return output_path


def main():
    print("=" * 60)
    print("Korean Medical Tokenizer Builder")
    print("=" * 60)

    # Step 1: Download CC-100 Korean
    cc100_path = download_cc100_korean(max_samples=300000)

    # Step 2: Extract medical terms from datasets
    extracted_terms = extract_medical_terms_from_datasets()

    # Step 3: Expand medical dictionary
    expanded_dict = expand_medical_dictionary(extracted_terms)

    # Step 4: Create tokenizer corpus
    corpus_path = create_tokenizer_corpus(cc100_path, expanded_dict)

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print(f"Tokenizer corpus: {corpus_path}")
    print("\nNext step: Run tokenizer training with this corpus")


if __name__ == "__main__":
    main()
