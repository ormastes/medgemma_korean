#!/usr/bin/env python3
"""
Translate KorMedLawQA from English to Korean using DeepSeek on TITAN RTX
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(device="cuda:1"):
    """Load NLLB translation model"""
    print(f"Loading NLLB translation model on {device}...")

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = "facebook/nllb-200-distilled-600M"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    # Set source language
    tokenizer.src_lang = "eng_Latn"

    return model, tokenizer

def translate_to_korean(model, tokenizer, text, max_new_tokens=512):
    """Translate English text to Korean using NLLB"""

    max_length = 512
    forced_bos_token_id = tokenizer.convert_tokens_to_ids("kor_Hang")

    # Simple sentence splitting for long texts
    if len(text) > 800:
        # Split into sentences
        sentences = text.replace('. ', '.|').split('|')
        translated_parts = []
        for sent in sentences:
            if sent.strip():
                inputs = tokenizer(sent.strip(), return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=max_length)
                translated_parts.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        return ' '.join(translated_parts)
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=max_length)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

def translate_sample(model, tokenizer, sample):
    """Translate a single KorMedLawQA sample"""

    translated = {}

    # Translate question
    translated['question'] = translate_to_korean(model, tokenizer, sample['question'])

    # Translate options
    translated['options'] = []
    for opt in sample['options']:
        translated['options'].append(translate_to_korean(model, tokenizer, opt, max_new_tokens=256))

    # Keep answer as is (A, B, C, D, E)
    translated['answer'] = sample['answer']

    # Translate reasoning
    if 'reasoning' in sample:
        translated['reasoning'] = translate_to_korean(model, tokenizer, sample['reasoning'])

    # Translate article
    if 'article' in sample:
        translated['article'] = translate_to_korean(model, tokenizer, sample['article'])

    # Keep law_title but add Korean version
    if 'law_title' in sample:
        translated['law_title_en'] = sample['law_title']
        translated['law_title'] = translate_to_korean(model, tokenizer, sample['law_title'], max_new_tokens=128)

    return translated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                       default="/home/ormastes/dev/pub/medgemma_korean/data/raw/korean_datasets/kormedlawqa_new/data/medical_law_qa_dataset.jsonl")
    parser.add_argument("--output", type=str,
                       default="/home/ormastes/dev/pub/medgemma_korean/data/raw/korean_datasets/kormedlawqa_korean/data.jsonl")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model(args.device)

    # Load input data
    print(f"Loading data from {args.input}...")
    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if args.limit:
        samples = samples[:args.limit]

    print(f"Translating {len(samples)} samples...")

    # Check for existing checkpoint
    checkpoint_file = output_path.parent / "checkpoint.json"
    start_idx = 0
    translated_samples = []

    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
            start_idx = checkpoint.get('last_idx', 0) + 1
            print(f"Resuming from checkpoint at index {start_idx}")

        # Load existing translations
        if output_path.exists():
            with open(output_path) as f:
                for line in f:
                    if line.strip():
                        translated_samples.append(json.loads(line))

    # Translate
    for idx in tqdm(range(start_idx, len(samples)), desc="Translating"):
        sample = samples[idx]

        try:
            translated = translate_sample(model, tokenizer, sample)
            translated_samples.append(translated)

            # Append to output file
            with open(output_path, 'a') as f:
                f.write(json.dumps(translated, ensure_ascii=False) + '\n')

            # Save checkpoint
            if (idx + 1) % args.checkpoint_every == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({'last_idx': idx}, f)
                print(f"\nCheckpoint saved at index {idx}")

                # Clear cache
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nError at index {idx}: {e}")
            # Save checkpoint on error
            with open(checkpoint_file, 'w') as f:
                json.dump({'last_idx': idx - 1}, f)
            raise

    print(f"\nDone! Translated {len(translated_samples)} samples to {output_path}")

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

if __name__ == "__main__":
    main()
