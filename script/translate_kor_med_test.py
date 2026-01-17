#!/usr/bin/env python3
"""
Translate KorMedMCQA data from Korean to English using Qwen2.5-14B-Instruct.
Adds translate_question, translate_A, translate_B, translate_C, translate_D, translate_E fields.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
import argparse
from tqdm import tqdm
import re

def load_model(model_name="Qwen/Qwen2.5-14B-Instruct", device="cuda:0", use_8bit=False):
    """Load Qwen model for translation."""
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if use_8bit:
        print("Using 8-bit quantization to save memory...")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="eager",
        )
    model.eval()

    print(f"Model loaded on {device}")
    return model, tokenizer


def translate_text(model, tokenizer, korean_text: str, max_new_tokens=256) -> str:
    """Translate Korean text to English."""
    if not korean_text or not korean_text.strip():
        return ""

    prompt = f"""Translate the following Korean medical text to English.
Preserve all medical terminology accurately. Output ONLY the English translation, nothing else.

Korean: {korean_text}

English:"""

    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    translation = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return translation


def translate_batch(model, tokenizer, texts: list, max_new_tokens=256) -> list:
    """Translate multiple texts in a batch for efficiency."""
    translations = []
    for text in texts:
        translation = translate_text(model, tokenizer, text, max_new_tokens)
        translations.append(translation)
    return translations


def process_file(input_path: str, output_path: str, model, tokenizer,
                 start_idx: int = 0, checkpoint_every: int = 50):
    """Process JSONL file and add translations."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    checkpoint_path = output_path.with_suffix('.checkpoint.jsonl')

    # Load existing data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print(f"Loaded {len(data)} samples from {input_path}")

    # Load checkpoint if exists and resuming
    if start_idx > 0 and checkpoint_path.exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            processed = [json.loads(line) for line in f]
        print(f"Resuming from checkpoint with {len(processed)} processed samples")
    else:
        processed = []
        start_idx = 0

    # Process each sample
    for i, sample in enumerate(tqdm(data[start_idx:], initial=start_idx, total=len(data))):
        idx = start_idx + i

        # Skip if already has translations
        if 'translate_question' in sample:
            processed.append(sample)
            continue

        # Translate each field
        sample['translate_question'] = translate_text(
            model, tokenizer, sample.get('question', ''), max_new_tokens=512
        )

        for choice in ['A', 'B', 'C', 'D', 'E']:
            if choice in sample:
                sample[f'translate_{choice}'] = translate_text(
                    model, tokenizer, sample[choice], max_new_tokens=128
                )

        processed.append(sample)

        # Save checkpoint
        if (idx + 1) % checkpoint_every == 0:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                for item in processed:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"\nCheckpoint saved at {idx + 1} samples")

    # Save final output
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nSaved {len(processed)} translated samples to {output_path}")

    # Remove checkpoint if completed
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint file removed")

    return processed


def main():
    parser = argparse.ArgumentParser(description="Translate KorMedMCQA to English")
    parser.add_argument("--input", type=str,
                        default="data/02_refined/02_kor_med_test/train.jsonl",
                        help="Input JSONL file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file (default: overwrite input)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct",
                        help="Translation model")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--8bit", action="store_true",
                        help="Use 8-bit quantization to save memory")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start index for resuming")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Save checkpoint every N samples")

    args = parser.parse_args()

    if args.output is None:
        args.output = args.input

    # Load model
    model, tokenizer = load_model(args.model, args.device, use_8bit=getattr(args, '8bit', False))

    # Process file
    process_file(
        args.input,
        args.output,
        model,
        tokenizer,
        start_idx=args.start_idx,
        checkpoint_every=args.checkpoint_every
    )


if __name__ == "__main__":
    main()
