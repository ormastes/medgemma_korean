#!/usr/bin/env python3
"""
Fast Division Annotation with Batch Processing on TITAN RTX
Uses batching to speed up annotation 5-10x
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse

class FastDivisionAnnotator:
    def __init__(self, model_name: str = "deepseek-ai/deepseek-llm-7b-chat", device: str = "cuda:1", batch_size: int = 4):
        """Initialize with batch processing for TITAN RTX"""
        self.device = device
        self.batch_size = batch_size
        
        print(f"Loading model: {model_name}")
        print(f"Device: {device} (TITAN RTX)")
        print(f"Batch size: {batch_size}")
        
        # Load taxonomy
        with open('med_division.json', 'r', encoding='utf-8') as f:
            self.taxonomy = json.load(f)
        
        # 4-bit quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="eager"  # For TITAN RTX compatibility
        )
        
        print("✓ Model loaded")
    
    def create_prompt(self, question: str, answer: str) -> str:
        """Create classification prompt - improved for Korean medical accuracy"""
        prompt = f"""한국어 의료 질문을 분석하여 주요 진료과목 ID를 분류하세요.

진료과목 분류:
1=심장내과(Cardiovascular) - 심장, 혈압, 심전도, 협심증, 심근경색
2=호흡기내과(Respiratory) - 폐, 기침, 호흡곤란, 폐렴, 천식, 결핵
3=소화기내과(GI) - 위장, 간, 복통, 설사, 간염
4=신장내과(Nephrology) - 신장, 요로, 투석, 부종
5=내분비내과(Endocrine) - 당뇨, 갑상선, 호르몬, 비만
6=혈액종양내과(Hematology/Oncology) - 빈혈, 암, 백혈병, 항암
7=신경과(Neurology) - 뇌, 신경, 두통, 경련, 마비, 치매
8=감염내과(Infectious) - 감염, 발열, 패혈증, 항생제, 백신
9=응급의학(Emergency) - 쇼크, 외상, 중독, 심정지
10=의료윤리/법(Ethics/Law) - 의료윤리, 의료법, 환자권리

질문: {question[:400]}
답변: {answer[:150]}

주요 진료과목 1-2개를 선택하세요. 첫번째가 가장 주된 과목입니다.
출력형식: [숫자] 또는 [숫자,숫자]
분류:"""
        
        return prompt
    
    def annotate_batch(self, samples: list) -> list:
        """Annotate a batch of samples"""
        prompts = []
        for sample in samples:
            question = sample.get('prompt', '')[:500]
            answer = sample.get('completion', '')[:200]
            prompt = self.create_prompt(question, answer)
            prompts.append(prompt)
        
        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Generate batch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode batch
        results = []
        for i, output in enumerate(outputs):
            response = self.tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            divisions = self.parse_divisions(response)
            
            # Add to sample
            annotated = samples[i].copy()
            annotated['divisions'] = divisions
            annotated['primary_division'] = divisions[0] if divisions else "UNKNOWN"
            
            results.append(annotated)
        
        return results
    
    def parse_divisions(self, response: str) -> list:
        """Parse division IDs from response"""
        import re
        
        # Try to find [1,2,3] format
        match = re.search(r'\[(\d+(?:,\s*\d+)*)\]', response)
        if match:
            ids = [id.strip() for id in match.group(1).split(',')]
            return ids[:3]  # Max 3
        
        # Try to find just numbers
        numbers = re.findall(r'\b([1-9]|10)\b', response)
        if numbers:
            return numbers[:3]
        
        return ["UNKNOWN"]
    
    def process_file(self, input_file: str, output_file: str):
        """Process file with batch annotation"""
        print(f"\nProcessing: {input_file}")
        print(f"Output: {output_file}")
        
        # Load all samples
        samples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        
        print(f"Total samples: {len(samples)}")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Process in batches
        annotated_samples = []
        batch = []
        
        with tqdm(total=len(samples), desc="Annotating") as pbar:
            for sample in samples:
                batch.append(sample)
                
                if len(batch) >= self.batch_size:
                    results = self.annotate_batch(batch)
                    annotated_samples.extend(results)
                    
                    # Save checkpoint
                    if len(annotated_samples) % 100 == 0:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            for s in annotated_samples:
                                f.write(json.dumps(s, ensure_ascii=False) + '\n')
                    
                    batch = []
                    pbar.update(self.batch_size)
            
            # Process remaining
            if batch:
                results = self.annotate_batch(batch)
                annotated_samples.extend(results)
                pbar.update(len(batch))
        
        # Final save
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in annotated_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"✓ Saved {len(annotated_samples)} samples to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Fast division annotation with batching")
    parser.add_argument('--type', type=str, required=True, 
                       choices=['type1_text', 'type2_text_reasoning', 'type3_word', 'type4_word_reasoning'],
                       help='Data type to process')
    parser.add_argument('--model', type=str, default='deepseek-ai/deepseek-llm-7b-chat',
                       help='Model to use')
    parser.add_argument('--device', type=str, default='cuda:1',
                       help='Device (cuda:1 for TITAN RTX)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (default: 8)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Fast Division Annotation - TITAN RTX")
    print("="*70)
    
    annotator = FastDivisionAnnotator(args.model, args.device, args.batch_size)
    
    # Process train
    train_input = f'data/reviewed/{args.type}/train/data.jsonl'
    train_output = f'data/division_added/{args.type}/train.jsonl'
    annotator.process_file(train_input, train_output)
    
    # Process validation
    val_input = f'data/reviewed/{args.type}/validation/data.jsonl'
    val_output = f'data/division_added/{args.type}/validation.jsonl'
    annotator.process_file(val_input, val_output)
    
    print("\n" + "="*70)
    print(f"✓ {args.type} annotation complete!")
    print("="*70)

if __name__ == "__main__":
    main()
