#!/usr/bin/env python3
"""
Add division annotations to reviewed data using DeepSeek on A6000
Saves annotated data to data/division_added/
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

class DeepSeekDivisionAnnotator:
    def __init__(self, model_name: str = "deepseek-ai/deepseek-llm-7b-chat", device: str = "cuda:0"):
        """Initialize DeepSeek model on A6000 (cuda:0)"""
        self.device = device
        print(f"Loading DeepSeek model: {model_name}")
        print(f"Using device: {device} (A6000)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Use 4-bit quantization for memory efficiency
        # Disable FlashAttention for A6000 (Turing, not Ampere)
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="eager"  # Disable FlashAttention for A6000
        )
        
        self.model.eval()
        
        # Load medical divisions taxonomy
        with open('med_division.json', 'r', encoding='utf-8') as f:
            self.taxonomy = json.load(f)
        
        self.division_map = self._build_division_map()
        
    def _build_division_map(self) -> Dict[str, str]:
        """Build ID to name map from taxonomy"""
        div_map = {}
        for node in self.taxonomy['nodes']:
            div_map[node['id']] = node['name']
            if 'children' in node:
                self._process_children(node['children'], div_map)
        return div_map
    
    def _process_children(self, children, div_map):
        """Recursively process children nodes"""
        for child in children:
            if isinstance(child, dict):
                div_map[child['id']] = child['name']
                if 'children' in child:
                    self._process_children(child['children'], div_map)
    
    def create_annotation_prompt(self, question: str, answer: str = "") -> str:
        """Create prompt for DeepSeek to annotate divisions"""
        
        major_divs = [
            "1. Cardiovascular Medicine",
            "2. Respiratory Medicine", 
            "3. Gastroenterology and Hepatology",
            "4. Nephrology",
            "5. Endocrinology and Metabolism",
            "6. Hematology and Oncology",
            "7. Neurology",
            "8. Infectious Diseases",
            "9. Emergency and Critical Care",
            "10. Ethics, Law, and Patient Safety"
        ]
        
        prompt = f"""You are a medical taxonomy expert. Analyze this medical question and assign medical division IDs.

QUESTION:
{question}

{f"ANSWER: {answer}" if answer else ""}

MEDICAL DIVISIONS:
{chr(10).join(major_divs)}

INSTRUCTIONS:
1. Assign 1-3 division IDs (e.g., "1", "2", "8")
2. First ID MUST be PRIMARY (majority) division
3. Additional IDs are secondary if applicable
4. Provide brief reasoning

OUTPUT FORMAT (JSON only, no other text):
{{
  "divisions": ["1", "8"],
  "primary": "1",
  "reasoning": "Primary: Cardiovascular. Secondary: Infectious disease complication"
}}

JSON OUTPUT:"""

        return prompt
    
    def annotate_single(self, question: str, answer: str = "") -> Dict[str, Any]:
        """Annotate a single question with divisions"""
        
        prompt = self.create_annotation_prompt(question, answer)
        
        messages = [{"role": "user", "content": prompt}]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                
                if 'divisions' in result and 'primary' in result:
                    return result
            
            return {
                "divisions": ["UNKNOWN"],
                "primary": "UNKNOWN",
                "reasoning": f"Parse error: {response[:100]}",
                "error": True
            }
            
        except Exception as e:
            return {
                "divisions": ["UNKNOWN"],
                "primary": "UNKNOWN", 
                "reasoning": f"Error: {str(e)}",
                "error": True
            }
    
    def process_dataset(self, input_file: str, output_file: str):
        """Process dataset and add division annotations"""
        
        print(f"\nProcessing: {input_file}")
        print(f"Output to: {output_file}")
        
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        print(f"Total samples: {len(data)}")
        
        annotated = []
        errors = 0
        
        for idx, sample in enumerate(tqdm(data, desc="Annotating")):
            question = self._extract_question(sample.get('prompt', ''))
            answer = sample.get('completion', '')
            
            annotation = self.annotate_single(question, answer)
            
            sample['divisions'] = annotation.get('divisions', ["UNKNOWN"])
            sample['primary_division'] = annotation.get('primary', "UNKNOWN")
            sample['division_reasoning'] = annotation.get('reasoning', '')
            
            if annotation.get('error'):
                errors += 1
            
            annotated.append(sample)
            
            if (idx + 1) % 100 == 0:
                self._save_checkpoint(annotated, output_file)
        
        self._save_checkpoint(annotated, output_file)
        
        print(f"\nCompleted!")
        print(f"Total annotated: {len(annotated)}")
        print(f"Errors: {errors}")
        print(f"Success rate: {(len(annotated) - errors) / len(annotated) * 100:.2f}%")
        
        return annotated
    
    def _extract_question(self, prompt: str) -> str:
        """Extract question text from prompt"""
        lines = prompt.split('\n')
        question_lines = []
        capture = False
        
        for line in lines:
            if '<|im_start|>user' in line:
                capture = True
                continue
            if '<|im_end|>' in line and capture:
                break
            if capture:
                question_lines.append(line)
        
        return '\n'.join(question_lines).strip()
    
    def _save_checkpoint(self, data: List[Dict], output_file: str):
        """Save checkpoint"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Add divisions to reviewed data")
    parser.add_argument('--type', type=str, required=True, 
                       choices=['type1_text', 'type2_text_reasoning', 'type3_word', 'type4_word_reasoning', 'all'],
                       help='Data type to process')
    parser.add_argument('--model', type=str, default='deepseek-ai/deepseek-llm-7b-chat',
                       help='DeepSeek model name')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device (cuda:0 for A6000)')
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(1)}")
    else:
        print("WARNING: No GPU available!")
    
    annotator = DeepSeekDivisionAnnotator(
        model_name=args.model,
        device=args.device
    )
    
    types_to_process = ['type1_text', 'type2_text_reasoning', 'type3_word', 'type4_word_reasoning'] \
                       if args.type == 'all' else [args.type]
    
    for dtype in types_to_process:
        print(f"\n{'='*60}")
        print(f"Processing {dtype}")
        print(f"{'='*60}")
        
        for split in ['train', 'validation']:
            input_file = f'data/reviewed/{dtype}/{split}/data.jsonl'
            output_file = f'data/division_added/{dtype}/{split}.jsonl'
            
            if not os.path.exists(input_file):
                print(f"Skipping {input_file} (not found)")
                continue
            
            annotator.process_dataset(input_file, output_file)
        
        print(f"\n{dtype} completed!")


if __name__ == "__main__":
    main()
