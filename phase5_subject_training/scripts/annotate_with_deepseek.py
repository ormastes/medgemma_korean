#!/usr/bin/env python3
"""
DeepSeek-based Medical Division Annotator
Uses DeepSeek on A6000 to annotate medical data with divisions
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
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
        
        # List major divisions for reference
        major_divs = [
            "1. Cardiovascular Medicine",
            "2. Respiratory Medicine", 
            "3. Gastroenterology and Hepatology",
            "4. Nephrology",
            "5. Endocrinology and Metabolism",
            "6. Hematology and Oncology",
            "7. Neurology",
            "8. Infectious Diseases (Cross-Cutting)",
            "9. Emergency and Critical Care (Cross-Cutting)",
            "10. Ethics, Law, and Patient Safety (Cross-Cutting)"
        ]
        
        prompt = f"""You are a medical taxonomy expert. Analyze the following medical question and assign appropriate medical division IDs.

QUESTION:
{question}

{f"ANSWER: {answer}" if answer else ""}

MEDICAL DIVISIONS:
{chr(10).join(major_divs)}

INSTRUCTIONS:
1. Assign 1-3 division IDs (e.g., "1", "2", "8")
2. First ID must be the PRIMARY (majority) division
3. Additional IDs are secondary divisions if applicable
4. Use sub-division IDs when specific (e.g., "1.4.1" for Ischemic Heart Disease)
5. Provide brief reasoning for each division

OUTPUT FORMAT (JSON):
{{
  "divisions": ["1.4.1", "8.3.1"],
  "primary": "1.4.1",
  "reasoning": "Primary: Cardiovascular - focuses on ACS/MI. Secondary: Infectious Diseases - mentions pneumonia complication"
}}

Respond with ONLY the JSON object, no other text."""

        return prompt
    
    def annotate_single(self, question: str, answer: str = "") -> Dict[str, Any]:
        """Annotate a single question with divisions"""
        
        prompt = self.create_annotation_prompt(question, answer)
        
        # Prepare input
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Parse JSON response
        try:
            # Find JSON object in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                
                # Validate structure
                if 'divisions' in result and 'primary' in result:
                    return result
            
            # If parsing failed, return error
            return {
                "divisions": ["UNKNOWN"],
                "primary": "UNKNOWN",
                "reasoning": f"Parse error: {response[:200]}",
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
        """Process entire dataset and add division annotations"""
        
        print(f"Processing: {input_file}")
        print(f"Output to: {output_file}")
        
        # Read input
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        print(f"Total samples: {len(data)}")
        
        # Process each sample
        annotated = []
        errors = 0
        
        for idx, sample in enumerate(tqdm(data, desc="Annotating")):
            # Extract question from prompt
            question = self._extract_question(sample.get('prompt', ''))
            answer = sample.get('completion', '')
            
            # Annotate
            annotation = self.annotate_single(question, answer)
            
            # Add to sample
            sample['divisions'] = annotation.get('divisions', ["UNKNOWN"])
            sample['primary_division'] = annotation.get('primary', "UNKNOWN")
            sample['division_reasoning'] = annotation.get('reasoning', '')
            
            if annotation.get('error'):
                errors += 1
            
            annotated.append(sample)
            
            # Save checkpoint every 100 samples
            if (idx + 1) % 100 == 0:
                self._save_checkpoint(annotated, output_file)
        
        # Final save
        self._save_checkpoint(annotated, output_file)
        
        print(f"\nCompleted!")
        print(f"Total annotated: {len(annotated)}")
        print(f"Errors: {errors}")
        print(f"Success rate: {(len(annotated) - errors) / len(annotated) * 100:.2f}%")
    
    def _extract_question(self, prompt: str) -> str:
        """Extract question text from prompt"""
        # Remove special tokens and extract user content
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
    parser = argparse.ArgumentParser(description="Annotate medical data with divisions using DeepSeek")
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file')
    parser.add_argument('--model', type=str, default='deepseek-ai/deepseek-llm-7b-chat',
                       help='DeepSeek model name')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device (cuda:0 for A6000)')
    
    args = parser.parse_args()
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(1)}")
    else:
        print("WARNING: No GPU available!")
    
    # Initialize annotator
    annotator = DeepSeekDivisionAnnotator(
        model_name=args.model,
        device=args.device
    )
    
    # Process dataset
    annotator.process_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
