# Manual Model Setup

MedGemma models require HuggingFace authentication and license acceptance.

## Prerequisites

### 1. HuggingFace Account

Create an account at: https://huggingface.co/join

### 2. Accept Model Licenses

Before downloading, you must accept the license agreements:

| Model | License Page |
|-------|--------------|
| medgemma-4b-it | https://huggingface.co/google/medgemma-4b-it |
| medgemma-27b-text-it | https://huggingface.co/google/medgemma-27b-text-it |

**Steps:**
1. Go to the model page
2. Click "Access repository" or "Agree and access repository"
3. Fill out the access form if required
4. Wait for approval (usually instant)

### 3. HuggingFace CLI Login

```bash
# Install huggingface-hub
pip install huggingface_hub

# Login (will prompt for token)
huggingface-cli login

# Or set token directly
huggingface-cli login --token YOUR_TOKEN
```

Get your token at: https://huggingface.co/settings/tokens

---

## Download Models

After authentication:

```bash
# Download medgemma-4b (recommended for testing)
python script/prepare/model/download_base_model.py --model medgemma-4b

# Download medgemma-27b (requires ~55GB disk space)
python script/prepare/model/download_base_model.py --model medgemma-27b

# Download all models
python script/prepare/model/download_base_model.py --model all
```

---

## Model Requirements

### medgemma-4b-it

| Requirement | Value |
|-------------|-------|
| Disk Space | ~8.5 GB |
| GPU Memory (inference) | ~8 GB |
| GPU Memory (training, 8-bit) | ~12 GB |
| Recommended GPU | RTX 3080 or higher |

### medgemma-27b-text-it

| Requirement | Value |
|-------------|-------|
| Disk Space | ~55 GB |
| GPU Memory (inference) | ~28 GB |
| GPU Memory (training, 8-bit) | ~38 GB |
| Recommended GPU | A100 40GB or higher |

---

## Troubleshooting

### Error: "401 Unauthorized"

- You haven't logged in: `huggingface-cli login`
- Token expired: Generate new token and re-login

### Error: "403 Forbidden"

- You haven't accepted the license
- Go to model page and accept

### Error: "Repository not found"

- Model name typo
- Check HuggingFace IDs:
  - `google/medgemma-4b-it`
  - `google/medgemma-27b-text-it`

### Error: "Disk space"

- Ensure sufficient disk space before download
- Use `df -h` to check available space

---

## Alternative: Manual Download

If automatic download fails, you can manually download:

1. Go to model page on HuggingFace
2. Click "Files and versions"
3. Download all files manually
4. Place in `model/raw/medgemma-4b/` or `model/raw/medgemma-27b/`

---

## Verification

After download, verify models:

```bash
# Check files exist
ls -la model/raw/medgemma-4b/

# Test loading
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('model/raw/medgemma-4b')
print(f'Tokenizer vocab size: {len(tokenizer)}')
"
```

---

## Directory Structure After Download

```
model/
├── raw/
│   ├── medgemma-4b/
│   │   ├── config.json
│   │   ├── model*.safetensors
│   │   ├── tokenizer.json
│   │   └── ...
│   └── medgemma-27b/
│       ├── config.json
│       ├── model*.safetensors
│       └── ...
└── tokenizer/
    ├── tokenizer.json
    ├── new_tokens.txt
    └── tokenizer_info.json
```
