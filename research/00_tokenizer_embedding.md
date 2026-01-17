# New tokenizer and embedding

## New token needed
| Korean-first / near-native fluency                     | **20k – 30k**     |

### Add dedicated tokens
29.9k = 30k - dedicated tokens
sample code
```python

```

### Take important sentences tokens (medical)
about 10k

```python
```

### Take general sentence tokens
about 20k

```python
```

### make tokenizer

```python
```

## train embedding
train embedding with new token dimension

```python
```

## Practical recommendation for “+30k tokens” Korean adaptation for lora

* **MedGemma 4B**: LoRA **r=16** is a strong default (≈9.2M params) plus +30k embedding rows (≈76.8M params).

```python
```

* **MedGemma 27B**: LoRA **r=8 or r=16** depending on data size; r=16 is ≈34.8M params, while +30k embedding rows cost ≈161.3M params.

```python
```
