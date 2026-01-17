MedGemma offers 4B multimodal (text+vision), 27B text-only, and 27B multimodal variants, all based on Gemma 3 architecture with medical pre-training on de-identified data like radiology images and clinical text. Quantized versions (e.g., 4-bit NF4 via QLoRA) enable efficient fine-tuning on consumer hardware for adding languages like Korean, preserving performance while cutting memory by up to 75%. Best results come from QLoRA on the 4B model first for rapid prototyping, then scaling to 27B variants with LoRA adapters.[1][2][3][4][5][6][7][8][9]

## Model Specs
MedGemma 4B handles text+images (SigLIP encoder), supports 128K context via GQA, and suits initial multilingual tests due to lower compute needs. The 27B text-only excels in pure medical text reasoning (e.g., MedQA 87.7%), while 27B multimodal adds image tasks but demands more VRAM. All are decoder-only transformers, fine-tunable via Hugging Face with PEFT (LoRA/QLoRA).[3][10][4][11][12][13][1]

## Quantization Options
Use 4-bit NormalFloat (NF4) quantization for QLoRA—superior to FP4 or INT4, enabling 27B on 40GB GPUs with minimal accuracy loss on biomedical tasks. For multilingual addition, quantize base models to 4/8-bit, freeze weights, and train LoRA adapters (rank 16-64) on language-specific medical data. Q-BLoRA extends this for better rank stability in low-resource setups.[5][14][15][16][7][8]

## Fine-Tuning Strategy
**Prep data**: Curate Korean medical corpora (PubMed/KoreaMed, K-BDS) paired with English prompts for bilingual alignment. Use SFTTrainer with custom prompts like "Translate and reason in Korean: [medical text/image]".[17][6]
**QLoRA workflow**: Load quantized model (e.g., `BitsAndBytesConfig(load_in_4bit=True)`), apply LoRA to attention/qkv layers, train 1-3 epochs on 8-24GB VRAM.[4][7][13]
**Multilingual tips**: Start with 4B (fits single A100), validate on KorMedMCQA/KMLE, merge adapters post-training.[18][19][20]

## Best Combinations
| MedGemma Variant | Quant Level | Method | Hardware Fit | Use Case | Notes |
|------------------|-------------|--------|--------------|----------|-------|
| 4B Multimodal   | 4-bit NF4  | QLoRA | 16-24GB VRAM | Add Korean to text+image | Fastest; strong VQA baseline [6][7] |
| 27B Text-Only   | 4-bit NF4  | QLoRA/LoRA | 40-80GB VRAM | Pure text multilingual | Best MedQA perf; text focus [5][12] |
| 27B Multimodal  | 8-bit      | LoRA   | 80GB+       | Image+new lang reasoning | Highest capacity; slower [3][21] |

Qwen2 variants pair well as SLM distillers for MedGemma (strong multilingual base), fine-tuned via QLoRA for Korean medical tasks. Test on your NVMe/FPGA setup for inference speed.[20][22][5]

[1](https://huggingface.co/google/medgemma-4b-pt)
[2](https://developers.google.com/health-ai-developer-foundations/medgemma)
[3](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
[4](https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora)
[5](https://arxiv.org/html/2509.04534v1)
[6](https://www.datacamp.com/tutorial/fine-tuning-medgemma)
[7](https://learnopencv.com/fine-tuning-gemma-3/)
[8](https://codecompass00.substack.com/p/qlora-visual-guide-finetune-quantized-llms-peft)
[9](https://huggingface.co/google/medgemma-4b-it)
[10](https://blog.cordatus.ai/featured-articles/medgemma-googles-revolutionary-ai-model-for-medical-text-and-image-analysis/)
[11](https://learnopencv.com/medgemma-explained/)
[12](https://arxiv.org/html/2507.05201v2)
[13](https://keras.io/examples/keras_recipes/parameter_efficient_finetuning_of_gemma_with_lora_and_qlora/)
[14](https://github.com/pprp/Awesome-LLM-Quantization)
[15](https://arxiv.org/abs/2305.14314)
[16](https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.23/132118/Accurate-and-Efficient-Fine-Tuning-of-Quantized)
[17](https://www.perplexity.ai/search/ec64387a-8ee9-4aaa-b413-cb45c5480fad)
[18](https://www.perplexity.ai/search/b0044446-5d22-4d1d-b769-886f729e03a2)
[19](https://arxiv.org/html/2509.04471v2)
[20](https://qwenlm.github.io/blog/qwen2/)
[21](https://www.reddit.com/r/LocalLLaMA/comments/1lvqtxa/multimodal_medgemma_27b/)
[22](https://forum.effectivealtruism.org/posts/H4ecMXSzQuGJxw4wE/impact-of-quantization-on-small-language-models-slms-for-1)
[23](https://llm-stats.com/models/medgemma-4b-it)
[24](https://github.com/Google-Health/medgemma)
[25](https://exceltic.com/en/generative-ia-with-full-control-small-slms-language-models-and-quantisation-2/)
[26](https://arxiv.org/abs/2507.05201)
[27](https://skywork.ai/blog/models/google-medgemma-27b-text-it-free-chat-online-skywork-ai/)
[28](https://ollama.com/alibayram/medgemma:27b)
[29](https://cloud.google.com/blog/topics/developers-practitioners/a-step-by-step-guide-to-fine-tuning-medgemma-for-breast-tumor-classification)
[30](https://arxiv.org/html/2404.14779v1)
[31](https://www.sciencedirect.com/science/article/pii/S1110866525002294)
[32](https://huggingface.co/blog/jjokah/small-language-model)
[33](https://arxiv.org/html/2504.17119v1)
[34](https://www.emergentmind.com/topics/gemma-3-4b)
[35](https://github.com/filipa131/LLM-fine-tuning)
[36](https://proudlynerd.vidiemme.it/qlora-and-gemma-2b-efficient-4-bit-llm-training-on-resource-constrained-gpus-2f57dfe5c92b)
[37](https://www.sciencedirect.com/science/article/abs/pii/S1568494625003953)
[38](https://arxiv.org/html/2410.16088v1)