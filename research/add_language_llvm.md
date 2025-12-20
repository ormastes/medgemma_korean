Adding New Language Support to a Monolingual LLM (e.g. English-only Med-GEMMA)

Large Language Models that are trained on a single language can be extended to understand and generate another language through various techniques. Below, we explore several methods – from simple translation pipelines to advanced retraining strategies – and provide guidance on how to implement each. We focus on approaches applicable to LLMs in general (with examples in the biomedical domain like Med-GEMMA) and prioritize open-source research and implementations.

1. Front-End Translation/Interpreter Layer

One straightforward approach is to add an external translation layer in front of and behind the model. In this pipeline, user input in the new language is translated into the model’s original language (English), the monolingual LLM processes it, and then the output is translated back to the target language. This effectively uses the LLM as-is, leveraging a machine translation system as an “interpreter.”

Process: For each query, detect the language and, if it’s not English, translate it to English. Feed the translated prompt into the LLM to get an English answer, then translate that answer into the target language (e.g. Korean) for the user. This can be done using online APIs or open-source translators (e.g. HuggingFace’s MarianMT or M2M100 models for many languages).

Advantages: No need to modify or retrain the LLM itself. You can utilize high-quality machine translation services. This preserves the LLM’s original capabilities and reliability in its native language.

Disadvantages: The overall performance depends on translation accuracy – errors in translation can propagate. It also introduces additional latency and complexity (two extra steps) at inference time. The model’s output might lose nuance if the translation isn’t perfect. There’s also an assumption that content can be translated accurately in both directions, which may be challenging for domain-specific jargon (e.g. biomedical terminology).

Implementation Tips: Use a language detection tool to route the pipeline. For open-source implementation, you could employ models like Helsinki-NLP/Opus-MT for specific language pairs or Google’s translation API for a robust solution. For example, one could wrap a HuggingFace MarianMT model: translator = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ko-en') (and the reverse en-ko model). Before querying Med-GEMMA, translate Korean input to English, and after getting the answer, translate it back to Korean. This approach essentially *“transforms a multilingual scenario into a monolingual one via an extra translation step”*, making it possible to use an English-only model for Korean queries.


2. Expanding the Tokenizer and Vocabulary, then Retraining

If we want the LLM itself to understand and generate the new language natively, a crucial step is to expand its tokenizer vocabulary to include the new language’s text. Monolingual models often have subword vocabularies optimized for the original language; for example, an English tokenizer might poorly encode Korean text into lots of fragmented or unknown tokens. By adding new tokens and retraining, we enable the model to represent the new language efficiently.

 Workflow for adding a new language to an LLM by extending its tokenizer and continued pre-training. Steps include (1) gathering target-language text, (2) training a new tokenizer, (3) merging it with the original tokenizer, (4) resizing the model’s embedding layers, and (5) continued pre-training on the new language.

Key steps to implement this approach:

1. Collect Target Language Text: Assemble a large corpus of text in the target language (e.g. Korean Wikipedia, news, or medical texts for a biomedical model). This will be used for tokenizer training and for continued pretraining. Ideally, ensure hundreds of millions of tokens for the new language if possible, but there have been successful experiments with smaller amounts (even ~16K tokens in low-resource settings – though more data yields better language proficiency).


2. Train a New Tokenizer for the New Language: Use the target-language corpus to train a subword tokenizer (BPE or SentencePiece). You can do this from scratch or by building on the existing tokenizer:

From Scratch (Multilingual): Train a tokenizer on a combined corpus of the original language + target language. This gives a single vocabulary covering both. However, training from scratch means you’ll lose the original token-ID mapping (making it hard to reuse the model’s learned embeddings without significant adjustments).

Augment Existing (Preferred): Train a monolingual tokenizer on the new language and then merge it with the original. For example, NVIDIA’s NeMo tutorial suggests training a GPT-2 style tokenizer on Thai text alone, then combining it with the English tokenizer. In code, you might load the original tokenizer and use train_new_from_iterator to learn new tokens from the Korean corpus, e.g.:


from transformers import AutoTokenizer
base_tok = AutoTokenizer.from_pretrained("med-gemma-base")  
new_tok = base_tok.train_new_from_iterator(korean_corpus_iter, vocab_size=8000)  
new_tok.save_pretrained("./tokenizer_ko/")

This learns ~8k new tokens capturing Korean text (the exact number can be tuned). Merging involves appending these new tokens to the original vocabulary file and updating the merges/rules. The goal is to preserve all original tokens and simply add new ones for Korean. This way, the model can still handle English exactly as before. Tools like 🦜 Hugging Face Tokenizers or SentencePiece can facilitate controlled augmentation of the vocabulary.


3. Resize and Initialize Model Embeddings: After extending the vocabulary, the model’s embedding matrix (and output layer if tied) must be resized to the new vocabulary size. Most ML frameworks allow this (e.g. model.resize_token_embeddings(len(new_tokenizer)) in Hugging Face Transformers). The newly added rows (for the new tokens) need to be initialized. Research shows that initialization can significantly impact learning efficiency. Good strategies include:

Initializing new token embeddings to small random values or zeros (the model will learn them during training, though this causes a temporary performance drop).

Mean-of-Subwords Initialization: If the new tokens can be broken into known subword pieces, initialize the embedding as the average of those pieces’ embeddings. For example, if you add a new token “독감” and the English model’s tokenizer would have parsed it as “독” + “감” (or some byte sequence), average those embeddings to initialize “독감”. This simple method was found to work as well as more complex schemes in a recent study.

Using external multilingual embeddings or dictionaries (discussed more in Section 5). For instance, map “hospital(병원)” to the existing embedding for “hospital” if a dictionary says they are equivalents.


After initialization, replace the model’s embedding layer with the new expanded embedding. In PyTorch, this might involve creating a new nn.Embedding with the bigger vocab size, copying over the old weights for unchanged tokens, and assigning the initialized vectors for new tokens. NVIDIA’s NeMo guide outlines this process: create a new weight matrix, copy original weights, fill new indices with zeros, and load it into the model state dict.

 Extending the embedding layer to accommodate new vocabulary. The original embedding matrix weights are copied, and new token slots are added (initialized to zero or an informed guess). These new embeddings (for Korean tokens in this example) will be learned during continued training.


4. Continued Pretraining on the New Language: With the tokenizer and model ready, fine-tune the LLM on the target language corpus (unlabeled text) to teach it the new language. This is essentially further language-model pretraining. All model weights can be updated here (or you may freeze some layers to retain more of the original knowledge – though typically, full-model continued pretraining is used to properly integrate the new language). Use a moderate learning rate and train for enough tokens so the model converges on the new language. For example, Tejaswi et al. (2024) continued training English LLMs on ~200M tokens of a new language to achieve strong performance. During this phase, the model learns the patterns, syntax, and terminology of the new language via the new token embeddings.

Monitoring Performance: It’s wise to monitor not just perplexity on the new language, but also ensure the model doesn’t degrade on English. This method can cause some initial drop in performance on original tasks due to the new embeddings being untrained. However, studies report that with sufficient continued training, the model recovers or even improves on those tasks. It appears the model can learn a new language without catastrophic forgetting of the original, especially if the original language data or multi-task mix is occasionally seen or the training is done carefully.

Efficiency Consideration: If one skips adding new tokens and instead forces the model to learn the new language using only its original vocabulary, it can still learn (e.g., training on millions of Korean words split into bytes/char tokens). Zhao et al. (2024) showed that an English LLM fine-tuned on enough target-language text (even without adding vocabulary) eventually matched the performance of a model that was trained from scratch in that language. The trade-off is efficiency: without proper tokens, the Korean text might be three to four times longer in tokens, making training and inference slower. Thus, vocabulary extension is recommended for efficient generation unless the new language shares an alphabet or lots of vocabulary with the original.




Open-Source Example: NVIDIA’s NeMo provides an end-to-end recipe for this workflow. In their example, they start with an English GPT-1.3B model and add Thai language support. They train a Thai tokenizer, merge it, extend the model, and continue pretraining on Thai Wikipedia. The result is a bilingual Thai-English model. This approach can be directly applied to adding Korean to Med-GEMMA: gather Korean medical texts, augment the tokenizer with Korean, resize embeddings, and train on that text. The same strategy was also used in research like “Exploring Language-Specific LLM Adaptation” (Tejaswi et al., EMNLP 2024), which systematically studied adding various languages to base LLMs. They found that adding ~10k new tokens for a new language and training on ~200M tokens yields strong performance, and that smart initialization of new embeddings (even just averaging subwords) speeds up adaptation.

3. Using Adapters or LoRA for Language-Specific Fine-Tuning

Instead of modifying the entire model or vocabulary, one can attach additional parameters (adapters) to the model and train those on the new language. This falls under Parameter-Efficient Fine-Tuning (PEFT) techniques, such as Adapters and LoRA (Low-Rank Adaptation). These methods let the base model weights largely frozen, while small new modules are learned to encode the new language knowledge.

Adapters: An adapter is a small neural network (often a bottleneck MLP) inserted at each transformer layer (for example, after the attention or feed-forward sub-layer). For multilingual support, you can give each language its own adapter modules. During training on Korean text, only the Korean adapter weights are updated; the original model stays mostly fixed. At inference, to use the model in Korean, you activate the Korean adapter. Research has shown this approach can “lift the curse of multilinguality” by allocating dedicated capacity to the new language without interfering with existing ones. For instance, the MAD-X framework (Pfeiffer et al. 2020) trained language adapters on unlabeled text and was able to add new languages to BERT with minimal impact on the original languages. Later, X-MOD (2022) went further by pre-training a transformer with some shared weights and some language-specific adapter modules; adding a language post-hoc meant training new adapters and embeddings for that language and plugging them in. In all cases, adapters provide modularity: you can keep a library of “Korean adapter,” “French adapter,” etc. and load them as needed on top of the base model.

LoRA: LoRA is a specific PEFT method that injects trainable low-rank matrices into existing weight matrices (e.g. each self-attention projection) instead of adding new layers. During fine-tuning, only those low-rank matrices (and an associated scaling factor) are learned, while the original weights remain frozen. To teach an English LLM Korean via LoRA, one would attach LoRA matrices to key weight tensors and fine-tune on Korean text or a translated instruction dataset. After training, the LoRA matrices (which are very small, typically a tiny percentage of the model’s size) can be applied to the model to infuse the new language capability. For example, a 2024 study applied LoRA to an English Gemma-2B model using a Marathi translated instruction corpus, and found the model produced much more fluent Marathi after fine-tuning. The base model’s weights were untouched, so its English competency and general knowledge remained, and only Low-Rank “deltas” were learned to capture Marathi specifics. (They did note a slight drop in some reasoning benchmarks post-LoRA, indicating some trade-off or evaluation artifact, but generally the model’s multilingual utility increased.)


How to apply Adapters/LoRA in practice:

Adapters Implementation: Using libraries like AdapterFusion/AdapterHub, you can add a new language adapter to your model. For a HuggingFace Transformer model, it’s often as simple as: model.add_adapter("korean", config=AdapterConfig(...)), then model.train_adapter("korean") on your Korean text data. Only the adapter weights (and optionally embeddings for new tokens if you add them) will be trained. After training, save the adapter. In use, activate it via model.set_active_adapters("korean"). The rest of the model remains as it was, so other language capabilities are preserved. This mitigates catastrophic forgetting by design.

LoRA Implementation: The popular PEFT library by Hugging Face makes it easy to apply LoRA to large models. You choose target modules (e.g., all W_q, W_k, W_v matrices in self-attention, or even all dense layers) and a rank (like rank=8 or 16). The library then wraps the model such that these weights get LoRA tweaks. For example:

from peft import LoraConfig, get_peft_model
lora_cfg = LoraConfig(target_modules=["q_proj","v_proj"], r=16, lora_alpha=32)
model = AutoModelForCausalLM.from_pretrained("med-gemma-base", device_map="auto")
model = get_peft_model(model, lora_cfg)

Then train model on the new language data (could be done similar to normal fine-tuning, using a language modeling objective or instruction tuning objective). Only the small LoRA matrices get updated. After training, you’ll save the LoRA adapter weights (often just a few MB). At inference, load the base model and merge or attach the LoRA weights (PEFT can do model.merge_and_unload() to bake them in, or keep them separate). The model will now respond in Korean when prompted appropriately. Importantly, by keeping the English weights intact, this approach “adds” Korean knowledge without overwriting English knowledge.

Mixing with Other Methods: Adapters and LoRA can be used in combination with the tokenizer expansion from section 2. For instance, you might expand the vocabulary to include Korean tokens (for efficiency) but still use LoRA to train the model on Korean data rather than full fine-tuning. This would involve resizing embeddings as before, then using LoRA for the remaining weights. There are also advanced adapter techniques like Adapter Fusion (combining multiple adapters) and FLARE (2024) which cleverly mixes source and target language representations within adapter layers. FLARE showed that by fusing English and target language representations inside a LoRA-like adapter, one can get improved cross-lingual understanding without adding any new parameters – essentially using the English latent knowledge to guide the new language. Such research highlights that the base model’s knowledge of English can assist learning of the new language, and adapters/LoRA provide a flexible framework to achieve that.


Summary: PEFT methods are very practical for extending LLMs. They are lightweight (training uses far less memory and the resulting adapters are small files) and they don’t require retraining the whole model from scratch. Open-source examples include Hugging Face’s LoRA fine-tuning scripts (often used for alpaca-style fine-tunes on new data) and AdapterHub’s many pretrained adapters. In a production setting, you could maintain one base model (e.g. Med-GEMMA English) and have separate LoRA weights or adapter modules for Korean, Chinese, etc., loading them on demand – this avoids deploying multiple full models. Moreover, because the original model isn’t fully altered, the risk of forgetting or degrading original language performance is minimized. This makes adapters/LoRA a popular choice for adding language support in a controlled, efficient way.

4. Multilingual Continued Pretraining on New Language Data

Another approach is to continue pretraining the model on a multilingual or target-language corpus without architectural changes. This is essentially language transfer through fine-tuning: you take your monolingual LLM and just feed it lots of new language text, possibly mixed with original language text, so that it becomes bilingual or multilingual. This method was historically used in works like GPT-2 to create a Spanish GPT-2 by further training the English GPT-2 on Spanish text, for example.

How to do it:

Prepare a Training Corpus: Gather a large corpus in the new language (and optionally combine it with the original language data). In the biomedical context, for adding Korean, you might scrape or use Korean medical literature, health news articles, or translation of English medical datasets. If you want a truly bilingual model, you can mix, say, 80% Korean text with 20% English text during training to ensure it retains English. If you prefer, you can also include parallel text (bitext) – pairs of English and Korean sentences – to explicitly teach translations or alignments.

Training Process: Simply continue the language modeling training of the LLM on this new data. Use the same training objective (next-word prediction or masked language modeling) as was originally used. Because the model’s tokenizer is fixed in this scenario, the new language text will be tokenized using the existing vocabulary. If the script is very different (e.g. Korean Hangul), the tokenizer might break words into character bytes or unknown tokens. The model will slowly adjust and learn the new language patterns, but it might require more training steps to compensate for the less-than-ideal tokenization (each Korean word could be many pieces). As noted, millions of target-language tokens can be sufficient to teach the model, but it’s less efficient than if it had native tokens.

Monitoring and Mitigation: Keep an eye on the model’s performance in English as well. Purely training on new language data can cause some forgetting of the original language (since the model weights shift to accommodate the new data distribution). To mitigate this, you can occasionally intermix some English data (so the model rehearses English) or use a smaller learning rate so as not to overwrite existing knowledge too quickly. Another trick from research is transliteration – e.g., Husain et al. (2024) converted Indic scripts to Latin script before feeding to an English LLM, effectively making Hindi look like pseudo-English characters. This allowed the model to leverage its familiarity with the Latin alphabet, albeit at the cost of not truly learning native script. For Korean, one could similarly romanize the text (e.g. "감기" -> "gamgi") and train on that, which an English model can ingest more naturally. However, this might be undesirable if you want the model to output native script; it’s an intermediate workaround when one cannot alter the tokenizer.

Use of Parallel Data (Bitext): Incorporating parallel corpora (the same sentences in English and Korean) during training can create cross-lingual links in the model’s representation. For instance, you can occasionally train the model with a Translation Language Modeling (TLM) setup: concatenate an English sentence and its Korean translation, and have the model predict masked words or the next sentence. Facebook’s XLM showed that such training helps the model align the languages in its latent space. Even without a special objective, simply having the model see translations of the same content in both languages will encourage it to associate their meanings. Recent work (Ji et al. 2024) cautioned that too much focus on translation can sometimes make the model’s internal representation language-specific (good at translation but less good at general cross-lingual tasks). So, a balanced approach is best: use bitext to seed alignment, but also train on plain monolingual text so the model learns to use the language fluently, not just translate. In practice, one strategy is a curriculum: first do a phase of bitext-based training (to quickly teach mappings between English and Korean words/phrases), then a phase of pure Korean language modeling to solidify fluency. This way you get the benefit of translation alignment without overly “hard-coding” the translation task at the expense of other abilities.


Outcome: After sufficient continued pretraining, the model becomes multilingual. For example, Google’s Med-PaLM (medical GPT model) was originally English, but by further pretraining on non-English medical texts, one can obtain a version that handles, say, Spanish or Korean patient questions. An open-source instance is the “Swallow” project (2024) which took LLaMA-2 (mostly English) and continued training it on a large Japanese corpus plus some Japanese-English parallel data, producing a model with drastically improved Japanese capabilities. This shows that just by feeding a lot of target language data (and a bit of parallel data for alignment), an English LLM can be transformed into a competent bilingual model. The cost is computational: you are effectively doing additional pretraining, which for very large models can be expensive (though still far cheaper than training from scratch on that language, thanks to transfer learning).

Note: This approach and the vocabulary-expansion approach (Section 2) overlap – often, when adding a language properly, you will both extend the vocab and do continued pretraining. However, it’s possible to do continued training without vocab changes (just with the original tokenizer), as discussed. The main takeaway is that the model’s knowledge is updated through further training on new data. Open-source LLM communities (like those around LLaMA) have employed this by taking an existing model checkpoint and fine-tuning on new language text – essentially treating it as unsupervised fine-tuning. If using Hugging Face, one would load the model and call Trainer.train() on the new corpus (formatted appropriately for causal or masked LM), similar to domain adaptation training (just now it’s language adaptation). Ensure to save this new checkpoint, as it represents a multilingual version of your model.

5. Embedding Alignment and Multilingual Embedding Projection

A more surgical set of methods focuses on projecting the new language into the model’s existing embedding space. The intuition is that we can teach the model what foreign words mean by aligning them with known words in the original language, without extensive model-wide training. These techniques often use bilingual dictionaries or cross-lingual word embeddings to initialize or adjust the model’s embeddings for the new language.

Bilingual Lexicon Mapping (WECHSEL approach): WECHSEL (Minixhofer et al., 2022) is a method originally designed to transfer an English-only model to a new language by replacing its vocabulary. The steps are:

1. Swap Tokenizer: Remove the English tokenizer and use a tokenizer of the target language (e.g., a Korean WordPiece tokenizer with a vocab of size N). In WECHSEL, the model becomes monolingual in the new language (English tokens are dropped), but one could adapt this to adding tokens rather than replacing.


2. Use Multilingual Word Embeddings: Obtain static word embeddings for both English and Korean (e.g., FastText vectors, which are available for many languages and are often aligned). For each token in the new Korean vocabulary, find an English word of equivalent meaning (via a bilingual dictionary or translation). For example, align Korean “의사”(doctor) with English “doctor,” “병원”(hospital) with “hospital,” etc.


3. Initialize New Embeddings: Initialize the model’s embedding for each Korean token to the vector of its English counterpart. If exact translations aren’t in the vocab, one can compose embeddings: WECHSEL proposed using the average of multiple subword embeddings to approximate the word. In practice, one might tokenize the Korean word using the English tokenizer (getting pieces) and then average the English embedding of those pieces. Or use the FastText bilingual embeddings directly if they cover subwords.


4. Fine-Tune the Model: After this initialization, perform a round of training on the target language text to adjust the model. Because the embeddings started in the “right place” (aligned to English semantic space), the model can quickly adjust to using them in context. WECHSEL demonstrated this by transferring GPT-2 and RoBERTa to German, French, Chinese, etc., achieving performance close to a native model with far less compute.



For adding a language (rather than replacing), one could partially apply WECHSEL: keep the English embeddings as-is, append new Korean embeddings initialized via this cross-lingual method. Essentially this is a sophisticated version of embedding initialization for new tokens (from Section 2) that uses bilingual semantic knowledge instead of simple averages.

Linear Mapping of Embedding Spaces: Another approach is to learn a linear projection that maps foreign word embeddings into the English embedding space. This is rooted in classic cross-lingual word embedding alignment (Mikolov et al. showed you can learn a linear map between vector spaces of two languages given a dictionary of matching words). To do this for an LLM:

Train or obtain a set of word embeddings for Korean (could be by running Word2Vec on Korean corpus or using pre-trained ones).

Take the LLM’s embedding matrix for English (or just the subset for common words).

Create a bilingual lexicon of word pairs. For each pair (e.g., “virus” – “바이러스”), get the English embedding E(virus) from the LLM and the Korean embedding K(바이러스) from the Korean embeddings.

Learn a linear transformation W such that W * K(x) ≈ E(x) for all word pairs x in the dictionary. This can be done with a simple least-squares solution.

Apply this transform W to all Korean embeddings to map them into the English space. Then use these transformed vectors to initialize the new token embeddings in the LLM.

This way, “바이러스” in the model’s embedding will be placed near “virus” in vector space, etc. The LLM will then treat Korean words in a manner similar to their English equivalents, to the extent the rest of the network relies on semantic closeness.


Past work did exactly this for aligning embeddings post-hoc. It’s a quick way to get a multilingual embedding matrix. However, without further fine-tuning, this doesn’t guarantee the deeper layers of the model fully understand the new tokens – so usually you’d still do some continued training on bilingual data to propagate this alignment through the network.

Progressive Embedding Expansion (CLP): Ostendorff & Rehm (2023) proposed a Cross-Lingual Progressive Transfer method. They start with an English model and gradually introduce German (in their case) by:

Identifying overlapping tokens between languages (e.g., numbers, names, or in biomedical context Latin terms that appear in both).

Copying those token embeddings from English to the new model (so the new model’s “Aspirin” token starts with the same vector as English “Aspirin”).

For new tokens without overlap, initialize somehow (random or based on simpler heuristics).

Then progressively train the model on the target language. “Progressive” might involve staging the training or growing the vocabulary in increments. The result was an efficiently trained model for the new language, reusing a lot of English knowledge directly.

In essence, this is another form of embedding reuse and careful initialization to shorten training time.



Practical Tutorial for Embedding Projection: Suppose we want to quickly add Korean medical terms to an English biomedical LLM:

1. Get Bilingual Terminology: For a medical domain, gather a list of important terms in English and Korean. For example, “fever – 발열”, “diabetes – 당뇨병”, etc. Resources like translated medical dictionaries or UMLS mappings can help.


2. Map to Model Tokens: Tokenize the English term with your model’s tokenizer to get its embedding (for a single-word term, it might be one token; for multi-word or out-of-vocab, use an average of subword embeddings). For Korean term, if you have extended the tokenizer to include it as one token, great – if not, you might add it as a new token for this purpose.


3. Initialize Embedding: Set the new token’s embedding vector equal (or close) to the English vector. If using FastText or another source, you could also directly find the Korean word’s vector in a common embedding space.


4. Repeat for many terms, perhaps all new vocab tokens that have a known translation. For tokens that have no clear counterpart, you might just do a naive initialization (small random).


5. Fine-tune on a small mixed corpus: For example, create a synthetic parallel corpus of short phrases or sentences with those terms (or use actual parallel data if available) and fine-tune the model on it for a few epochs. Even a brief tuning can adjust the model’s weights so that it properly associates the Korean tokens with the same contexts as the English ones.



This process is somewhat manual, but it leverages prior knowledge. An open-source implementation is provided by the authors of FOCUS (Dobler & de Melo, 2023), which introduced a way to initialize new token embeddings as combinations of existing tokens based on semantic similarity. They even released code (e.g., a GitHub repo “konstantinjdobler/focus”) to automate embedding initialization for new vocabulary using overlapping tokens and lexicons. Their findings, as well as others, indicate that informed initialization (via dictionary or overlap) significantly speeds up convergence when adding a language, compared to random init.

When to use these methods: Embedding projection techniques are especially useful if you have limited target-language data to train on. By starting the model off with a good guess of what each new word means (in terms of the model’s existing knowledge), you reduce the amount of training needed. For a domain-specific model like Med-GEMMA, this could be very handy: you might not have billions of Korean medical sentences to train on, but you do have medical terminology translations. By aligning those, the model can immediately leverage its English medical knowledge for Korean inputs. After alignment, even a smaller corpus of Korean text might suffice to finetune and fill in the gaps. This approach complements the full pretraining in method 4 – you could align embeddings first, then do a short continued pretraining. The EMNLP 2024 study in fact compared various initialization schemes and concluded that a simple mean-of-subwords approach was as effective as more complex ones in their experiments. That suggests even basic alignment steps can be worthwhile and don’t require extremely sophisticated algorithms.

6. Prompt Tuning and In-Context Techniques

Beyond changing model weights or embeddings, one can coax a monolingual model to handle other languages using prompt-based methods. These techniques leverage the model’s existing knowledge, sometimes treating the other language as an unknown task to solve via clever prompting or minor adjustments.

Soft Prompt Tuning for Languages: Prompt tuning involves learning a set of continuous prompt vectors that guide the model, without altering the model’s main parameters. In a multilingual context, you can train a prompt that “activates” the model’s ability to work in the target language. For instance, even an English-only LLM might have latent capacity to imitate other languages (especially if it saw a few foreign words during training). A soft prompt can amplify this. The process:

1. Create N learnable prompt tokens (embeddings) that will be prepended to every input.


2. Choose a training objective – for example, feed the model many Korean sentences and train the prompt embeddings such that the model reconstructs those sentences (self-supervised). Essentially, you treat the prompt as the only thing you’re adjusting to minimize perplexity on Korean text.


3. Alternatively, use a translation task: provide an English prompt and a Korean output, or vice versa, and tune the prompt to encourage that mapping.


4. After training (which typically requires far fewer steps than full model fine-tuning), you save these prompt embeddings. Now, to get the model to work in Korean, you attach the prompt to the input.



This method is appealing because the prompt vectors are tiny compared to the model and can be learned quickly. It’s like teaching the model “when you see this special learned trigger, shift into Korean mode.” In practice, prompt tuning has seen more use in task-specific settings, but the idea extends to languages. One challenge is that if the base model truly has never seen the language, a prompt alone might not conjure fluency out of thin air. It works better if the model has some latent knowledge (for example, GPT-3 was mostly English but did see some non-English data, so a prompt could help it surface that). For an English-only model with zero Korean exposure, you might first need a bit of training data (which essentially becomes a few-shot learning scenario rather than pure prompt tuning).

In-Context Learning and Examples: This is a zero-code solution: use prompt examples or structure to help the model with another language. A powerful trick is cross-lingual few-shot prompting. Suppose you want an English-only model to answer a question in Korean. You can provide an in-context demonstration like:

Example 1:
Q (English): "What are the symptoms of flu?"
A (English): "Common symptoms include fever, cough, and body aches."

Example 2:
Q (Korean): "독감의 증상은 무엇인가요?"
A (Korean):

Even though the model might not be fluent in Korean, seeing the pattern that “Q in one language -> A in that language” and leveraging its knowledge of flu symptoms (from English) can allow it to fill in a plausible Korean answer. In tests with advanced models like GPT-4, it was found that providing few-shot examples in a high-resource language can improve performance on a low-resource language query. Essentially, the model uses English exemplars to figure out what the question is asking, then attempts to respond in the target language context.

Another in-context strategy is to explicitly include a translation step in the prompt. For example:

You are a multilingual medical assistant. 
Patient asks in Korean: "열이 있고 기침이 나요. 무슨 병일까요?"
(Translation: The patient says they have a fever and cough. What illness could it be?)
Answer in Korean: 가능성 있는 질병은 감기 또는 독감입니다.

By writing the prompt to include a parenthetical English translation of the user’s Korean query, you help the model understand it (since the model knows English), and then you ask it to answer in Korean. This approach is a manual “interpreter layer” done within the prompt itself, guiding an English model to function in another language. It’s particularly useful if you cannot alter the model weights and must rely on prompting only.

When to use prompt-based methods: These are most useful if you want to avoid training entirely or cannot (for instance, using a closed API model). For open-source models, prompt techniques can serve as an immediate band-aid while longer-term solutions (like fine-tuning) are in progress. They can also be combined: after doing all the heavy lifting of adding Korean via any of methods 2-5, you might still use a cleverly crafted prompt to ensure the model responds in Korean (for example, prefix every user query with “Translate the following question to English, answer it, then translate the answer back to Korean:” – though this is just a specific prompting approach basically implementing method 1 internally).


In summary, prompt and in-context methods leverage the model’s existing capabilities. They don’t truly expand the language proficiency of the model (unlike training-based methods), but they can yield surprisingly good results with no model updates. As an analogy: you’re not teaching the model new words, you’re teaching it tricks to handle or bypass the language barrier (like using translations or examples). This might be sufficient for some applications and can always serve as a baseline to compare against more involved methods.


---

Conclusion: Adding support for an additional language to a monolingual LLM can be achieved in numerous ways, and often the best solution is a combination of techniques. For a model like Med-GEMMA (English-only) looking to support Korean, one might expand the tokenizer, do continued pretraining on Korean medical texts, and perhaps use LoRA to fine-tune on a translated instruction dataset for conversational ability – covering both the base language understanding and instruction-following in Korean. Throughout the process, it’s crucial to use open-source tools and data: e.g. tokenizers from Hugging Face, corpora from OSCAR or Wikipedia, bilingual dictionaries (like CCAligned or UMLS translations), and adaptation frameworks like PEFT or AdapterHub. By following the research-backed strategies outlined above – from translation pipelines to vocabulary augmentation, from adapter modules to embedding alignment – one can systematically extend an LLM’s linguistic reach without starting from scratch. Each method has trade-offs in complexity, required data, and performance, so the choice may depend on resources available and the importance of maintaining original model performance. Fortunately, the community has demonstrated success in all these approaches, enabling practitioners to pick a suitable path to multilingual AI.

Sources:

Translation pipeline and multilingual retrieval examples

NVIDIA NeMo tutorial on adding a new language (Thai) to GPT-3 models

EMNLP 2024 Findings: language-specific LLM adaptation (tokenizer expansion, training tips)

Rohan Paul (2025) overview of cross-lingual transfer (bitext, adapters, LoRA, prompts)

Khade et al. (2024) – LoRA adaptation on Gemma for Marathi (low-resource)

Minixhofer et al. (2022) – WECHSEL embedding initialization for cross-lingual LM transfer

Ostendorff & Rehm (2023) – Cross-lingual Progressive Transfer (CLP) for efficient language adaptation

Microsoft Medium blog (2023) – Multilingual RAG and translation vs. multilingual model considerations
