Below is an operational training recipe that matches your plan (1.1 raw CPT → 1.2 staged prompt training), including **how to validate**, **when to stop**, and **how to verify “QA loss ~0 but raw Korean generalization is still good.”** I’ll give concrete metrics, stopping rules, and PyTorch-style pseudo code.

---

# 1.1 Train raw Korean texts (CPT)

## Objective

Next-token prediction on packed sequences.

* Inputs: `input_ids` `[B,T]`
* Labels: `labels` `[B,T]` = shifted inputs
* Loss: cross entropy over vocab

### Training

* Pack many documents into a continuous stream
* Insert EOS between docs

## Validation: what to measure

Use a **held-out raw Korean validation set** (not in training). Compute:

1. **Validation loss**
2. **Validation perplexity**: `ppl = exp(loss)`
3. Optional: **tokens/sec**, to detect data pipeline issues

### When to stop (practical)

You stop when **val loss stops improving**:

* Keep best checkpoint by `val_loss`
* Early stop if no improvement for `patience` evals
  Example: evaluate every N steps, patience 5–10 evaluations.

**Rule of thumb**

* If `val_loss` improves < **0.01** (or < 0.2%) for several evals, you’re near convergence for that dataset distribution.

---

## Pseudo code: CPT training + validation + early stop

```python
import math, torch
import torch.nn.functional as F

@torch.no_grad()
def eval_causal_lm(model, val_loader, device="cuda", max_batches=200):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches: break
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

        out = model(x)
        logits = out.logits if hasattr(out, "logits") else out  # [B,T,V]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="mean")

        # token count (no masking assumed)
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

def train_cpt(model, train_loader, val_loader, opt, device="cuda",
              eval_every=500, max_steps=50_000, patience=8):
    best_val = float("inf")
    bad_evals = 0

    model.train()
    for step, batch in enumerate(train_loader, start=1):
        if step > max_steps: break
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

        out = model(x)
        logits = out.logits if hasattr(out, "logits") else out
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % eval_every == 0:
            val_loss, val_ppl = eval_causal_lm(model, val_loader, device=device)
            print(f"[CPT] step={step} train_loss={loss.item():.3f} val_loss={val_loss:.3f} ppl={val_ppl:.2f}")

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                bad_evals = 0
                # save_checkpoint(model, "best_cpt.pt")
            else:
                bad_evals += 1
                if bad_evals >= patience:
                    print("Early stop: raw Korean val stopped improving")
                    break
```

---

# 1.2 Train exam prompt (staged)

You want:

* QA/example loss almost 0
* BUT raw Korean validation should remain similar to 1.1 (no forgetting)
* Final stage mixes QA + examples + raw Korean to keep both.

## Key issue

If you only train on “QA/example” data, you will often see:

* QA train loss → near 0
* QA val loss → low
* Raw Korean ppl gets worse (catastrophic forgetting)

So you need **two validations** at every stage:

1. `val_raw_ko` (same as 1.1)
2. `val_exam_prompt` (held-out QA/example format, not trained)

---

## Stage 1 and 2: “QA loss almost 0” — how to validate?

Define an **exam-format evaluation** that measures:

* **Prompt-format CE loss** on held-out examples
* Optionally **answer-only loss** (mask prompt tokens)

### Why answer-only loss?

If you include prompt tokens, the model can “cheat” by just copying prompt patterns. You care about answer generation.

**Implementation:** set label = `-100` for tokens that are part of the prompt, compute CE only on answer tokens.

---

## Pseudo code: answer-only loss for QA examples

Assume each QA item is formatted like:

```
<bos> Question: ... \n Answer: ... <eos>
```

and you can compute where answer begins.

```python
def make_labels_answer_only(input_ids, answer_start_idx, ignore_index=-100):
    """
    input_ids: [T]
    answer_start_idx: int index where answer tokens begin (in input_ids)
    returns labels: [T] shifted labels, but prompt region ignored
    """
    T = input_ids.size(0)
    labels = input_ids.clone()
    labels[:answer_start_idx] = ignore_index
    return labels

@torch.no_grad()
def eval_exam_answer_only(model, val_loader, device="cuda", max_batches=200):
    model.eval()
    total_loss, total_count = 0.0, 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches: break
        x = batch["input_ids"].to(device)          # [B,T]
        y = batch["labels"].to(device)             # [B,T], with -100 outside answer

        out = model(x)
        logits = out.logits if hasattr(out, "logits") else out

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        count = (y != -100).sum().item()
        total_loss += loss.item()
        total_count += count

    avg = total_loss / max(1, total_count)
    return avg
```

### Stage 1/2 stop rule

Stop stage 1 (and stage 2) when:

* `val_exam_answer_loss` stops improving (or drops below a target), AND
* `val_raw_ko_ppl` has not degraded beyond a tolerance (e.g. < 3–5% relative increase)

**Concrete criterion**

* Exam: `val_exam_answer_loss <= target` (e.g., 0.2–0.5) **or** early stop on plateau
* Raw: `val_raw_ko_loss <= best_raw_ko_loss * 1.03` (allow 3% regression max)

---

# Stage 3: mix QA + examples + raw Korean text

This stage is how you satisfy:

* “raw Korean validation similar to 1.1”
* “QA loss almost 0”

## How to verify “raw Korean is enough”?

You verify two curves at once:

1. **Raw Korean val ppl** stays near the best CPT baseline
2. **Exam answer-only val loss** stays low

If you can keep both stable for several evaluations, you’re done.

### Stage 3 stop rule (multi-objective)

Stop when BOTH conditions hold for N consecutive evaluations (e.g., 5):

* `raw_ko_val_loss <= baseline_raw_best * (1 + eps)` where eps = 0.02–0.05
* `exam_val_answer_loss <= target_exam_loss` (or within 5% of best exam loss)

This is the simplest “enough” definition.

---

## Stage 3 pseudo code: weighted multi-loader training + dual validation

```python
import random

def sample_batch(iters, weights):
    i = random.choices(range(len(iters)), weights=weights, k=1)[0]
    return next(iters[i]), i

def train_stage3(model, dl_exam, dl_raw, opt, device="cuda",
                 weights=(0.6, 0.4),  # (exam, raw)
                 eval_every=500, max_steps=30_000,
                 baseline_raw_best=None, raw_eps=0.03,
                 exam_target=0.3, consecutive=5):

    it_exam = iter(dl_exam)
    it_raw  = iter(dl_raw)

    ok_streak = 0

    for step in range(1, max_steps + 1):
        model.train()
        batch, src = sample_batch([it_exam, it_raw], weights)

        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

        out = model(x)
        logits = out.logits if hasattr(out, "logits") else out

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=-100 if src == 0 else -1000000,  # exam uses -100 mask; raw usually no mask
        )
        # For raw batches, you'd typically not set ignore_index at all;
        # keep separate collate for raw vs exam to avoid confusion.

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % eval_every == 0:
            raw_loss, raw_ppl = eval_causal_lm(model, dl_raw, device=device)
            exam_loss = eval_exam_answer_only(model, dl_exam, device=device)

            print(f"[S3] step={step} raw_loss={raw_loss:.3f} raw_ppl={raw_ppl:.2f} exam_ans_loss={exam_loss:.3f}")

            raw_ok = (baseline_raw_best is None) or (raw_loss <= baseline_raw_best * (1 + raw_eps))
            exam_ok = (exam_loss <= exam_target)

            if raw_ok and exam_ok:
                ok_streak += 1
                if ok_streak >= consecutive:
                    print("Stop: raw Korean retained and exam prompt solved")
                    break
            else:
                ok_streak = 0
```

**Note:** In real code, keep raw and exam loaders distinct:

* raw: standard shifted labels (no -100)
* exam: answer-only labels with -100 for prompt region

Do not mix ignore_index logic inside CE like above; keep it clean per batch type.

---

# Practical values that usually work

## Stage 1 / 2

* Use **answer-only loss**
* Use relatively higher learning rate than CPT (but not huge)
* Stop on plateau + check raw ppl regression

## Stage 3 mixing ratios

Start with:

* **(exam:raw) = 70:30**
  Then if raw ppl degrades:
* move to **50:50**
  If exam loss rises:
* move back to **60:40**

You are balancing retention vs specialization.

---

# “QA/example loss almost 0” is not sufficient by itself

Because it can mean overfitting to the prompt format.
You need these checks:

1. **Held-out exam validation** (not training examples)
2. **Answer-only loss**, not prompt+answer loss
3. **Raw Korean perplexity retention** vs 1.1 baseline

If all three pass, it is enough.

---

If you tell me your exact exam format (how you mark answer start), I can give a concrete `collate_fn` that computes `answer_start_idx` automatically and builds `labels` correctly for answer-only evaluation/training.
