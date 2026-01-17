You prevent damage to the original token embeddings by doing **two things**:

1. **Freeze everything except the embedding table** (and optionally LM head).
2. **Mask gradients so only the new token rows receive updates**.

That way:

* Old token rows **never change** (their gradients are forced to zero).
* New token rows learn normally from your Korean data.

Below is a complete, minimal PyTorch pattern.

---

## Concept

Let the embedding matrix be:

* `E ∈ R[V × H]`
* Old tokens: rows `[0 ... V_old-1]`
* New tokens: rows `[V_old ... V_new-1]`

During backprop, PyTorch produces `grad(E)` for all rows that appeared in the batch.

We register a **gradient hook**:

* set `grad(E)[:V_old] = 0`
* keep `grad(E)[V_old:]` unchanged

Then optimizer updates only new rows.

---

## Code: train only new token embeddings (no hurt to old tokens)

```python
import torch
import torch.nn.functional as F

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_embedding_only(model, tied=True):
    # adjust attribute names if your model differs
    model.embed_tokens.weight.requires_grad = True
    if (not tied) and hasattr(model, "lm_head"):
        model.lm_head.weight.requires_grad = True

def mask_old_rows(param: torch.nn.Parameter, v_old: int):
    """
    param is an embedding weight matrix [V_new, H].
    This hook zeros gradients for old token rows, so only new token rows update.
    """
    def hook(grad):
        # grad has same shape as param
        grad[:v_old].zero_()
        return grad
    param.register_hook(hook)

def train_new_token_embeddings(model, dataloader, v_old, steps=2000, lr=5e-4, tied=True, device="cuda"):
    model.to(device)
    model.train()

    # 1) Freeze everything
    freeze_all(model)

    # 2) Only embeddings are trainable
    unfreeze_embedding_only(model, tied=tied)

    # 3) Gradient mask: old token rows never update
    mask_old_rows(model.embed_tokens.weight, v_old)
    if (not tied) and hasattr(model, "lm_head"):
        mask_old_rows(model.lm_head.weight, v_old)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    it = iter(dataloader)
    for step in range(steps):
        batch = next(it)
        x = batch["input_ids"].to(device)   # [B,T]
        y = batch["labels"].to(device)      # [B,T]

        # forward
        out = model(x)
        logits = out.logits if hasattr(out, "logits") else out   # HF vs custom
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"step={step} loss={loss.item():.4f}")
```

### What guarantees “no hurt”?

* **Old rows never change** because their gradients are forcibly zeroed every step.
* Even if old tokens appear in the batch, their gradient is discarded.

---

## Optional: verify old embeddings did not change

This check ensures your masking is working.

```python
@torch.no_grad()
def verify_old_rows_unchanged(model, old_snapshot, v_old, atol=0.0):
    cur = model.embed_tokens.weight[:v_old].detach().cpu()
    diff = (cur - old_snapshot).abs().max().item()
    print("max_abs_diff(old_rows) =", diff)
    assert diff <= atol, "Old embeddings changed!"
```

Usage before training:

```python
old_snapshot = model.embed_tokens.weight[:v_old].detach().cpu().clone()
# train...
verify_old_rows_unchanged(model, old_snapshot, v_old, atol=0.0)
```

---

## How the new tokens actually learn (mechanism)

When a new token ID appears in `input_ids`, the model looks up its embedding row `E[token_id]`.

The loss gradient flows back through the network to that row:

* `dLoss/dE[new_token_id] ≠ 0` (so it updates)
* `dLoss/dE[old_token_id] = 0` after our mask (so it never updates)

So new tokens learn “meaning” by being used in real contexts, while old tokens are unchanged.

---

## Practical tips (so it works with small data)

* Ensure your batches contain new tokens frequently (oversample “important sentences” and synthetic contexts).
* Initialize new rows with **mean** of old embeddings (better than zeros).
* Do a short embedding warm-up (1k–10k steps), then continue with LoRA for broader adaptation.

---

If you tell me whether your model uses **tied embeddings** (input embedding == lm head weight), I’ll give you the exact safe setting and whether you should mask `lm_head` separately.
