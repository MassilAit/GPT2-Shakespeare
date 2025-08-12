# GPT-2 Shakespeare — From-Scratch vs Fine-Tuning

This project compares two approaches for generating Shakespeare-style text with GPT-2:

1) **Fine-tuning** a pretrained GPT-2 (124M) on *Tiny Shakespeare*  
2) **Re-implementing GPT-2 from scratch** in PyTorch and training on the same corpus

The goal is to understand the trade-offs between reusing a pretrained model and training a custom implementation, and to document the training/decoding choices that make both work well on a small corpus.

---

## Highlights

- **Datasets**: Tiny Shakespeare (≈40k lines) with GPT-2 tokenizer; 90/10 train/val split and blockified sequences (context up to 1024).  
- **Models**:
  - *Fine-tune*: Hugging Face GPT-2 (124M) adapted to the corpus.
  - *From-scratch*: Minimal GPT-2 (decoder-only Transformer) in PyTorch, faithful to the original architecture, with a few practical tweaks (dropout; loss computed in the training loop).
- **Training**: AdamW, cosine LR with warmup, gradient clipping (1.0), dropout, early stopping.
- **GPU/Speedups**: `torch.compile`, AMP/`bfloat16`, and FlashAttention for faster training.
- **Decoding**: Top-p sampling (p=0.9), temperature=0.8 for a good style/variety balance.
- **Key takeaways**:  
  - From-scratch model is **very stylistically faithful** to Shakespeare but narrower in knowledge.  
  - Fine-tuning retains broader knowledge and is **more flexible/creative**; longer fine-tune can overfit to Shakespeare names/themes.

---

## Results (Tiny Shakespeare)

| Setting | Train Loss | Val Loss | Notes |
|---|---:|---:|---|
| From-scratch (progressive context; early stop) | — | **≈4.75** | Progressive 128→512→1024 contexts stabilized training. |
| Fine-tune (1 epoch) | **3.8225** | **3.3424** | Keeps more general knowledge; outputs still in Shakespearean style. |
| Fine-tune (≈6 epochs) | **≈3.10** | **≈3.20** | Lower loss but tends to “forget” general knowledge and mimic Shakespeare names/structures more rigidly. |

> We generally use **top-p = 0.9** and **temperature = 0.8** for sampling.

---
## Quickstart

1. Open **`INF8225_Project.ipynb`** in Jupyter Notebook or Jupyter Lab.

2. The code is organized in separate files:
   - **`model.py`** — GPT model definition (`GPT`, `GPTConfig`)
   - **`dataloader.py`** — data loading utilities
   - **`train.py`** — training loop

3. In the notebook, choose your model:
   
   # From-scratch model
   model = GPT(GPTConfig())

   # OR: Hugging Face pretrained GPT-2 (124M) for fine-tuning
   model = GPT.from_pretrained("gpt2")
