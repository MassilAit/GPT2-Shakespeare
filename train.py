# Script d'entraînement avec scheduler cosine, early stopping et support pour wandb.


import math
import wandb
import torch, torch.nn.functional as F
from typing import Optional, Callable, Dict, List
from contextlib import nullcontext



def cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 200,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int):
        if step < warmup_steps:                          # warm-up phase
            return float(step + 1) / warmup_steps
        # cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(torch.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)




def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    *,
    epochs: int                     = 6,
    lr: float                       = 3e-4,
    device: Optional[torch.device]  = None,
    clip_grad_norm: Optional[float] = 1.0,
    weight_decay: float             = 0.05,
    warmup_steps: int               = 200,
    wandb_project: Optional[str]    = None,   
    wandb_run_name: Optional[str]   = None,
    log_every : int                 =5,
    verbose: bool                   = True,
    early_stopping_patience: int    = 5,
    early_stopping_delta: float     = 0.0, 
    max_val_loss_increase: float    = 0.5, 
) -> Dict[str, List[float]]:
    
    # -------------- wandb init (optional) ----------------------------
    if wandb_project is not None:
        wandb.init(project=wandb_project, name=wandb_run_name,
                   config=dict(epochs=epochs, lr=lr,
                                weight_decay=weight_decay,
                                warmup_steps=warmup_steps,
                                batch_size=train_loader.batch_size))
    log_to_wandb = wandb_project is not None

    # ------ device -----------------------------------------------------
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    # ------ optimiser & scheduler -------------------------------------
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * len(train_loader)
    scheduler   = cosine_with_warmup(optim, total_steps, warmup_steps)

    history = {"train": [], "val": []}

    # ------ automatic BF16 autocast -----------------------------------
    if device.type == "cuda":
        autocast_ctx = torch.autocast("cuda", torch.bfloat16)
    elif hasattr(torch.cpu, "is_bf16_supported") and torch.cpu.is_bf16_supported():
        autocast_ctx = torch.autocast("cpu",  torch.bfloat16)
    else:
        autocast_ctx = nullcontext()                       # fall back to FP32
    # ------------------------------------------------------------------

    def run_epoch(loader, train_flag: bool) -> float:
        model.train(train_flag)
        total = 0.0

        for step, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            with torch.set_grad_enabled(train_flag), autocast_ctx:
                logits = model(xb)
                loss   = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            yb.view(-1)
                         )

            if train_flag:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optim.step()
                scheduler.step()

                if log_to_wandb and (step % log_every == 0) and train_flag:
                    wandb.log({
                        "train_loss_batch": loss.item(),
                        "lr":         optim.param_groups[0]["lr"],
                    })

            total += loss.item()


        return total / len(loader)

    # ------ training loop ---------------------------------------------

        # EARLY STOPPING state
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in range(1, epochs + 1):
        tr_loss = run_epoch(train_loader, train_flag=True)
        vl_loss = run_epoch(val_loader,   train_flag=False)

        history["train"].append(tr_loss)
        history["val"].append(vl_loss)

        if verbose:
            print(f"Epoch {epoch:2d} | train {tr_loss:.4f} | val {vl_loss:.4f}")


        if log_to_wandb:
            wandb.log({
                "train_loss_epoch": tr_loss,
                "val_loss_epoch":   vl_loss,
            })


        # --- EARLY STOPPING LOGIC ---
        if vl_loss < best_val_loss - early_stopping_delta:
            best_val_loss = vl_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= early_stopping_patience:
            if verbose:
                print(f"Early stopping triggered (no improvement for {early_stopping_patience} epochs).")
            break

        if vl_loss > max_val_loss_increase * best_val_loss:
            if verbose:
                print(f"Early stopping triggered (validation loss increased too much).")
            break
        # -----------------------------
    
    if log_to_wandb:
        wandb.finish()

    return history
