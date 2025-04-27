# DataLoader pour tokenizer un fichier texte et créer des séquences avec overlap pour l'entraînement.

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from pathlib import Path
from typing import Tuple

class GPTOverlapDataset(Dataset):

    def __init__(
        self,
        tokens: torch.Tensor,
        block_size: int,
        overlap: int = 0,
    ):
        assert 0 <= overlap < block_size, "`overlap` must be in [0, block_size-1]"
        self.tokens = tokens
        self.block  = block_size
        self.stride = block_size - overlap
        # number of start positions whose [start + 1 + block] stays in-bounds
        self.n      = (len(tokens) - block_size - 1) // self.stride + 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        x = self.tokens[start              : start + self.block]
        y = self.tokens[start + 1          : start + 1 + self.block]
        return x, y



def make_loaders(
    path: str | Path,
    block_size: int,
    overlap: int,
    batch_size: int,
    val_frac: float = 0.1,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader) with token-disjoint splits.
    """
    # 1. tokenise
    enc = tiktoken.get_encoding("gpt2")
    tokens = torch.tensor(enc.encode(Path(path).read_text()), dtype=torch.long)

    # 2. split the token tensor once
    split = int(len(tokens) * (1 - val_frac))
    train_tok, val_tok = tokens[:split], tokens[split:]

    # 3. build datasets
    train_ds = GPTOverlapDataset(train_tok, block_size, overlap)
    val_ds   = GPTOverlapDataset(val_tok,   block_size, overlap)

    # 4. data loaders
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,          
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,         
        num_workers=num_workers,
        pin_memory=True,
    )

    print("Number of batches : ")
    print(f"Number of training batches : {len(train_dl)}")
    print(f"Number of validation batches : {len(val_dl)}\n")

    print(f"Batch size : {batch_size}\n")

    print("number of exemples : ")
    print(f"Number of training exemples : {len(train_dl.dataset)}")
    print(f"Number of validation exemples : {len(val_dl.dataset)}\n")


    return train_dl, val_dl