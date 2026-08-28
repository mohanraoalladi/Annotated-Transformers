# pipeline/dataloader.py

from torch.utils.data import DataLoader
import torch
from model.utils import Log
from pipeline.dataset import Multi30kDataset


def collate_fn(batch, vocab_src, vocab_tgt):
    pad_id_src = vocab_src.get_stoi()["<pad>"]
    pad_id_tgt = vocab_tgt.get_stoi()["<pad>"]

    src_batch = []
    tgt_batch = []

    for src_tokens, tgt_tokens in batch:

        # CORRECT: use __getitem__ to get a SINGLE ID per token
        src_ids = [vocab_src[t] for t in src_tokens]
        tgt_ids = [vocab_tgt[t] for t in tgt_tokens]

        src_batch.append(torch.tensor(src_ids, dtype=torch.long))
        tgt_batch.append(torch.tensor(tgt_ids, dtype=torch.long))

    # Padding
    src_max_len = max(x.size(0) for x in src_batch)
    tgt_max_len = max(x.size(0) for x in tgt_batch)

    padded_src = torch.full((len(src_batch), src_max_len), pad_id_src, dtype=torch.long)
    padded_tgt = torch.full((len(tgt_batch), tgt_max_len), pad_id_tgt, dtype=torch.long)

    for i, x in enumerate(src_batch):
        padded_src[i, :x.size(0)] = x
    for i, y in enumerate(tgt_batch):
        padded_tgt[i, :y.size(0)] = y

    return padded_src, padded_tgt


def create_dataloaders(vocab_src, vocab_tgt, spacy_de, spacy_en, batch_size):
    Log.yellow(">>> Loading dataset split: train")
    train_dataset = Multi30kDataset("train")

    Log.yellow(">>> Loading dataset split: validation")
    valid_dataset = Multi30kDataset("validation")

    Log.green(f"Train dataset size: {len(train_dataset)}")
    Log.green(f"Validation dataset size: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, vocab_src, vocab_tgt),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, vocab_src, vocab_tgt),
    )

    return train_loader, valid_loader
