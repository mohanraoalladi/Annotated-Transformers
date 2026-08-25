# pipeline/dataloader.py

from model.utils import Log
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.functional import pad
from datasets import load_dataset
from .tokenize import tokenize


class Multi30kDataset(Dataset):
    def __init__(self, split):
        Log.yellow(f">>> Loading dataset split: {split}")
        ds = load_dataset("bentrevett/multi30k", split=split, trust_remote_code=False)
        self.data = list(ds)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return (self.data[idx]["de"], self.data[idx]["en"])


def collate_batch(batch, src_pipeline, tgt_pipeline, src_vocab, tgt_vocab,
                  device, max_padding=128, pad_id=2):

    Log.yellow("Collating batch...")

    bos_id = torch.tensor([0], device=device)
    eos_id = torch.tensor([1], device=device)

    src_list, tgt_list = [], []

    for src_text, tgt_text in batch:
        src_tokens = src_pipeline(src_text)
        tgt_tokens = tgt_pipeline(tgt_text)

        src_ids = torch.cat([bos_id,
                             torch.tensor(src_vocab(src_tokens), device=device),
                             eos_id])

        tgt_ids = torch.cat([bos_id,
                             torch.tensor(tgt_vocab(tgt_tokens), device=device),
                             eos_id])

        src_padded = pad(src_ids, (0, max_padding - len(src_ids)), value=pad_id)
        tgt_padded = pad(tgt_ids, (0, max_padding - len(tgt_ids)), value=pad_id)

        Log.blue(f"Source padded shape: {src_padded.shape}")
        Log.blue(f"Target padded shape: {tgt_padded.shape}")

        src_list.append(src_padded)
        tgt_list.append(tgt_padded)

    return torch.stack(src_list), torch.stack(tgt_list)


def create_dataloaders(device, vocab_src, vocab_tgt, spacy_de, spacy_en,
                       batch_size=12000, max_padding=128, is_distributed=True):

    Log.blue(">>> Creating dataloaders...")

    def tokenize_de(text): return tokenize(text, spacy_de)
    def tokenize_en(text): return tokenize(text, spacy_en)

    train_dataset = Multi30kDataset("train")
    valid_dataset = Multi30kDataset("validation")

    Log.green(f"Train dataset size: {len(train_dataset)}")
    Log.green(f"Validation dataset size: {len(valid_dataset)}")

    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    valid_sampler = DistributedSampler(valid_dataset) if is_distributed else None

    def collate_fn(batch):
        return collate_batch(batch, tokenize_de, tokenize_en,
                             vocab_src, vocab_tgt, device,
                             max_padding=max_padding,
                             pad_id=vocab_src.get_stoi()["<blank>"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler, collate_fn=collate_fn)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=(valid_sampler is None),
                              sampler=valid_sampler, collate_fn=collate_fn)

    return train_loader, valid_loader
