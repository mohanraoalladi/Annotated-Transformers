# pipeline/dataset.py

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from model.utils import Log


class Multi30kDataset(Dataset):
    """
    Loads Multi30k automatically using HuggingFace datasets.
    """

    def __init__(self, split):
        Log.yellow(f"Loading Multi30k split: {split}")

        hf_split = "train" if split == "train" else "validation"

        dataset = load_dataset("bentrevett/multi30k", split=hf_split)

        self.src_lines = dataset["de"]
        self.tgt_lines = dataset["en"]

        assert len(self.src_lines) == len(self.tgt_lines)

        Log.green(f"{split.capitalize()} dataset size: {len(self.src_lines)}")

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src = self.src_lines[idx].split()
        tgt = self.tgt_lines[idx].split()

        # ADD SPECIAL TOKENS
        src = ["<s>"] + src + ["</s>"]
        tgt = ["<s>"] + tgt + ["</s>"]

        return src, tgt
