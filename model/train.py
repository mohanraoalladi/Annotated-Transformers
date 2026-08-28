import torch
import torch.nn as nn

from model.utils import (
    Log, Batch, TrainState, SimpleLossCompute,
    DummyOptimizer, DummyScheduler, rate, run_epoch
)
from model.transformer import make_model
from pipeline.dataloader import create_dataloaders
from torch.nn import KLDivLoss
from torch.optim.lr_scheduler import LambdaLR


class LabelSmoothing(torch.nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super().__init__()
        self.criterion = KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x, target):
        true_dist = x.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        return self.criterion(x, true_dist)


def train_worker(gpu, ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, is_distributed=False):
    Log.blue(">>> Training worker started")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    Log.green(f"Using device: {device}")

    model = make_model(len(vocab_src), len(vocab_tgt), N=6).to(device)
    Log.yellow(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    pad_idx = vocab_tgt.get_stoi()["<pad>"]

    criterion = LabelSmoothing(
        size=len(vocab_tgt),
        padding_idx=pad_idx,
        smoothing=0.1
    ).to(device)

    train_loader, valid_loader = create_dataloaders(
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        config["batch_size"]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["base_lr"],
        betas=(0.9, 0.98),
        eps=1e-9
    )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: rate(step, 512, 1, config["warmup"])
    )

    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        Log.blue(f"--- Epoch {epoch} started ---")

        model.train()
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx, device) for b in train_loader),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            scheduler,
            mode="train",
            accum_iter=config["accum_iter"],
            train_state=train_state,
            config=config
        )

        model.eval()
        sloss, _ = run_epoch(
            (Batch(b[0], b[1], pad_idx, device) for b in valid_loader),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
            config=config
        )

        Log.green(f"Validation loss: {sloss}")

    torch.save(model.state_dict(), f"{config['file_prefix']}final.pt")
    Log.green(">>> Training complete. Model saved.")


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    train_worker(0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False)
