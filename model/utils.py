# model/utils.py

# ===== Colored Logging =====
class Log:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    END = "\033[0m"

    @staticmethod
    def red(msg):
        print(f"{Log.RED}{msg}{Log.END}")

    @staticmethod
    def green(msg):
        print(f"{Log.GREEN}{msg}{Log.END}")

    @staticmethod
    def yellow(msg):
        print(f"{Log.YELLOW}{msg}{Log.END}")

    @staticmethod
    def blue(msg):
        print(f"{Log.BLUE}{msg}{Log.END}")


import torch

class Batch:
    def __init__(self, src, tgt, pad=2):
        self.src = src
        self.tgt = tgt
        self.src_mask = (src != pad).unsqueeze(-2)
        self.tgt_mask = self.make_std_mask(tgt, pad)
        self.ntokens = (tgt != pad).sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        size = tgt.size(-1)
        subsequent_mask = torch.triu(torch.ones(1, size, size), diagonal=1).type_as(tgt_mask) == 0
        return tgt_mask & subsequent_mask


class TrainState:
    def __init__(self):
        self.step = 0
        self.accum_step = 0
        self.samples = 0
        self.tokens = 0


class DummyOptimizer:
    def step(self): pass
    def zero_grad(self): pass


class DummyScheduler:
    def step(self): pass


def rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


def run_epoch(data_iter, model, loss_compute, optimizer, scheduler,
              mode="train", accum_iter=1, train_state=None, config=None):

    Log.blue(">>> Running epoch...")

    total_loss = 0
    total_tokens = 0

    for i, batch in enumerate(data_iter):
        Log.yellow(f"Batch {i}: src={batch.src.shape}, tgt={batch.tgt.shape}")

        out = model.decode(
            model.encode(batch.src, batch.src_mask),
            batch.src_mask,
            batch.tgt[:, :-1],
            batch.tgt_mask[:, :-1, :-1],
        )

        loss = loss_compute(out, batch.tgt[:, 1:], batch.ntokens)

        Log.green(f"Loss: {loss.item()}")
        Log.blue(f"Tokens: {batch.ntokens}")

        if mode.startswith("train"):
            loss.backward()
            if (i + 1) % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                Log.yellow(f"LR: {optimizer.param_groups[0]['lr']}")

        total_loss += loss.item()
        total_tokens += batch.ntokens.item()

    Log.green(">>> Epoch complete")
    return total_loss / total_tokens, train_state


class SimpleLossCompute:
    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(
            x.contiguous().view(-1, x.size(-1)),
            y.contiguous().view(-1)
        ) / norm
        return loss
