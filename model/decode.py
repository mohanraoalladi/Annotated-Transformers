# model/decode.py

from model.utils import Log, Batch
import torch

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    Log.blue(">>> Starting greedy decode")

    memory = model.encode(src, src_mask)
    Log.yellow(f"Memory shape: {memory.shape}")

    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src)
    Log.green(f"Initial token: {start_symbol}")

    for i in range(max_len - 1):
        tgt_mask = Batch.make_std_mask(ys, pad=2)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        Log.yellow(f"Step {i}: next_word = {next_word}")

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src).fill_(next_word)], dim=1)

    return ys[0]