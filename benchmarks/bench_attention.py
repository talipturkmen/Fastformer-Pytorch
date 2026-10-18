"""Time and peak memory of FastAttention forward+backward.

python benchmarks/bench_attention.py [--device cpu|cuda|mps] [--impl sdpa|einsum|reference]
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from reference_attention import reference_forward  # noqa: E402

from fast_transformer_torch import FastAttention, RotaryEmbedding  # noqa: E402


def saved_activation_mib(fn):
    """Bytes autograd keeps for the backward pass (unique storages), on any device."""
    storages = {}

    def pack(t):
        storages[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    out.sum().backward()
    return sum(storages.values()) / 2**20


def run(impl, device, batch, seq_len, dim, heads, dim_head, rotary, steps):
    torch.manual_seed(0)
    attn = FastAttention(dim, heads, dim_head, RotaryEmbedding(dim_head) if rotary else None).to(
        device
    )
    x = torch.randn(batch, seq_len, dim, device=device, requires_grad=True)
    fn = (lambda: reference_forward(attn, x)) if impl == "reference" else (lambda: attn(x))
    for _ in range(2):
        fn().sum().backward()
    sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(device, lambda: None)
    saved = saved_activation_mib(fn)
    sync()
    start = time.perf_counter()
    for _ in range(steps):
        fn().sum().backward()
    sync()
    return (time.perf_counter() - start) / steps * 1000, saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--impl", default="sdpa", choices=["sdpa", "einsum", "reference"])
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    for rotary in (False, True):
        for seq_len in (1024, 4096, 16384):
            ms, mem = run(args.impl, args.device, 4, seq_len, 512, 8, 64, rotary, args.steps)
            print(
                f"{args.impl:9s} rotary={rotary!s:5s} n={seq_len:6d}  "
                f"{ms:8.1f} ms/step  saved activations {mem:7.1f} MiB"
            )
