import torch
import torch.nn as nn
from einops import rearrange, reduce


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding over interleaved pairs of feature dimensions.

    https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"rotary dimension must be even, got {dim}")
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Rotates `t` of shape (..., n, d) by its position along the n axis."""
        n = t.shape[-2]
        pos = torch.arange(n, device=t.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("n, f -> n f", pos, self.inv_freq)
        freqs = freqs.repeat_interleave(2, dim=-1).to(t.dtype)
        return t * freqs.cos() + _rotate_half(t) * freqs.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    return rearrange(torch.stack((-x2, x1), dim=-1), "... d r -> ... (d r)")


class FastAttention(nn.Module):
    """Additive attention from "Fastformer: Additive Attention Can Be All You Need".

    https://arxiv.org/abs/2108.09084

    Runs in O(n) time and memory: the queries are pooled into one global query,
    which modulates the keys; the keys are pooled into one global key, which
    modulates the values.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        rotary_emb: RotaryEmbedding | None = None,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = rotary_emb

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # with rotary embeddings, adjacent feature pairs are summed after the
        # element-wise products, which recovers a relative-position inner product
        kv_dim = dim_head // 2 if rotary_emb is not None else dim_head

        self.to_q_attn_logits = nn.Linear(dim_head, 1, bias=False)
        self.to_k_attn_logits = nn.Linear(kv_dim, 1, bias=False)
        self.to_r = nn.Linear(kv_dim, dim_head)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: input of shape (batch, seq_len, dim).
            mask: optional bool tensor of shape (batch, seq_len); True marks
                tokens to attend to, False marks padding.

        Returns:
            Tensor of shape (batch, seq_len, dim).
        """
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))

        if self.rotary_emb is not None:
            q_aggr, k_aggr, v_aggr = (self.rotary_emb(t) for t in (q, k, v))
        else:
            q_aggr, k_aggr, v_aggr = q, k, v

        if mask is not None:
            mask = rearrange(mask, "b n -> b 1 n")
        mask_value = torch.finfo(x.dtype).min

        # global query: additive-attention pooling of the queries
        q_attn_logits = rearrange(self.to_q_attn_logits(q), "b h n 1 -> b h n") * self.scale
        if mask is not None:
            q_attn_logits = q_attn_logits.masked_fill(~mask, mask_value)
        q_attn = q_attn_logits.softmax(dim=-1)
        global_q = torch.einsum("b h n, b h n d -> b h d", q_attn, q_aggr)

        # keys modulated by the global query
        p = k_aggr * rearrange(global_q, "b h d -> b h 1 d")
        p_logits_in = p
        if self.rotary_emb is not None:
            p_logits_in = reduce(p, "b h n (d r) -> b h n d", "sum", r=2)

        # global key: additive-attention pooling of the modulated keys
        k_attn_logits = (
            rearrange(self.to_k_attn_logits(p_logits_in), "b h n 1 -> b h n") * self.scale
        )
        if mask is not None:
            k_attn_logits = k_attn_logits.masked_fill(~mask, mask_value)
        k_attn = k_attn_logits.softmax(dim=-1)
        global_k = torch.einsum("b h n, b h n d -> b h d", k_attn, p)

        # values modulated by the global key
        u = v_aggr * rearrange(global_k, "b h d -> b h 1 d")
        if self.rotary_emb is not None:
            u = reduce(u, "b h n (d r) -> b h n d", "sum", r=2)

        # output transform, with the queries as a residual
        r = self.to_r(u) + q
        r = rearrange(r, "b h n d -> b n (h d)")
        return self.to_out(r)
