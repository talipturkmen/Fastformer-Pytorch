import torch
import torch.nn as nn
from einops import rearrange


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
        self._cache: tuple[int, torch.device, torch.dtype, torch.Tensor, torch.Tensor] | None = None

    def _cos_sin(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # cos/sin tables are shared by every layer and reused across calls
        cache = self._cache
        if cache is None or cache[0] < n or cache[1] != device or cache[2] != dtype:
            pos = torch.arange(n, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(pos, self.inv_freq.to(device))  # (n, dim / 2): one angle per pair
            cache = (n, device, dtype, freqs.cos().to(dtype), freqs.sin().to(dtype))
            self._cache = cache
        return cache[3][:n], cache[4][:n]

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Rotates `t` of shape (..., n, d) by its position along the n axis."""
        cos, sin = self._cos_sin(t.shape[-2], t.device, t.dtype)
        x1, x2 = t.unflatten(-1, (-1, 2)).unbind(dim=-1)  # each adjacent pair (x1, x2)
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class FastAttention(nn.Module):
    """Additive attention from "Fastformer: Additive Attention Can Be All You Need".

    https://arxiv.org/abs/2108.09084

    Runs in O(n) time and memory: the queries are pooled into one global query,
    which modulates the keys; the keys are pooled into one global key, which
    modulates the values.

    The modulated keys ``p = k * global_q`` and values ``u = v * global_k`` are
    never materialised: both are only used through linear maps, so the global
    vectors are folded into those maps instead. This is exact and saves two
    (batch, heads, seq_len, dim_head) activations per layer.
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
        qkv = rearrange(self.to_qkv(x), "b n (three h d) -> three b h n d", three=3, h=self.heads)
        q = qkv[0]
        q_aggr, k_aggr, v_aggr = (
            self.rotary_emb(qkv) if self.rotary_emb is not None else qkv
        ).unbind(0)

        if mask is not None:
            mask = rearrange(mask, "b n -> b 1 n")
        mask_value = torch.finfo(x.dtype).min

        # global query: additive-attention pooling of the queries
        q_attn_logits = (
            torch.einsum("b h n d, d -> b h n", q, self.to_q_attn_logits.weight[0]) * self.scale
        )
        if mask is not None:
            q_attn_logits = q_attn_logits.masked_fill(~mask, mask_value)
        global_q = torch.einsum("b h n, b h n d -> b h d", q_attn_logits.softmax(dim=-1), q_aggr)

        # global key: additive-attention pooling of p = k * global_q.
        #   logits_n = w_k . p_n          = k_n . (global_q * w_k)
        #   global_k = sum_n a_n * p_n    = global_q * sum_n a_n * k_n
        # With rotary embeddings, w_k acts on sums of adjacent feature pairs, so
        # it is repeated over each pair.
        w_k = self._pair_expand(self.to_k_attn_logits.weight[0], dim=0)
        k_attn_logits = torch.einsum("b h n d, b h d -> b h n", k_aggr, global_q * w_k) * self.scale
        if mask is not None:
            k_attn_logits = k_attn_logits.masked_fill(~mask, mask_value)
        k_pooled = torch.einsum("b h n, b h n d -> b h d", k_attn_logits.softmax(dim=-1), k_aggr)
        global_k = global_q * k_pooled

        # output transform of u = v * global_k (pair-summed with rotary), with
        # the queries as a residual: to_r(u) = v @ (diag(global_k) @ W_r^T) + b_r
        w_r = self._pair_expand(self.to_r.weight, dim=1)  # (dim_head, dim_head)
        r = torch.einsum("b h n d, b h d e -> b h n e", v_aggr, global_k.unsqueeze(-1) * w_r.t())
        r = r + self.to_r.bias + q
        r = rearrange(r, "b h n d -> b n (h d)")
        return self.to_out(r)

    def _pair_expand(self, weight: torch.Tensor, dim: int) -> torch.Tensor:
        if self.rotary_emb is None:
            return weight
        return weight.repeat_interleave(2, dim=dim)
