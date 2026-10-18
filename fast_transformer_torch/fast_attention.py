import torch
import torch.nn as nn
import torch.nn.functional as F
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
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.use_sdpa = use_sdpa
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

        # global query: additive-attention pooling of the queries
        w_q = self.to_q_attn_logits.weight[0].expand(*q.shape[:2], -1)
        global_q = self._pool(w_q, q, q_aggr, mask)

        # global key: additive-attention pooling of p = k * global_q.
        #   logits_n = w_k . p_n          = k_n . (global_q * w_k)
        #   global_k = sum_n a_n * p_n    = global_q * sum_n a_n * k_n
        # With rotary embeddings, w_k acts on sums of adjacent feature pairs, so
        # it is repeated over each pair.
        w_k = self._pair_expand(self.to_k_attn_logits.weight[0], dim=0)
        global_k = global_q * self._pool(global_q * w_k, k_aggr, k_aggr, mask)

        # output transform of u = v * global_k (pair-summed with rotary), with
        # the queries as a residual: to_r(u) = v @ (diag(global_k) @ W_r^T) + b_r
        w_r = self._pair_expand(self.to_r.weight, dim=1)  # (dim_head, dim_head)
        r = torch.einsum("b h n d, b h d e -> b h n e", v_aggr, global_k.unsqueeze(-1) * w_r.t())
        r = r + self.to_r.bias + q
        r = rearrange(r, "b h n d -> b n (h d)")
        return self.to_out(r)

    def _pool(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Additive-attention pooling: sum_n softmax_n(scale * query . key_n) * value_n.

        This is attention with a single query token, so it runs through
        ``scaled_dot_product_attention`` (fused flash / memory-efficient kernels
        on CUDA). query: (b, h, d); key, value: (b, h, n, d); returns (b, h, d).
        """
        if self.use_sdpa:
            attn_mask = None if mask is None else rearrange(mask, "b n -> b 1 1 n")
            out = F.scaled_dot_product_attention(
                query.unsqueeze(-2), key, value, attn_mask=attn_mask, scale=self.scale
            )
            return out.squeeze(-2)
        logits = torch.einsum("b h n d, b h d -> b h n", key, query) * self.scale
        if mask is not None:
            logits = logits.masked_fill(
                ~rearrange(mask, "b n -> b 1 n"), torch.finfo(logits.dtype).min
            )
        return torch.einsum("b h n, b h n d -> b h d", logits.softmax(dim=-1), value)

    def _pair_expand(self, weight: torch.Tensor, dim: int) -> torch.Tensor:
        if self.rotary_emb is None:
            return weight
        return weight.repeat_interleave(2, dim=dim)
