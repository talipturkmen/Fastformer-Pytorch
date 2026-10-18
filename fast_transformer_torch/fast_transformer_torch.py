import torch
import torch.nn as nn

from .fast_attention import FastAttention, RotaryEmbedding


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FastTransformer(nn.Module):
    """Fastformer language model: token embedding, `depth` pre-norm blocks of
    additive attention and feed-forward, and a projection to token logits.

    Args:
        num_tokens: vocabulary size.
        dim: model width.
        depth: number of attention + feed-forward blocks.
        max_seq_len: longest sequence the model accepts.
        heads: number of attention heads.
        dim_head: width of each head.
        ff_mult: hidden width multiplier of the feed-forward layers.
        absolute_pos_emb: use learned absolute position embeddings instead of
            rotary embeddings.
        share_attn_logit_weights: share the query and key attention-logit
            projections across all layers, as proposed in the paper.
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        max_seq_len: int,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        absolute_pos_emb: bool = False,
        share_attn_logit_weights: bool = True,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)

        rotary_emb = None
        if absolute_pos_emb:
            self.abs_pos_emb = nn.Embedding(max_seq_len, dim)
        else:
            if dim_head % 2 != 0:
                raise ValueError("dim_head must be even to use rotary embeddings")
            self.abs_pos_emb = None
            rotary_emb = RotaryEmbedding(dim_head)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(dim),
                        FastAttention(dim, heads=heads, dim_head=dim_head, rotary_emb=rotary_emb),
                        nn.LayerNorm(dim),
                        FeedForward(dim, mult=ff_mult),
                    ]
                )
            )

        if share_attn_logit_weights and depth > 1:
            first_attn = self.layers[0][1]
            for _, attn, _, _ in self.layers[1:]:
                attn.to_q_attn_logits = first_attn.to_q_attn_logits
                attn.to_k_attn_logits = first_attn.to_k_attn_logits

        self.to_logits = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_tokens))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: token ids of shape (batch, seq_len).
            mask: optional bool tensor of shape (batch, seq_len); True marks
                real tokens, False marks padding.

        Returns:
            Logits of shape (batch, seq_len, num_tokens).
        """
        n = x.shape[1]
        if n > self.max_seq_len:
            raise ValueError(f"sequence length {n} exceeds max_seq_len {self.max_seq_len}")

        x = self.token_emb(x)
        if self.abs_pos_emb is not None:
            x = x + self.abs_pos_emb(torch.arange(n, device=x.device))

        for attn_norm, attn, ff_norm, ff in self.layers:
            x = attn(attn_norm(x), mask=mask) + x
            x = ff(ff_norm(x)) + x

        return self.to_logits(x)
