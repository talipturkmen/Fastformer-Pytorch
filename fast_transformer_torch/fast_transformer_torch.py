import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding
from fast_attention import FastAttention


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super(FeedForward, self).__init__()
        self.dim = dim
        self.mult = mult

        self.ff = nn.Sequential(
                nn.Linear(dim, dim * mult),
                nn.GELU(),
                nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.ff(x)


class FastTransformer(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        heads=8,
        dim_head=64,
        ff_mult=4,
        absolute_pos_emb=False,
        mask=None,
    ):
        super(FastTransformer, self).__init__()

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.mask = mask

        # positional embeddings
        if absolute_pos_emb:
            self.abs_pos_emb = nn.Embedding(max_seq_len, dim)
        else:
            self.abs_pos_emb = None

        layer_pos_emb = None
        if not absolute_pos_emb:
            assert (dim_head % 4) == 0, ( "dimension of the head must be divisible by 4 to use rotary embeddings")
            layer_pos_emb = RotaryEmbedding(dim_head // 2)

        fast_tranformer_layers = []

        for _ in range(depth):
            attn = FastAttention(
                dim,
                dim_head=dim_head,
                heads=heads,
                pos_emb=layer_pos_emb,
                max_seq_len=max_seq_len,
                mask=self.mask,
            )
            ff = FeedForward(dim, mult=ff_mult)

            fast_tranformer_layers.append(PreNorm(dim, attn))
            fast_tranformer_layers.append(PreNorm(dim, ff))

        self.fast_tranformer_layers = nn.ModuleList(fast_tranformer_layers)

        first_block = self.fast_tranformer_layers[0]
        for block in self.fast_tranformer_layers[1:]:
            block.fn.to_q_attn_logits = first_block.fn.to_q_attn_logits
            block.fn.to_k_attn_logits = first_block.fn.to_k_attn_logits

        self.to_logits = nn.Sequential(nn.LayerNorm(dim),
                                       nn.Linear(dim, num_tokens),
                                       )

    def forward(self, x):
        n = x.shape[1]
        x = self.token_emb(x)

        if self.abs_pos_emb is not None:
            pos_emb = self.abs_pos_emb(torch.arange(n))
            x = x + pos_emb.unsqueeze(0)

        for current_layer in self.fast_tranformer_layers:
            x = current_layer(x) + x

        return self.to_logits(x)


if __name__ == '__main__':
    mask = torch.ones([16, 4096], dtype=torch.bool)
    model = FastTransformer(num_tokens = 20000,
                            dim = 512,
                            depth = 2,
                            max_seq_len = 4096,
                            absolute_pos_emb = True, # Absolute positional embeddings
                            mask = mask
                            )
    x = torch.randint(0, 20000, (16, 4096))

    logits = model(x) # (1, 4096, 20000)
    print(logits.shape)