"""The v0.2.0 FastAttention forward pass, kept as a numerical reference."""

import torch
from einops import rearrange, reduce


def reference_rotary(t, inv_freq):
    """The v0.2.0 rotary embedding: full-width tables and a stacked rotate-half copy."""
    n = t.shape[-2]
    pos = torch.arange(n, device=t.device, dtype=inv_freq.dtype)
    freqs = torch.einsum("n, f -> n f", pos, inv_freq).repeat_interleave(2, dim=-1).to(t.dtype)
    x1, x2 = rearrange(t, "... (d r) -> ... d r", r=2).unbind(dim=-1)
    rotated = rearrange(torch.stack((-x2, x1), dim=-1), "... d r -> ... (d r)")
    return t * freqs.cos() + rotated * freqs.sin()


def reference_forward(attn, x, mask=None):
    q, k, v = attn.to_qkv(x).chunk(3, dim=-1)
    q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=attn.heads) for t in (q, k, v))
    rotary = attn.rotary_emb is not None
    if rotary:
        q_aggr, k_aggr, v_aggr = (reference_rotary(t, attn.rotary_emb.inv_freq) for t in (q, k, v))
    else:
        q_aggr, k_aggr, v_aggr = q, k, v
    if mask is not None:
        mask = rearrange(mask, "b n -> b 1 n")
    mask_value = torch.finfo(x.dtype).min

    q_logits = rearrange(attn.to_q_attn_logits(q), "b h n 1 -> b h n") * attn.scale
    if mask is not None:
        q_logits = q_logits.masked_fill(~mask, mask_value)
    global_q = torch.einsum("b h n, b h n d -> b h d", q_logits.softmax(-1), q_aggr)

    p = k_aggr * rearrange(global_q, "b h d -> b h 1 d")
    p_in = reduce(p, "b h n (d r) -> b h n d", "sum", r=2) if rotary else p
    k_logits = rearrange(attn.to_k_attn_logits(p_in), "b h n 1 -> b h n") * attn.scale
    if mask is not None:
        k_logits = k_logits.masked_fill(~mask, mask_value)
    global_k = torch.einsum("b h n, b h n d -> b h d", k_logits.softmax(-1), p)

    u = v_aggr * rearrange(global_k, "b h d -> b h 1 d")
    if rotary:
        u = reduce(u, "b h n (d r) -> b h n d", "sum", r=2)
    r = attn.to_r(u) + q
    return attn.to_out(rearrange(r, "b h n d -> b n (h d)"))
