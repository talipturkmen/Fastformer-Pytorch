import pytest
import torch

from fast_transformer_torch import FastAttention, FastTransformer, RotaryEmbedding

NUM_TOKENS = 100
SEQ_LEN = 32


def make_model(**kwargs):
    torch.manual_seed(0)
    params = dict(num_tokens=NUM_TOKENS, dim=32, depth=2, max_seq_len=SEQ_LEN, heads=4, dim_head=8)
    params.update(kwargs)
    return FastTransformer(**params).eval()


@pytest.mark.parametrize("absolute_pos_emb", [False, True])
def test_output_shape_and_finite(absolute_pos_emb):
    model = make_model(absolute_pos_emb=absolute_pos_emb)
    x = torch.randint(0, NUM_TOKENS, (3, SEQ_LEN))
    logits = model(x)
    assert logits.shape == (3, SEQ_LEN, NUM_TOKENS)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("absolute_pos_emb", [False, True])
def test_padding_does_not_affect_real_tokens(absolute_pos_emb):
    model = make_model(absolute_pos_emb=absolute_pos_emb)
    real = 20
    mask = torch.zeros(1, SEQ_LEN, dtype=torch.bool)
    mask[:, :real] = True

    x1 = torch.randint(0, NUM_TOKENS, (1, SEQ_LEN))
    x2 = x1.clone()
    x2[:, real:] = torch.randint(0, NUM_TOKENS, (1, SEQ_LEN - real))

    with torch.no_grad():
        y1 = model(x1, mask=mask)
        y2 = model(x2, mask=mask)

    assert torch.isfinite(y1).all()
    torch.testing.assert_close(y1[:, :real], y2[:, :real])


def test_all_true_mask_matches_no_mask():
    model = make_model()
    x = torch.randint(0, NUM_TOKENS, (2, SEQ_LEN))
    mask = torch.ones(2, SEQ_LEN, dtype=torch.bool)
    with torch.no_grad():
        torch.testing.assert_close(model(x, mask=mask), model(x))


def test_shorter_sequences_and_varying_batch():
    model = make_model()
    for batch, n in [(1, 5), (4, SEQ_LEN), (2, 17)]:
        assert model(torch.randint(0, NUM_TOKENS, (batch, n))).shape == (batch, n, NUM_TOKENS)


def test_sequence_longer_than_max_raises():
    model = make_model()
    with pytest.raises(ValueError, match="max_seq_len"):
        model(torch.randint(0, NUM_TOKENS, (1, SEQ_LEN + 1)))


def test_heads_are_independent():
    """Perturbing one head's slice of the input projection only changes that head's output."""
    torch.manual_seed(0)
    heads, dim_head = 4, 8
    attn = FastAttention(dim=16, heads=heads, dim_head=dim_head).eval()
    x = torch.randn(2, 10, 16)

    def per_head(module):
        captured = {}
        hook = module.to_out.register_forward_hook(lambda m, i, o: captured.setdefault("r", i[0]))
        module(x)
        hook.remove()
        return captured["r"].view(2, 10, heads, dim_head)

    before = per_head(attn)
    with torch.no_grad():
        # rows of to_qkv for the value slice of head 0
        inner = heads * dim_head
        attn.to_qkv.weight[2 * inner : 2 * inner + dim_head] += 1.0
    after = per_head(attn)

    assert not torch.allclose(before[..., 0, :], after[..., 0, :])
    torch.testing.assert_close(before[..., 1:, :], after[..., 1:, :])


def test_gradients_reach_all_parameters():
    model = make_model().train()
    x = torch.randint(0, NUM_TOKENS, (2, SEQ_LEN))
    mask = torch.ones(2, SEQ_LEN, dtype=torch.bool)
    mask[1, 10:] = False
    model(x, mask=mask).sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_attention_logit_weights_shared_across_layers():
    model = make_model(depth=3)
    attns = [layer[1] for layer in model.layers]
    assert all(a.to_q_attn_logits is attns[0].to_q_attn_logits for a in attns)
    assert all(a.to_k_attn_logits is attns[0].to_k_attn_logits for a in attns)

    unshared = make_model(depth=3, share_attn_logit_weights=False)
    attns = [layer[1] for layer in unshared.layers]
    assert attns[1].to_q_attn_logits is not attns[0].to_q_attn_logits


def test_rotary_preserves_relative_inner_products():
    rotary = RotaryEmbedding(8)
    q = torch.randn(8).expand(6, 8)
    k = torch.randn(8).expand(6, 8)
    q_rot, k_rot = rotary(q), rotary(k)
    # <R_m q, R_n k> depends only on m - n
    torch.testing.assert_close(q_rot[3] @ k_rot[1], q_rot[5] @ k_rot[3])
    torch.testing.assert_close(q_rot[0].norm(), q[0].norm())


@pytest.mark.parametrize("rotary", [False, True])
@pytest.mark.parametrize("with_mask", [False, True])
def test_optimized_attention_matches_reference(rotary, with_mask):
    from reference_attention import reference_forward

    torch.manual_seed(0)
    attn = FastAttention(
        dim=32, heads=4, dim_head=8, rotary_emb=RotaryEmbedding(8) if rotary else None
    )
    x = torch.randn(3, 50, 32, dtype=torch.float64)
    attn = attn.double()
    mask = None
    if with_mask:
        mask = torch.ones(3, 50, dtype=torch.bool)
        mask[1, 30:] = False
    x_ref = x.clone().requires_grad_(True)
    x_opt = x.clone().requires_grad_(True)
    out_ref = reference_forward(attn, x_ref, mask)
    out_opt = attn(x_opt, mask=mask)
    torch.testing.assert_close(out_opt, out_ref)

    grad = torch.randn_like(out_ref)
    ref_grads = torch.autograd.grad(out_ref, [x_ref, *attn.parameters()], grad)
    opt_grads = torch.autograd.grad(out_opt, [x_opt, *attn.parameters()], grad)
    for g_opt, g_ref in zip(opt_grads, ref_grads, strict=True):
        torch.testing.assert_close(g_opt, g_ref)


def test_rotary_cache_handles_growing_and_shrinking_lengths():
    rotary = RotaryEmbedding(8)
    short, long = torch.randn(2, 5, 8), torch.randn(2, 40, 8)
    expected_short = rotary(short)
    rotary(long)
    torch.testing.assert_close(rotary(short), expected_short)
