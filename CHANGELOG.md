# Changelog

## 0.3.0

### Performance
- `FastAttention` no longer materializes the modulated keys and values; the global
  query and key are folded into the linear maps that consume them. Same outputs and
  gradients, 22-40% less activation memory at long sequence lengths.
- Rotary embeddings cache half-width cos/sin tables and rotate q, k and v in one call
  (about 1.7x faster attention with rotary embeddings on Apple MPS at 16k tokens).
- Both pooling steps run through `F.scaled_dot_product_attention` as single-query
  attention (fused kernels on CUDA); `FastAttention(use_sdpa=False)` keeps the einsum path.
  Up to ~10% faster on CPU at long sequence lengths, unchanged on Apple MPS.
- Added `benchmarks/bench_attention.py`.

## 0.2.0

Rewrite. Version 0.1.0 could not be imported and produced NaN outputs.

### Breaking
- `mask` moved from the `FastTransformer` constructor to `forward(x, mask=None)`.
  It is now optional, and `True` marks real tokens.
- Requires Python ≥ 3.10 and PyTorch ≥ 2.0.
- Dropped the `rotary-embedding-torch` and `numpy` dependencies; rotary embeddings are built in.

### Fixed
- Package import failed (`from fast_attention import ...` was not relative).
- Inverted mask filled every attention logit with `-inf`, so all outputs were NaN.
- Reshape into heads mixed sequence positions with heads.
- Position indices were always created on the CPU, which broke CUDA and MPS.
- Attention-logit weight sharing also attached unused modules to feed-forward blocks.
- Package metadata listed the license as MIT; the project is Apache 2.0.

### Added
- `share_attn_logit_weights` option.
- `max_seq_len` check in `forward`.
- Test suite and ruff linting, with GitHub Actions workflows for tests (Python 3.10–3.13) and lint.
- PyPI trusted-publishing release workflow.
