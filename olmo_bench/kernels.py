"""Fused Triton kernels for the OLMo 3 non-GEMM ops (opt-in).

enable(model) swaps the forwards of the Olmo3 attention, decoder layer, MLP and RMSNorm classes:
  - q_norm + k_norm + RoPE        -> one launch (qk_norm_rope)
  - q/k/v projections             -> one GEMM if merge.merge_qkv(model) was called
  - post-norm + residual add      -> one launch each (rmsnorm with residual), also the final norm
  - silu(gate) * up               -> one launch (silu_mul)
Rounding follows the HF code: fp32 math, bf16 wherever HF produces a bf16 tensor.
"""
import argparse
import time

import torch
from transformers.models.olmo3 import modeling_olmo3 as olmo3

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None


def available():
    return triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _rmsnorm_kernel(x_ptr, w_ptr, res_ptr, out_ptr, eps, N: tl.constexpr, HAS_RES: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, N)
        x = tl.load(x_ptr + row * N + cols).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
        y = (tl.load(w_ptr + cols).to(tl.float32) * (x * rstd)).to(tl.bfloat16)
        if HAS_RES:
            y = (tl.load(res_ptr + row * N + cols).to(tl.float32) + y.to(tl.float32)).to(tl.bfloat16)
        tl.store(out_ptr + row * N + cols, y)

    @triton.jit
    def _norm_rope_row(x_ptr, x_stride, w_ptr, o_ptr, cos_ptr, sin_ptr, row, crow, eps,
                       H: tl.constexpr, D: tl.constexpr):
        HALF: tl.constexpr = D // 2
        d = tl.arange(0, HALF)[None, :]
        lo = tl.arange(0, H)[:, None] * D + d
        hi = lo + HALF
        x1 = tl.load(x_ptr + row * x_stride + lo).to(tl.float32)
        x2 = tl.load(x_ptr + row * x_stride + hi).to(tl.float32)
        rstd = tl.rsqrt((tl.sum(x1 * x1) + tl.sum(x2 * x2)) / (H * D) + eps)
        n1 = (tl.load(w_ptr + lo).to(tl.float32) * (x1 * rstd)).to(tl.bfloat16).to(tl.float32)
        n2 = (tl.load(w_ptr + hi).to(tl.float32) * (x2 * rstd)).to(tl.bfloat16).to(tl.float32)
        c1 = tl.load(cos_ptr + crow + d)
        c2 = tl.load(cos_ptr + crow + HALF + d)
        s1 = tl.load(sin_ptr + crow + d)
        s2 = tl.load(sin_ptr + crow + HALF + d)
        tl.store(o_ptr + row * H * D + lo, (n1 * c1 - n2 * s1).to(tl.bfloat16))
        tl.store(o_ptr + row * H * D + hi, (n2 * c2 + n1 * s2).to(tl.bfloat16))

    @triton.jit(do_not_specialize=["S", "cos_bstride"])
    def _qk_norm_rope_kernel(q_ptr, k_ptr, q_stride, k_stride, qw_ptr, kw_ptr, cos_ptr, sin_ptr, qo_ptr, ko_ptr,
                             S, cos_bstride, eps, H: tl.constexpr, D: tl.constexpr):
        row = tl.program_id(0)
        crow = (row // S) * cos_bstride + (row % S) * D
        if tl.program_id(1) == 0:
            _norm_rope_row(q_ptr, q_stride, qw_ptr, qo_ptr, cos_ptr, sin_ptr, row, crow, eps, H, D)
        else:
            _norm_rope_row(k_ptr, k_stride, kw_ptr, ko_ptr, cos_ptr, sin_ptr, row, crow, eps, H, D)

    @triton.jit(do_not_specialize=["n"])
    def _silu_mul_kernel(g_ptr, u_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # same as torch: x / (1 + expf(-x)) in fp32, rounded to bf16 before the multiply
        act = tl.math.div_rn(g, 1.0 + libdevice.exp(-g)).to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + offs, (act * u).to(tl.bfloat16), mask=mask)


def rmsnorm(x, norm, residual=None):
    # residual + norm(x) when residual is given (OLMo 3 post-norm), else norm(x)
    x = x.contiguous()
    n = x.shape[-1]
    out = torch.empty_like(x)
    res = x if residual is None else residual.contiguous()
    _rmsnorm_kernel[(x.numel() // n,)](x, norm.weight, res, out, norm.variance_epsilon,
                                       N=n, HAS_RES=residual is not None, num_warps=8)
    return out


def _rows(x):
    # [rows, width] view with unit column stride; slices of a merged GEMM output need no copy
    x = x.reshape(-1, x.shape[-1])
    return x if x.stride(-1) == 1 else x.contiguous()


def qk_norm_rope(q, k, q_norm, k_norm, cos, sin, head_dim):
    # q, k: [B, S, H * D] projections (rows may be strided); returns [B, H, S, D] views like the HF code
    cos, sin = cos.contiguous(), sin.contiguous()
    assert cos.dtype == torch.float32 and cos.shape[-1] == head_dim
    b, s, hd = q.shape
    qo = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ko = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    q, k = _rows(q), _rows(k)
    cos_bstride = 0 if cos.shape[0] == 1 else s * head_dim
    _qk_norm_rope_kernel[(b * s, 2)](q, k, q.stride(0), k.stride(0), q_norm.weight, k_norm.weight, cos, sin, qo, ko,
                                     s, cos_bstride, q_norm.variance_epsilon,
                                     H=hd // head_dim, D=head_dim, num_warps=8, enable_fp_fusion=False)
    shape = (b, s, -1, head_dim)
    return qo.view(shape).transpose(1, 2), ko.view(shape).transpose(1, 2)


def silu_mul(gate, up):
    gate, up = gate.contiguous(), up.contiguous()
    out = torch.empty_like(gate)
    n = gate.numel()
    block = 1024
    _silu_mul_kernel[(triton.cdiv(n, block),)](gate, up, out, n, BLOCK=block, num_warps=4)
    return out


# Copies of the HF 5.19 forwards with the norm / RoPE / residual / activation lines replaced.

def attention_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    if hasattr(self, "qkv_proj"):
        sizes = [self.q_proj.out_features, self.k_proj.out_features, self.v_proj.out_features]
        q, k, v = self.qkv_proj(hidden_states).split(sizes, -1)
    else:
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

    cos, sin = position_embeddings
    query_states, key_states = qk_norm_rope(q, k, self.q_norm, self.k_norm, cos, sin, self.head_dim)
    # strided view of the merged output is fine: SDPA and the caches only need stride(-1) == 1
    value_states = v.view(hidden_shape).transpose(1, 2)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    attention_interface = olmo3.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, olmo3.eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling, sliding_window=self.sliding_window, **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def layer_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                  use_cache=False, position_embeddings=None, **kwargs):
    attn_output, _ = self.self_attn(
        hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
        past_key_values=past_key_values, use_cache=use_cache, position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = rmsnorm(attn_output, self.post_attention_layernorm, residual=hidden_states)
    return rmsnorm(self.mlp(hidden_states), self.post_feedforward_layernorm, residual=hidden_states)


def mlp_forward(self, x):
    return self.down_proj(silu_mul(self.gate_proj(x), self.up_proj(x)))


def norm_forward(self, hidden_states):
    return rmsnorm(hidden_states, self)


PATCHES = [
    (olmo3.Olmo3Attention, attention_forward),
    (olmo3.Olmo3DecoderLayer, layer_forward),
    (olmo3.Olmo3MLP, mlp_forward),
    (olmo3.Olmo3RMSNorm, norm_forward),
]
_saved = None


def _patch():
    global _saved
    if _saved is None:
        _saved = [(cls, cls.__dict__["forward"]) for cls, _ in PATCHES]
        for cls, fn in PATCHES:
            cls.forward = fn


def _unpatch():
    global _saved
    if _saved is not None:
        for cls, fn in _saved:
            cls.forward = fn
        _saved = None


def enabled():
    return _saved is not None


def enable(model):
    assert available(), "triton or CUDA is missing"
    assert isinstance(model, olmo3.Olmo3ForCausalLM), type(model)
    assert model.device.type == "cuda" and model.dtype == torch.bfloat16, (model.device, model.dtype)
    assert model.config._attn_implementation == "sdpa", model.config._attn_implementation
    _patch()


def disable(model=None):
    _unpatch()


def warmup(model):
    # compiles every kernel variant once (none of them specialize on the row count)
    cfg = model.config
    layer = model.model.layers[0]
    attn = layer.self_attn
    dev = model.device
    x = torch.randn(1, 3, cfg.hidden_size, device=dev, dtype=torch.bfloat16)
    rmsnorm(x, layer.post_attention_layernorm)
    rmsnorm(x, layer.post_attention_layernorm, residual=x)
    q = torch.randn(1, 3, cfg.num_attention_heads * attn.head_dim, device=dev, dtype=torch.bfloat16)
    cos = torch.randn(1, 3, attn.head_dim, device=dev)
    qk_norm_rope(q, q, attn.q_norm, attn.k_norm, cos, cos, attn.head_dim)
    if hasattr(attn, "qkv_proj"):
        qkv = torch.randn(1, 3, attn.qkv_proj.out_features, device=dev, dtype=torch.bfloat16)
        q, k, _ = qkv.split([attn.q_proj.out_features, attn.k_proj.out_features, attn.v_proj.out_features], -1)
        qk_norm_rope(q, k, attn.q_norm, attn.k_norm, cos, cos, attn.head_dim)
    g = torch.randn(1, 3, cfg.intermediate_size, device=dev, dtype=torch.bfloat16)
    silu_mul(g, g)
    torch.cuda.synchronize()


def _time_us(fn, iters):
    # enqueue = host time per call before the sync; wall = per call including the GPU drain
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return (t1 - t0) / iters * 1e6, (t2 - t0) / iters * 1e6


def bench(iters=1000):
    from transformers import Olmo3Config

    cfg = Olmo3Config(hidden_size=4096, intermediate_size=11008, num_attention_heads=32, num_key_value_heads=32,
                      num_hidden_layers=4, max_position_embeddings=65536, rope_theta=500000.0,
                      rope_scaling={"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 8192,
                                    "attention_factor": 1.2079441541679836, "beta_fast": 32, "beta_slow": 1})
    dev = torch.device("cuda")
    bf = torch.bfloat16
    norm = olmo3.Olmo3RMSNorm(4096, 1e-6).to(dev, bf)
    k_norm = olmo3.Olmo3RMSNorm(4096, 1e-6).to(dev, bf)
    rotary = olmo3.Olmo3RotaryEmbedding(cfg).to(dev)
    print(f"{'op':28s} {'rows':>5s} {'enqueue us':>11s} {'wall us':>9s}")
    for b in (1, 8):
        x = torch.randn(b, 1, 4096, device=dev, dtype=bf)
        res = torch.randn_like(x)
        g = torch.randn(b, 1, 11008, device=dev, dtype=bf)
        u = torch.randn_like(g)
        pos = torch.full((1, 1), 600, device=dev)
        cos, sin = rotary(x, pos, "full_attention")

        def hf_qk_rope():
            q = norm(x).view(b, 1, -1, 128).transpose(1, 2)
            k = k_norm(x).view(b, 1, -1, 128).transpose(1, 2)
            return olmo3.apply_rotary_pos_emb(q, k, cos, sin)

        cases = [
            ("hf  norm + residual", lambda: res + norm(x)),
            ("fused rmsnorm(residual)", lambda: rmsnorm(x, norm, residual=res)),
            ("hf  q/k norm + rope", hf_qk_rope),
            ("fused qk_norm_rope", lambda: qk_norm_rope(x, x, norm, k_norm, cos, sin, 128)),
            ("hf  silu * mul", lambda: torch.nn.functional.silu(g) * u),
            ("fused silu_mul", lambda: silu_mul(g, u)),
        ]
        with torch.no_grad():
            for name, fn in cases:
                enq, wall = _time_us(fn, iters)
                print(f"{name:28s} {b:5d} {enq:11.1f} {wall:9.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bench", action="store_true", help="per-call time of each kernel vs the HF op chain")
    p.add_argument("--iters", type=int, default=1000)
    args = p.parse_args()
    if args.bench:
        bench(args.iters)
