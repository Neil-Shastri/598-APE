import pytest
import torch
import torch.nn.functional as F
from transformers import Olmo3Config
from transformers.models.olmo3 import modeling_olmo3 as olmo3

from olmo_bench import kernels

pytestmark = pytest.mark.skipif(not kernels.available(), reason="needs triton and CUDA")

DEV = "cuda"
BF = torch.bfloat16


def ulp(x):
    # spacing of bf16 numbers around |x| (8 significant bits)
    a = x.float().abs().clamp_min(2.0**-126)
    return torch.exp2(torch.floor(torch.log2(a)) - 7)


def check(out, ref, tol):
    # every element within tol, and nearly all of them bit-identical
    assert out.shape == ref.shape and out.dtype == ref.dtype
    diff = (out.float() - ref.float()).abs()
    assert (diff <= tol).all(), f"max diff {diff.max().item()}, worst excess {(diff - tol).max().item()}"
    assert (diff > 0).float().mean().item() < 1e-3


def make_norm(n, seed=0):
    torch.manual_seed(seed)
    norm = olmo3.Olmo3RMSNorm(n, 1e-6).to(DEV, BF)
    norm.weight.data = (torch.rand(n, device=DEV) + 0.5).to(BF)
    return norm


def real_rotary():
    cfg = Olmo3Config(hidden_size=4096, num_attention_heads=32, num_key_value_heads=32, num_hidden_layers=4,
                      max_position_embeddings=65536, rope_theta=500000.0,
                      rope_scaling={"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 8192,
                                    "attention_factor": 1.2079441541679836, "beta_fast": 32, "beta_slow": 1})
    return olmo3.Olmo3RotaryEmbedding(cfg).to(DEV)


SHAPES = [(1, 1), (8, 1), (1, 7), (1, 512), (1, 2048), (8, 64)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("with_residual", [False, True])
def test_rmsnorm(shape, with_residual):
    norm = make_norm(4096)
    x = torch.randn(*shape, 4096, device=DEV, dtype=BF) * 3
    with torch.no_grad():
        normed = norm(x)
        if with_residual:
            res = torch.randn_like(x)
            ref, out = res + normed, kernels.rmsnorm(x, norm, residual=res)
            # a 1-ulp difference in the normed value carries into the sum
            tol = ulp(ref) + ulp(normed)
        else:
            ref, out = normed, kernels.rmsnorm(x, norm)
            tol = ulp(ref)
    check(out, ref, tol)


def test_rmsnorm_non_contiguous():
    norm = make_norm(4096)
    x = torch.randn(1, 7, 8192, device=DEV, dtype=BF)[..., ::2]
    with torch.no_grad():
        check(kernels.rmsnorm(x, norm), norm(x), ulp(norm(x)))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
@pytest.mark.parametrize("per_row_positions", [False, True])
def test_qk_norm_rope(shape, layer_type, per_row_positions):
    b, s = shape
    q_norm, k_norm = make_norm(4096, 1), make_norm(4096, 2)
    q = torch.randn(b, s, 4096, device=DEV, dtype=BF) * 2
    k = torch.randn(b, s, 4096, device=DEV, dtype=BF)
    # model.generate passes [B, S] position_ids (cos batch B); plain forward gives [1, S]
    start = torch.arange(b, device=DEV)[:, None] * 37 + 500 if per_row_positions else torch.tensor([[500]], device=DEV)
    pos = start + torch.arange(s, device=DEV)
    cos, sin = real_rotary()(q, pos, layer_type)
    with torch.no_grad():
        nq = q_norm(q).view(b, s, -1, 128).transpose(1, 2)
        nk = k_norm(k).view(b, s, -1, 128).transpose(1, 2)
        ref_q, ref_k = olmo3.apply_rotary_pos_emb(nq, nk, cos, sin)
        out_q, out_k = kernels.qk_norm_rope(q, k, q_norm, k_norm, cos, sin, 128)
    # same memory order as HF; strides of size-1 dims don't matter
    assert [st for st, n in zip(out_q.stride(), out_q.shape) if n > 1] == \
        [st for st, n in zip(ref_q.stride(), ref_q.shape) if n > 1]
    # a 1-ulp difference in a normed value is scaled by |cos| or |sin| (at most the YaRN factor 1.21)
    for out, ref, n in [(out_q, ref_q, nq), (out_k, ref_k, nk)]:
        tol = ulp(ref) + 1.25 * (ulp(n) + ulp(olmo3.rotate_half(n)))
        check(out, ref, tol)


@pytest.mark.parametrize("rows", [1, 7, 512, 2048])
def test_silu_mul(rows):
    g = torch.randn(rows, 11008, device=DEV, dtype=BF) * 4
    u = torch.randn_like(g)
    ref = F.silu(g) * u
    check(kernels.silu_mul(g, u), ref, ulp(ref))


@pytest.fixture
def tiny_cuda(tiny_dir):
    from olmo_bench.model import load_model
    model, tok = load_model(DEV, BF, str(tiny_dir), None)
    yield model, tok
    kernels.disable(model)


def test_enable_disable_end_to_end(tiny_cuda):
    from olmo_bench.workload import greedy_generate, make_input_ids

    model, tok = tiny_cuda
    ids = make_input_ids(tok, 2, 40, DEV)
    forwards = {cls: cls.forward for cls, _ in kernels.PATCHES}
    with torch.no_grad():
        ref = model(ids).logits
        ref_gen = greedy_generate(model, ids, 12)
        kernels.enable(model)
        kernels.enable(model)
        kernels.warmup(model)
        out = model(ids).logits
        gen = greedy_generate(model, ids, 12)
        hf_gen = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=12)
        kernels.disable(model)
        kernels.disable(model)
        again = model(ids).logits
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.02, atol=0.02 * ref.float().abs().max().item())
    assert torch.equal(gen, ref_gen)
    assert torch.equal(gen[:, :hf_gen.shape[1]], hf_gen)
    assert torch.equal(again, ref)
    assert all(cls.forward is fn for cls, fn in forwards.items())


def test_enable_rejects_eager_attention(tiny_dir):
    from olmo_bench.model import load_model
    model, _ = load_model(DEV, BF, str(tiny_dir), None, attn_implementation="eager")
    with pytest.raises(AssertionError):
        kernels.enable(model)
    assert not kernels.enabled()
