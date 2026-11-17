import pytest
import torch
from transformers.models.olmo3 import modeling_olmo3 as olmo3

from olmo_bench import kernels
from olmo_bench.merge import merge_qkv
from olmo_bench.model import load_model


@pytest.fixture
def model(tiny_dir):
    return load_model("cpu", torch.float32, str(tiny_dir), None)[0]


def stored_bytes(model):
    storages = {p.untyped_storage().data_ptr(): p.untyped_storage().nbytes() for p in model.parameters()}
    return sum(storages.values())


def ref_qk_norm_rope(q, k, q_norm, k_norm, cos, sin, head_dim):
    # the HF ops, standing in for the Triton kernel on CPU
    shape = (*q.shape[:-1], -1, head_dim)
    q = q_norm(q).view(shape).transpose(1, 2)
    k = k_norm(k).view(shape).transpose(1, 2)
    return olmo3.apply_rotary_pos_emb(q, k, cos, sin)


def test_merge_qkv_shares_memory(model):
    before = stored_bytes(model)
    merge_qkv(model)
    merged = [layer.self_attn.qkv_proj for layer in model.model.layers]
    merge_qkv(model)
    assert stored_bytes(model) == before
    for layer, qkv in zip(model.model.layers, merged):
        attn = layer.self_attn
        assert attn.qkv_proj is qkv and qkv.bias is None
        ptr = qkv.weight.untyped_storage().data_ptr()
        for lin in (attn.q_proj, attn.k_proj, attn.v_proj):
            assert lin.weight.untyped_storage().data_ptr() == ptr and lin.weight.is_contiguous()
        assert torch.equal(qkv.weight, torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight]))


def test_merge_qkv_keeps_stock_forward(model):
    ids = torch.randint(0, model.config.vocab_size, (2, 40))
    with torch.no_grad():
        ref = model(ids).logits
        merge_qkv(model)
        assert torch.equal(model(ids).logits, ref)


def test_merged_slices_are_not_copied(model):
    merge_qkv(model)
    attn = model.model.layers[0].self_attn
    x = torch.randn(2, 5, model.config.hidden_size)
    with torch.no_grad():
        q, k, v = attn.qkv_proj(x).split([attn.q_proj.out_features] * 3, -1)
        for part, lin in [(q, attn.q_proj), (k, attn.k_proj), (v, attn.v_proj)]:
            rows = kernels._rows(part)
            assert rows.data_ptr() == part.data_ptr() and rows.stride() == (3 * lin.out_features, 1)
            assert torch.equal(part, lin(x))


def test_merged_attention_forward(model, monkeypatch):
    # kernels.attention_forward with the Triton op swapped for the HF ops, so it runs on CPU
    monkeypatch.setattr(kernels, "qk_norm_rope", ref_qk_norm_rope)
    ids = torch.randint(0, model.config.vocab_size, (2, 40))
    with torch.no_grad():
        ref = model(ids).logits
        monkeypatch.setattr(olmo3.Olmo3Attention, "forward", kernels.attention_forward)
        separate = model(ids).logits
        merge_qkv(model)
        merged = model(ids).logits
    assert torch.equal(separate, ref)
    assert torch.equal(merged, ref)
