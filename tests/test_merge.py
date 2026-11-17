import pytest
import torch
from transformers.models.olmo3 import modeling_olmo3 as olmo3

from olmo_bench import kernels
from olmo_bench.merge import merge_gateup, merge_qkv
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


def ref_silu_mul(gate, up):
    return torch.nn.functional.silu(gate) * up


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


def test_merge_keeps_stock_forward(model):
    ids = torch.randint(0, model.config.vocab_size, (2, 40))
    with torch.no_grad():
        ref = model(ids).logits
        merge_qkv(model)
        merge_gateup(model)
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


def test_merged_forwards(model, monkeypatch):
    # kernels.attention_forward / mlp_forward with the Triton ops swapped for the HF ops, so they run on CPU
    monkeypatch.setattr(kernels, "qk_norm_rope", ref_qk_norm_rope)
    monkeypatch.setattr(kernels, "silu_mul", ref_silu_mul)
    ids = torch.randint(0, model.config.vocab_size, (2, 40))
    with torch.no_grad():
        ref = model(ids).logits
        monkeypatch.setattr(olmo3.Olmo3Attention, "forward", kernels.attention_forward)
        monkeypatch.setattr(olmo3.Olmo3MLP, "forward", kernels.mlp_forward)
        separate = model(ids).logits
        merge_qkv(model)
        merge_qkv_logits = model(ids).logits
        merge_gateup(model)
        merged = model(ids).logits
    assert torch.equal(separate, ref)
    assert torch.equal(merge_qkv_logits, ref)
    assert torch.equal(merged, ref)


def test_merge_gateup_shares_memory(model):
    before = stored_bytes(model)
    merge_gateup(model)
    merged = [layer.mlp.gate_up_proj for layer in model.model.layers]
    merge_gateup(model)
    assert stored_bytes(model) == before
    x = torch.randn(2, 5, model.config.hidden_size)
    for layer, gate_up in zip(model.model.layers, merged):
        mlp = layer.mlp
        assert mlp.gate_up_proj is gate_up and gate_up.bias is None
        ptr = gate_up.weight.untyped_storage().data_ptr()
        for lin in (mlp.gate_proj, mlp.up_proj):
            assert lin.weight.untyped_storage().data_ptr() == ptr and lin.weight.is_contiguous()
        with torch.no_grad():
            gate, up = gate_up(x).chunk(2, -1)
            assert torch.equal(gate, mlp.gate_proj(x)) and torch.equal(up, mlp.up_proj(x))
            rows = kernels._rows(up)
            assert rows.data_ptr() == up.data_ptr() and rows.stride() == (2 * mlp.intermediate_size, 1)
