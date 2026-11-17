import pytest
import torch

from olmo_bench.fast_decode import GraphDecoder
from olmo_bench.workload import greedy_generate, make_input_ids

# the tiny model has sliding_window 16
CASES = [(1, 5), (3, 5), (1, 16), (3, 16), (1, 40), (3, 40)]
FLAGS = [dict(additive_mask=True), dict(cached_slots=True)]
ALL_FLAGS = {k: v for f in FLAGS for k, v in f.items()}
FLAG_SETS = FLAGS + [ALL_FLAGS]


def flag_id(flags):
    return "+".join(flags) or "none"


def run_with_logits(model, dec, ids, new):
    logits = []
    hook = model.lm_head.register_forward_hook(lambda m, i, o: logits.append(o[:, -1].clone()))
    try:
        tokens = dec.greedy_generate(ids, new)
    finally:
        hook.remove()
    return tokens, logits


@pytest.mark.parametrize("batch,prompt_len", CASES)
def test_matches_dynamic_cache_and_generate(tiny, batch, prompt_len):
    model, tok = tiny
    ids = make_input_ids(tok, batch, prompt_len, "cpu")
    new = 24
    dec = GraphDecoder(model, batch, prompt_len + new, use_graph=False)
    ours = dec.greedy_generate(ids, new)
    with torch.no_grad():
        ref = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                             max_new_tokens=new, eos_token_id=None)
    assert torch.equal(ours, greedy_generate(model, ids, new))
    assert torch.equal(ours, ref)


@pytest.mark.parametrize("flags", FLAG_SETS, ids=flag_id)
@pytest.mark.parametrize("batch,prompt_len", CASES)
def test_flags_are_bit_exact(tiny, flags, batch, prompt_len):
    model, tok = tiny
    ids = make_input_ids(tok, batch, prompt_len, "cpu")
    new = 24
    base = GraphDecoder(model, batch, prompt_len + new, use_graph=False)
    ref_tokens, ref_logits = run_with_logits(model, base, ids, new)
    dec = GraphDecoder(model, batch, prompt_len + new, use_graph=False, **flags)
    tokens, logits = run_with_logits(model, dec, ids, new)
    assert torch.equal(tokens, ref_tokens)
    assert torch.equal(tokens, greedy_generate(model, ids, new))
    assert len(logits) == len(ref_logits) == new
    assert all(torch.equal(a, b) for a, b in zip(logits, ref_logits))


@pytest.mark.parametrize("flags", [{}] + FLAG_SETS, ids=flag_id)
def test_repeated_prefill_is_fresh(tiny, flags):
    model, tok = tiny
    ids = make_input_ids(tok, 2, 12, "cpu")
    dec = GraphDecoder(model, 2, 40, use_graph=False, **flags)
    first = dec.greedy_generate(ids, 20)
    other = make_input_ids(tok, 2, 30, "cpu")[:, 18:]
    dec.greedy_generate(other, 20)
    assert torch.equal(dec.greedy_generate(ids, 20), first)


def test_max_len_rounded_and_overflow_guarded(tiny):
    model, tok = tiny
    dec = GraphDecoder(model, 1, 65, use_graph=False)
    assert dec.max_len == 128
    with pytest.raises(AssertionError):
        dec.prefill(make_input_ids(tok, 1, 128, "cpu"))


@pytest.mark.parametrize("flags", [{}] + FLAG_SETS, ids=flag_id)
def test_step_has_no_host_syncs(tiny, monkeypatch, flags):
    model, tok = tiny
    dec = GraphDecoder(model, 2, 64, use_graph=False, **flags)
    token = dec.prefill(make_input_ids(tok, 2, 20, "cpu"))
    calls = []
    for name in ["item", "tolist", "__bool__", "__int__", "__float__", "nonzero", "cpu", "numpy"]:
        orig = getattr(torch.Tensor, name)
        monkeypatch.setattr(torch.Tensor, name,
                            lambda self, *a, _n=name, _o=orig, **k: (calls.append(_n), _o(self, *a, **k))[1])
    dec.step(token)
    monkeypatch.undo()
    assert calls == []


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("flags", [{}, ALL_FLAGS], ids=flag_id)
@pytest.mark.parametrize("prompt_len", [6, 16, 40])
def test_graph_matches_eager_static(tiny_dir, prompt_len, flags):
    from olmo_bench.model import load_model
    model, tok = load_model("cuda", torch.bfloat16, str(tiny_dir), None)
    ids = make_input_ids(tok, 3, prompt_len, "cuda")
    new = 30
    eager = GraphDecoder(model, 3, prompt_len + new, use_graph=False).greedy_generate(ids, new)
    dec = GraphDecoder(model, 3, prompt_len + new, **flags)
    dec.prefill(ids)
    dec.capture()
    assert torch.equal(dec.greedy_generate(ids, new), eager)
    # the graph stays valid across fresh prefills
    assert torch.equal(dec.greedy_generate(ids, new), eager)


@cuda
def test_cudnn_attention_graph_matches_eager(tiny_dir):
    from olmo_bench.model import load_model
    model, tok = load_model("cuda", torch.bfloat16, str(tiny_dir), None)
    ids = make_input_ids(tok, 2, 20, "cuda")
    new = 30
    eager = GraphDecoder(model, 2, 20 + new, use_graph=False, cudnn_attention=True).greedy_generate(ids, new)
    dec = GraphDecoder(model, 2, 20 + new, cudnn_attention=True)
    dec.prefill(ids)
    dec.capture()
    assert torch.equal(dec.greedy_generate(ids, new), eager)
