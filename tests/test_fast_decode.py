import pytest
import torch

from olmo_bench.fast_decode import GraphDecoder
from olmo_bench.workload import greedy_generate, make_input_ids

# the tiny model has sliding_window 16
CASES = [(1, 5), (3, 5), (1, 16), (3, 16), (1, 40), (3, 40)]


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


def test_repeated_prefill_is_fresh(tiny):
    model, tok = tiny
    ids = make_input_ids(tok, 2, 12, "cpu")
    dec = GraphDecoder(model, 2, 40, use_graph=False)
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


def test_step_has_no_host_syncs(tiny, monkeypatch):
    model, tok = tiny
    dec = GraphDecoder(model, 2, 64, use_graph=False)
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
@pytest.mark.parametrize("prompt_len", [6, 16, 40])
def test_graph_matches_eager_static(tiny_dir, prompt_len):
    from olmo_bench.model import load_model
    model, tok = load_model("cuda", torch.bfloat16, str(tiny_dir), None)
    ids = make_input_ids(tok, 3, prompt_len, "cuda")
    new = 30
    eager = GraphDecoder(model, 3, prompt_len + new, use_graph=False).greedy_generate(ids, new)
    dec = GraphDecoder(model, 3, prompt_len + new)
    dec.prefill(ids)
    dec.capture()
    assert torch.equal(dec.greedy_generate(ids, new), eager)
    # the graph stays valid across fresh prefills
    assert torch.equal(dec.greedy_generate(ids, new), eager)
