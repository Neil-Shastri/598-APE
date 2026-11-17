import pytest
import torch
from transformers.models.olmo3 import modeling_olmo3 as olmo3

from olmo_bench import band_attention
from olmo_bench.fast_decode import GraphDecoder
from olmo_bench.workload import greedy_generate, make_input_ids

# the tiny model has sliding_window 16 and 3 sliding + 1 full layer


@pytest.fixture
def band(tiny, monkeypatch):
    # enables band attention and counts banded calls
    model, _ = tiny
    calls = []
    orig = band_attention.banded_sdpa
    monkeypatch.setattr(band_attention, "banded_sdpa", lambda *a: (calls.append(a[0].shape), orig(*a))[1])

    def on(chunk=1024):
        band_attention.enable(model, chunk)
        return calls
    yield on
    band_attention.disable(model)


def logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("prompt_len,chunk", [(40, 1024), (60, 1024), (60, 7), (17, 1024)])
def test_logits_close(tiny, band, batch, prompt_len, chunk):
    model, tok = tiny
    ids = make_input_ids(tok, batch, prompt_len, "cpu")
    ref = logits(model, ids)
    calls = band(chunk)
    out = logits(model, ids)
    assert len(calls) == 3
    diff = (out - ref).abs().max().item()
    print(f"b{batch} len{prompt_len} chunk{chunk} max|dlogit| {diff:.2e}")
    assert diff < 1e-4
    assert torch.equal(out.argmax(-1), ref.argmax(-1))


@pytest.mark.parametrize("batch", [1, 2])
def test_greedy_tokens_equal(tiny, band, batch):
    model, tok = tiny
    ids = make_input_ids(tok, batch, 50, "cpu")
    ref = greedy_generate(model, ids, 20)
    calls = band()
    assert torch.equal(greedy_generate(model, ids, 20), ref)
    assert len(calls) == 3  # prefill only, decode passes through


@pytest.mark.parametrize("batch", [1, 2])
def test_static_path(tiny, band, batch):
    model, tok = tiny
    ids = make_input_ids(tok, batch, 45, "cpu")
    ref = GraphDecoder(model, batch, 45 + 20, use_graph=False).greedy_generate(ids, 20)
    calls = band()
    out = GraphDecoder(model, batch, 45 + 20, use_graph=False).greedy_generate(ids, 20)
    assert torch.equal(out, ref)
    assert len(calls) == 3


@pytest.mark.parametrize("prompt_len", [5, 16])
def test_short_prompt_unchanged(tiny, band, prompt_len):
    model, tok = tiny
    ids = make_input_ids(tok, 2, prompt_len, "cpu")
    ref = logits(model, ids)
    calls = band()
    assert torch.equal(logits(model, ids), ref)
    assert calls == []


def test_disable_restores(tiny, band):
    model, tok = tiny
    fns = olmo3.ALL_ATTENTION_FUNCTIONS
    before = dict(fns._local_mapping), fns["sdpa"]
    ids = make_input_ids(tok, 2, 40, "cpu")
    ref = logits(model, ids)
    band()
    assert fns["sdpa"] is not before[1]
    band_attention.disable(model)
    assert (dict(fns._local_mapping), fns["sdpa"]) == before
    assert not band_attention.enabled()
    assert torch.equal(logits(model, ids), ref)


GOOD_BAND = band_attention.band_mask


def wider(w, c, dtype, device):
    mask = GOOD_BAND(w, c, dtype, device)
    mask[torch.arange(c), torch.arange(c)] = 0  # also sees key i - window
    return mask


def narrower(w, c, dtype, device):
    mask = GOOD_BAND(w, c, dtype, device)
    mask[torch.arange(c), torch.arange(c) + 1] = float("-inf")  # drops the oldest key
    return mask


@pytest.mark.parametrize("bad", [wider, narrower])
def test_off_by_one_is_caught(tiny, monkeypatch, bad):
    model, tok = tiny
    ids = make_input_ids(tok, 1, 40, "cpu")
    ref = logits(model, ids)
    monkeypatch.setattr(band_attention, "band_mask", bad)
    band_attention.enable(model)
    try:
        assert (logits(model, ids) - ref).abs().max() > 1e-3
    finally:
        band_attention.disable(model)
