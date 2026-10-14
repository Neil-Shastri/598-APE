import torch

from olmo_bench.workload import decode_step, greedy_generate, make_input_ids, prefill, text_token_ids


def test_make_input_ids_shape_and_determinism(tiny):
    _, tok = tiny
    a = make_input_ids(tok, 3, 50, "cpu")
    b = make_input_ids(tok, 3, 50, "cpu")
    assert a.shape == (3, 50) and a.dtype == torch.long
    assert torch.equal(a, b)
    assert not torch.equal(a[0], a[1])
    assert a[0].tolist() == text_token_ids(tok)[:50]


def test_make_input_ids_tiles_short_text(tiny):
    _, tok = tiny
    n = len(text_token_ids(tok))
    ids = make_input_ids(tok, 1, n + 10, "cpu")
    assert ids.shape == (1, n + 10)
    assert torch.equal(ids[0, n:], ids[0, :10])


def test_prefill_and_decode_shapes(tiny):
    model, tok = tiny
    ids = make_input_ids(tok, 2, 24, "cpu")
    token, cache = prefill(model, ids)
    assert token.shape == (2, 1) and token.dtype == torch.long
    assert cache.get_seq_length() == 24
    token, cache = decode_step(model, token, cache)
    assert token.shape == (2, 1)
    assert cache.get_seq_length() == 25


def test_greedy_generate_matches_hf_generate(tiny):
    model, tok = tiny
    # 40-token prompt + 20 new tokens crosses the 16-token sliding window
    ids = make_input_ids(tok, 2, 40, "cpu")
    ours = greedy_generate(model, ids, 20)
    with torch.no_grad():
        ref = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                             max_new_tokens=20, eos_token_id=None)
    assert ours.shape == (2, 60)
    assert torch.equal(ours, ref)


def test_cached_decode_matches_full_forward(tiny):
    model, tok = tiny
    ids = make_input_ids(tok, 1, 30, "cpu")
    out = greedy_generate(model, ids, 10)
    with torch.no_grad():
        logits = model(input_ids=out[:, :-1]).logits
    assert torch.equal(logits[:, 29:].argmax(-1), out[:, 30:])
