import json

from olmo_bench.correctness import NEW_TOKENS, PROMPTS, run_checks

META = {"dtype": "float32", "attn_implementation": "sdpa"}


def test_checks_pass_and_golden_roundtrip(tiny, tmp_path):
    model, tok = tiny
    first = run_checks(model, tok, "cpu", tmp_path, META)
    assert first["golden"]["status"] == "created"
    assert all(g["matches_generate"] for g in first["generations"])
    assert len(first["generations"]) == len(PROMPTS)
    assert first["perplexity"]["ppl"] > 1
    assert first["passed"]

    second = run_checks(model, tok, "cpu", tmp_path, META)
    assert second["golden"] == {**first["golden"], "status": "compared", "match": True}
    assert second["passed"]


def test_golden_mismatch_fails(tiny, tmp_path):
    model, tok = tiny
    res = run_checks(model, tok, "cpu", tmp_path, META)
    path = tmp_path / "cpu_float32_sdpa.json"
    golden = json.loads(path.read_text())
    ids = golden["generations"][0]["new_token_ids"]
    assert len(ids) == NEW_TOKENS
    ids[-1] = (ids[-1] + 1) % len(tok)
    path.write_text(json.dumps(golden))

    res = run_checks(model, tok, "cpu", tmp_path, META)
    assert res["golden"]["match"] is False
    assert not res["passed"]
