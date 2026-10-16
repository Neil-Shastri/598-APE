import csv
import json
import subprocess
import sys

from olmo_bench.benchmark import DECODE, PREFILL
from olmo_bench.env_info import REPO

STATS = {"n", "median", "min", "max", "iqr", "mean", "cv"}


def test_benchmark_end_to_end(tiny_dir, tmp_path):
    subprocess.run(
        [sys.executable, "-m", "olmo_bench.benchmark", "--device", "cpu", "--dtype", "float32",
         "--model-id", str(tiny_dir), "--revision", "main", "--warmup", "1", "--repeats", "2",
         "--out-root", str(tmp_path), "--run-tag", "t", "--golden-dir", str(tmp_path / "golden")],
        cwd=REPO, check=True, capture_output=True,
    )
    res = json.loads((tmp_path / "t" / "bench.json").read_text())
    meta = res["meta"]
    for key in ["hostname", "slurm_job_id", "gpu_name", "nvidia_smi", "torch", "torch_cuda",
                "transformers", "model_id", "revision", "dtype", "attn_implementation",
                "git_hash", "timestamp", "workload"]:
        assert key in meta
    assert meta["attn_implementation"] == "sdpa"
    assert res["correctness"]["passed"]

    configs = res["configs"]
    assert len(configs) == len(PREFILL) + len(DECODE)
    for c in configs:
        assert "peak_mem_mib" in c
        for name, s in c["metrics"].items():
            assert set(s) == STATS
            assert name == "step_ms" or s["n"] == 2
    dec = [c for c in configs if c["mode"] == "decode"]
    assert set(dec[0]["metrics"]) == {"ttft_ms", "tpot_ms", "decode_tok_s", "e2e_tok_s", "step_ms"}
    assert dec[0]["metrics"]["step_ms"]["n"] == 2 * (dec[0]["new_tokens"] - 1)
    assert all(len(s) == dec[0]["new_tokens"] - 1 for s in dec[0]["raw"]["step_s"])

    rows = list(csv.DictReader(open(tmp_path / "t" / "bench.csv")))
    assert len(rows) == sum(len(c["metrics"]) for c in configs)


def test_benchmark_opt_static(tiny_dir, tmp_path):
    subprocess.run(
        [sys.executable, "-m", "olmo_bench.benchmark", "--device", "cpu", "--dtype", "float32",
         "--model-id", str(tiny_dir), "--revision", "main", "--warmup", "1", "--repeats", "2",
         "--out-root", str(tmp_path), "--run-tag", "t", "--opt", "static"],
        cwd=REPO, check=True, capture_output=True,
    )
    res = json.loads((tmp_path / "t" / "bench.json").read_text())
    assert res["meta"]["opts"] == ["static"]
    check = res["correctness"]
    assert check["passed"]
    assert all(check["generations_match_eager"])
    assert check["workload_tokens_match"] == 1.0
    assert all(t["top1_agreement"] == 1.0 for t in check["texts"])
    assert len(res["configs"]) == len(PREFILL) + len(DECODE)
