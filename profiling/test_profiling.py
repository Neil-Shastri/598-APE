import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.profiler import ProfilerActivity, profile
from transformers import Olmo3Config, Olmo3ForCausalLM, PreTrainedTokenizerFast

from olmo_bench.workload import TEXT_FILE, decode_step, prefill
from profiling import ncu_table, nsys_gaps
from profiling.labels import Labels, category, region
from profiling.trace_analysis import (category_rows, folded_stacks, gpu_kernels, load_events, regions,
                                      step_rows)

REPO = Path(__file__).resolve().parent.parent
LAYERS = 4


def tiny_model():
    torch.manual_seed(0)
    cfg = Olmo3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=LAYERS,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096, sliding_window=16,
        pad_token_id=None, eos_token_id=1,
        rope_scaling={"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 512},
    )
    return Olmo3ForCausalLM(cfg).eval()


def generate(model, ids, steps):
    token, cache = prefill(model, ids)
    out = [token]
    for _ in range(steps):
        token, cache = decode_step(model, token, cache)
        out.append(token)
    return torch.cat(out, 1)


def test_labels_keep_numerics_and_restore():
    model = tiny_model()
    ids = torch.randint(0, 128, (2, 24))
    ref = generate(model, ids, 6)
    labels = Labels(model)
    with labels:
        labels.phase = "decode_b2"
        assert torch.equal(generate(model, ids, 6), ref)
        assert "forward" in model.model.layers[0].mlp.act_fn.__dict__
    assert all("forward" not in m.__dict__ for m in model.modules())
    from transformers import cache_utils
    from transformers.models.olmo3 import modeling_olmo3
    assert not hasattr(modeling_olmo3.apply_rotary_pos_emb, "__wrapped__")
    assert not hasattr(cache_utils.Cache.update, "__wrapped__")
    assert modeling_olmo3.ALL_ATTENTION_FUNCTIONS._local_mapping == {}


def test_category():
    assert category("decode_b1:layer") == "residual"
    assert category("decode_b1:mlp") == "mlp_mul"
    assert category("prefill_b8:qk_norm") == "qk_norm"
    assert category("decode_b1") == "other"
    assert category(None) == "unlabeled"
    assert category("something_else") == "unlabeled"


@pytest.fixture(scope="module")
def fake_gpu_trace(tmp_path_factory):
    """A real CPU trace of prefill + 3 decode steps with one fake 1 us kernel launched from every
    leaf aten op, shaped like Kineto's CUDA trace (cuda_runtime launch + kernel, linked by correlation)."""
    model = tiny_model()
    ids = torch.randint(0, 128, (1, 20))
    path = tmp_path_factory.mktemp("trace") / "t.json"
    with torch.no_grad(), Labels(model) as labels, profile(activities=[ProfilerActivity.CPU]) as prof:
        labels.phase = "prefill_b1"
        with region("prefill_b1", False):
            token, cache = prefill(model, ids)
        labels.phase = "decode_b1"
        for _ in range(3):
            with region("decode_b1", False):
                token, cache = decode_step(model, token, cache)
    prof.export_chrome_trace(str(path))
    trace = json.loads(path.read_text())
    ops = sorted((e for e in trace["traceEvents"] if e.get("cat") in ("cpu_op", "user_annotation")),
                 key=lambda e: (e["ts"], -e["dur"]))
    stack, leaves = [], []
    for e in ops:
        while stack and stack[-1]["ts"] + stack[-1]["dur"] <= e["ts"]:
            stack.pop()
        if stack and stack[-1]["cat"] == "cpu_op" and stack[-1] in leaves:
            leaves.remove(stack[-1])
        if e["cat"] == "cpu_op":
            leaves.append(e)
        stack.append(e)
    gpu_t = 0.0
    for i, op in enumerate(sorted(leaves, key=lambda e: e["ts"])):
        t = op["ts"] + op["dur"] / 2
        gpu_t = max(gpu_t, t) + 0.5
        trace["traceEvents"].append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": op["pid"],
                                     "tid": op["tid"], "ts": t, "dur": 0.0, "args": {"correlation": 1000 + i}})
        trace["traceEvents"].append({"ph": "X", "cat": "kernel", "name": f"k_{op['name']}", "pid": 0, "tid": 7,
                                     "ts": gpu_t, "dur": 1.0, "args": {"correlation": 1000 + i}})
        gpu_t += 1.0
    path.write_text(json.dumps(trace))
    return path, len(leaves)


def test_trace_analysis(fake_gpu_trace):
    path, n_leaves = fake_gpu_trace
    events = load_events(path)
    kernels = gpu_kernels(events)
    regs = regions(events)
    assert len(kernels) == n_leaves
    assert [r["name"] for r in regs] == ["prefill_b1"] + ["decode_b1"] * 3

    cats = Counter((k["region"]["name"], k["category"], k["name"]) for k in kernels)
    # the residual adds are the only ops left directly in the decoder layer
    assert {n for (_, c, n) in cats if c == "residual"} == {"k_aten::add"}
    assert sum(v for (r, c, _), v in cats.items() if r == "decode_b1" and c == "residual") == 2 * LAYERS * 3
    assert {n for (_, c, n) in cats if c == "mlp_mul"} == {"k_aten::mul"}
    assert {n for (_, c, n) in cats if c == "act"} == {"k_aten::silu"}
    assert "unlabeled" not in {c for (_, c, _) in cats}

    rows = category_rows(kernels, regs)
    dec = {r["category"]: r for r in rows if r["region"] == "decode_b1"}
    assert dec["total"]["kernels"] == sum(r["kernels"] for c, r in dec.items() if c != "total")
    assert dec["residual"]["kernels_per_step"] == 2 * LAYERS
    assert abs(sum(r["share_pct"] for c, r in dec.items() if c != "total") - 100) < 0.1
    for c in ("rmsnorm", "qk_norm", "rope", "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head",
              "attn_core", "kv_cache", "final_norm", "embed"):
        assert c in dec, c

    steps = step_rows(kernels, regs)
    assert [s["region"] for s in steps] == ["prefill_b1"] + ["decode_b1"] * 3
    for s in steps:
        assert s["gpu_busy_us"] == pytest.approx(s["kernels"] * 1.0, abs=0.2)
        assert s["gpu_busy_us"] <= s["gpu_span_us"] + 0.1

    lines = folded_stacks(kernels)
    assert all(line.rsplit(" ", 1)[1].isdigit() for line in lines)
    assert any(line.startswith("decode_b1;model;layer;attn;qk_norm;") for line in lines)
    assert sum(int(line.rsplit(" ", 1)[1]) for line in lines) == n_leaves * 1000


def tiny_checkpoint(path):
    words = sorted(set(TEXT_FILE.read_text().split()))
    vocab = {"[UNK]": 0, "<|endoftext|>": 1} | {w: i + 2 for i, w in enumerate(words)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", eos_token="<|endoftext|>")
    model = tiny_model()
    model.resize_token_embeddings(len(tok))
    model.save_pretrained(path)
    tok.save_pretrained(path)


def test_run_torch_tool_on_cpu(tmp_path):
    tiny_checkpoint(tmp_path / "ckpt")
    cmd = [sys.executable, "-m", "profiling.run", "--tool", "torch", "--device", "cpu", "--dtype", "float32",
           "--model-id", str(tmp_path / "ckpt"), "--revision", "main", "--seq-len", "32", "--batches", "1,2",
           "--decode-steps", "4", "--warmup", "1", "--out", str(tmp_path / "out"), "--raw", str(tmp_path / "raw")]
    subprocess.run(cmd, cwd=REPO, check=True)
    out = tmp_path / "out"
    for name in ("prefill_b1", "decode_b1", "prefill_b2", "decode_b2"):
        assert (tmp_path / "raw" / f"{name}.json.gz").exists()
        assert "Self CPU" in (out / f"{name}_ops_self.txt").read_text()
        assert (out / f"{name}_ops_total.txt").exists()
    rows = list(csv.DictReader(open(out / "label_overhead.csv")))
    assert [r["batch"] for r in rows] == ["1", "2"]


NCU_HEADER = ["ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context", "Stream", "Block Size",
              "Grid Size", "Device", "CC", "dram__bytes_read.sum", "dram__bytes_write.sum", "gpu__time_duration.sum",
              "lts__t_sector_hit_rate.pct", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
              "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
              "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
              "sm__warps_active.avg.pct_of_peak_sustained_active", "launch__grid_size", "launch__block_size",
              "launch__registers_per_thread"]
NCU_UNITS = [""] * 11 + ["byte", "byte", "nsecond", "%", "%", "%", "%", "%", "", "", "register/thread"]


def write_ncu(path, names):
    rows = [["0", "1", "python", "h", names[0], "1", "7", "(256, 1, 1)", "(32, 1, 1)", "0", "8.6",
             "1,048,576", "524,288", "4000", "12.5", "3.1", "80.2", "79.9", "60.0", "32", "256", "16"],
            ["1", "1", "python", "h", names[1], "1", "7", "(128, 1, 1)", "(4096, 1, 1)", "0", "8.6",
             "8388608", "0", "16000", "1.0", "40.0", "90.0", "90.0", "30.5", "4096", "128", "90"]]
    with open(path, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(NCU_HEADER)
        w.writerow(NCU_UNITS)
        w.writerows(rows)


def test_ncu_table(tmp_path):
    write_ncu(tmp_path / "raw.csv", ["void at::native::reduce_kernel<...>(...)", "ampere_bf16_gemv"])
    write_ncu(tmp_path / "nvtx.csv", ["decode_b1:qk_norm/void at::native::reduce_kernel<...>(...)",
                                      "prefill_b1:gate_up_proj/ampere_bf16_gemv"])
    rows = ncu_table.kernel_rows(tmp_path / "raw.csv", tmp_path / "nvtx.csv")
    a, b = rows
    assert (a["region"], a["category"], a["group"]) == ("decode_b1", "qk_norm", "nongemm")
    assert a["duration_us"] == 4.0
    assert a["dram_mb"] == pytest.approx(1.572864, abs=1e-3)
    assert a["bw_gbs"] == pytest.approx(1572864 / 4000, rel=1e-3)
    assert a["pct_of_578"] == pytest.approx(100 * 1572864 / 4000 / 578, rel=1e-3)
    assert (a["l2_hit_pct"], a["occupancy_pct"], a["regs"]) == (12.5, 60.0, 16.0)
    assert b["category"] == "gate_up_proj" and b["bw_gbs"] == pytest.approx(524.288, rel=1e-3)
    assert a["kernel"] == "reduce_kernel<...>(...)"
    summary = ncu_table.rollup(rows, "category")
    assert [(s["region"], s["category"], s["kernels"], s["share_pct"]) for s in summary] == [
        ("decode_b1", "qk_norm", 1, 100), ("prefill_b1", "gate_up_proj", 1, 100)]
    assert [s["group"] for s in ncu_table.rollup(rows, "group")] == ["nongemm", "gemm"]


def test_nsys_gaps(tmp_path):
    kern = tmp_path / "k.csv"
    with open(kern, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Start (ns)", "Duration (ns)", "CorrId", "Name"])
        # step 0: kernels at 1000-2000, 3000-4000 (one 1 us gap); step 1: 10000-12000
        for s, d in [(1000, 1000), (3000, 1000), (10000, 2000), (50000, 10)]:
            w.writerow([s, d, 1, "k"])
    nvtx = tmp_path / "n.csv"
    with open(nvtx, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Name", "Projected Start (ns)", "Projected Duration (ns)", "Orig Start (ns)",
                    "Orig Duration (ns)", "Style", "PID", "TID", "NumGPUOps", "Lvl", "NumChild", "RangeId",
                    "ParentId", "RangeStack"])
        # nsys prefixes the (empty) default domain: ":decode_b1"
        w.writerow([":decode_b1", 1000, 3000, 500, 5000, "", 1, 1, 2, 0, 0, 1, "", ":1"])
        w.writerow([":decode_b1:rmsnorm", 1000, 1000, 600, 100, "", 1, 1, 1, 1, 0, 2, 1, ":1:2"])
        w.writerow([":decode_b1", 10000, 2000, 9000, 4000, "", 1, 1, 1, 0, 0, 3, "", ":3"])
    rows = nsys_gaps.step_rows(kern, nvtx)
    assert [(r["index"], r["kernels"], r["gpu_busy_us"], r["gpu_idle_us"], r["max_gap_us"]) for r in rows] == [
        (0, 2, 2.0, 1.0, 1.0), (1, 1, 2.0, 0.0, 0)]
    assert rows[0]["busy_pct_of_wall"] == 40.0
    assert {r["region"] for r in rows} == {"decode_b1"}
    summary = nsys_gaps.summary_rows(rows)
    assert summary[0]["n"] == 2 and summary[0]["median_gpu_busy_us"] == 2.0
