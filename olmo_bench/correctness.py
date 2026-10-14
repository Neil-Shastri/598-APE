import argparse
import json
from pathlib import Path

import torch

from olmo_bench import env_info
from olmo_bench.model import MODEL_ID, REVISION, load_model
from olmo_bench.workload import LONG_TEXT_FILE, TEXT_FILE, greedy_generate, text_token_ids

PROMPTS = [
    "The capital of France is",
    "Water boils at sea level when it reaches a temperature of",
    "Once upon a time, in a small village by the sea, there lived",
    "The theory of general relativity, proposed by Albert Einstein,",
]
NEW_TOKENS = 32
PPL_TOKENS = 1536
LONG_PPL_TOKENS = 8192
PPL_FLAG = 20.0
GOLDEN_DIR = env_info.REPO / "results" / "baseline" / "golden"


def generate_and_compare(model, tokenizer, device):
    out = []
    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        ours = greedy_generate(model, ids, NEW_TOKENS)
        with torch.no_grad():
            ref = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                 max_new_tokens=NEW_TOKENS)
        # generate stops at EOS, our loop does not, so compare over generate's length
        n = ref.shape[1]
        new = ours[0, ids.shape[1]:].tolist()
        out.append({
            "prompt": prompt,
            "new_token_ids": new,
            "text": tokenizer.decode(new),
            "matches_generate": torch.equal(ours[:, :n], ref),
            "generate_new_tokens": n - ids.shape[1],
        })
    return out


def perplexity(model, tokenizer, device, text_file=TEXT_FILE, n_tokens=PPL_TOKENS):
    ids = torch.tensor([text_token_ids(tokenizer, text_file)[:n_tokens]], device=device)
    with torch.no_grad():
        loss = model(input_ids=ids, labels=ids).loss
    return {"tokens": ids.shape[1], "ppl": torch.exp(loss.float()).item()}


def compare_golden(gens, path, meta):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"meta": meta, "generations": gens}, indent=2))
        return {"golden": str(path), "status": "created", "match": None}
    golden = json.loads(path.read_text())
    same = [g["new_token_ids"] for g in golden["generations"]] == [g["new_token_ids"] for g in gens]
    return {"golden": str(path), "status": "compared", "match": same}


def run_checks(model, tokenizer, device, golden_dir, meta):
    gens = generate_and_compare(model, tokenizer, device)
    ppl = perplexity(model, tokenizer, device)
    # long enough that the sliding-window layers only see part of the context
    ppl_long = perplexity(model, tokenizer, device, LONG_TEXT_FILE, LONG_PPL_TOKENS)
    # bf16 greedy tokens can legitimately differ across GPU models, so key the golden file on the GPU
    hw = (meta.get("gpu_name") or torch.device(device).type).replace(" ", "_")
    name = f"{hw}_{meta['dtype']}_{meta['attn_implementation']}.json"
    golden = compare_golden(gens, Path(golden_dir) / name, meta | {"ppl": ppl["ppl"]})
    passed = all(g["matches_generate"] for g in gens) and golden["match"] is not False
    return {"passed": passed, "ppl_flagged": max(ppl["ppl"], ppl_long["ppl"]) > PPL_FLAG,
            "perplexity": ppl, "perplexity_long": ppl_long,
            "golden": golden, "generations": gens}


def report(res):
    for g in res["generations"]:
        print(f"--- {g['prompt']!r}  matches_generate={g['matches_generate']}")
        print(g["prompt"] + g["text"])
    ppl = res["perplexity"]
    flag = "  <-- SUSPICIOUS" if res["ppl_flagged"] else ""
    print(f"perplexity over {ppl['tokens']} tokens: {ppl['ppl']:.3f}{flag}")
    long = res["perplexity_long"]
    print(f"long-context perplexity over {long['tokens']} tokens: {long['ppl']:.3f}")
    print(f"golden {res['golden']['status']}: {res['golden']['golden']} match={res['golden']['match']}")
    print("CORRECTNESS", "PASS" if res["passed"] else "FAIL")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn", default=None)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--golden-dir", default=str(GOLDEN_DIR))
    p.add_argument("--out", default=None, help="optional path for the result json")
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    model, tokenizer = load_model(args.device, dtype, args.model_id, args.revision, args.attn)
    meta = env_info.collect(model, args.device, dtype, args.model_id, args.revision)
    res = run_checks(model, tokenizer, args.device, args.golden_dir, meta)
    report(res)
    if args.out:
        Path(args.out).write_text(json.dumps({"meta": meta, **res}, indent=2))
    raise SystemExit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
