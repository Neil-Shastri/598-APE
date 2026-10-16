import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from olmo_bench.correctness import GOLDEN_DIR, NEW_TOKENS, PROMPTS
from olmo_bench.workload import LONG_TEXT_FILE, TEXT_FILE, greedy_generate, make_input_ids, text_token_ids

TOP1_MIN = 0.99
PPL_TOL = 0.005
TEXTS = [(TEXT_FILE, 1536), (LONG_TEXT_FILE, 8192)]
WORKLOAD = (1, 512, 128)


@torch.no_grad()
def teacher_forced(model, tokenizer, device, text_file, n_tokens):
    ids = torch.tensor([text_token_ids(tokenizer, text_file)[:n_tokens]], device=device)
    logits = model(input_ids=ids).logits[0, :-1].float()
    nll = F.cross_entropy(logits, ids[0, 1:])
    return {"ppl": math.exp(nll.item()), "top1": logits.argmax(-1).cpu()}


def eager_generate(model):
    return lambda ids, n: greedy_generate(model, ids, n)


def generations(gen, tokenizer, device):
    out = []
    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        out.append(gen(ids, NEW_TOKENS)[0, ids.shape[1]:].tolist())
    return out


def workload_tokens(gen, tokenizer, device):
    batch, seq_len, new_tokens = WORKLOAD
    ids = make_input_ids(tokenizer, batch, seq_len, device)
    return gen(ids, new_tokens)[:, seq_len:].cpu()


def snapshot(model, tokenizer, device, gen):
    return {
        "texts": [teacher_forced(model, tokenizer, device, f, n) for f, n in TEXTS],
        "generations": generations(gen, tokenizer, device),
        "workload": workload_tokens(gen, tokenizer, device),
    }


def golden_generations(meta):
    hw = (meta.get("gpu_name") or "cpu").replace(" ", "_")
    path = Path(GOLDEN_DIR) / f"{hw}_{meta['dtype']}_{meta['attn_implementation']}.json"
    if not path.exists():
        return None
    return [g["new_token_ids"] for g in json.loads(path.read_text())["generations"]]


def compare(ref, new, golden):
    texts = []
    for (f, n), r, o in zip(TEXTS, ref["texts"], new["texts"]):
        texts.append({
            "text": f.name, "tokens": n,
            "ppl_ref": r["ppl"], "ppl": o["ppl"],
            "ppl_delta_pct": 100 * (o["ppl"] / r["ppl"] - 1),
            "top1_agreement": (r["top1"] == o["top1"]).float().mean().item(),
        })
    same = (ref["workload"] == new["workload"]).all(dim=0)
    diverged = (~same).nonzero()
    res = {
        "texts": texts,
        "generations_match_eager": [a == b for a, b in zip(ref["generations"], new["generations"])],
        "generations_match_golden": None if golden is None else [a == b for a, b in zip(golden, new["generations"])],
        "workload_tokens_match": same.float().mean().item(),
        "workload_first_divergence": diverged[0].item() if len(diverged) else None,
    }
    res["passed"] = all(t["top1_agreement"] >= TOP1_MIN and abs(t["ppl_delta_pct"]) <= 100 * PPL_TOL
                        for t in texts)
    return res


def report(res):
    for t in res["texts"]:
        print(f"{t['text']} ({t['tokens']} tokens): ppl {t['ppl_ref']:.4f} -> {t['ppl']:.4f} "
              f"({t['ppl_delta_pct']:+.3f}%), top-1 agreement {t['top1_agreement']:.4f}")
    print(f"generations match eager: {res['generations_match_eager']}")
    print(f"generations match golden: {res['generations_match_golden']}")
    print(f"workload tokens match: {res['workload_tokens_match']:.3f}, "
          f"first divergence: {res['workload_first_divergence']}")
    print("NUMERICS", "PASS" if res["passed"] else "FAIL")
