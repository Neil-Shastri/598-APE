import functools

import torch
from transformers import cache_utils
from transformers.models.olmo3 import modeling_olmo3 as olmo3

# Ranges are named "<region>:<label>", e.g. "decode_b1:rmsnorm". Regions are "prefill_b<N>" and
# "decode_b<N>" (one range per decode step). A kernel belongs to the innermost label around it.
# Labels on container modules catch whatever their children don't cover:
REMAINDER = {"model": "model_other", "layer": "residual", "attn": "attn_other", "mlp": "mlp_mul"}

GROUP = {
    "qkv_proj": "gemm", "o_proj": "gemm", "gate_up_proj": "gemm", "down_proj": "gemm", "lm_head": "gemm",
    "rmsnorm": "nongemm", "qk_norm": "nongemm", "final_norm": "nongemm", "rope": "nongemm",
    "rope_table": "nongemm", "residual": "nongemm", "act": "nongemm", "mlp_mul": "nongemm",
    "attn_core": "attention", "kv_cache": "attention", "attn_other": "attention",
    "embed": "other", "model_other": "other", "other": "other", "unlabeled": "other",
}


def is_region(name):
    return name.startswith(("prefill_b", "decode_b"))


def category(label):
    # label: innermost range name, or None if the kernel is outside every range
    if label is None or not is_region(label):
        return "unlabeled"
    if ":" not in label:
        return "other"  # inside a region but outside the model (argmax, logits slicing)
    name = label.split(":", 1)[1]
    return REMAINDER.get(name, name)


def region(name, nvtx):
    if nvtx:
        return torch.cuda.nvtx.range(name)
    return torch.profiler.record_function(name)


class Labels:
    """Profiling-only ranges around the OLMo 3 ops; numerics are untouched.

    nvtx=False opens torch.profiler.record_function ranges, nvtx=True opens NVTX ranges.
    Set .phase to the current region name before running the model.
    """

    def __init__(self, model, nvtx=False):
        self.model = model
        self.nvtx = nvtx
        self.phase = ""
        self._undo = []

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *exc):
        self.disable()

    def _wrap(self, fn, name):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with region(f"{self.phase}:{name}", self.nvtx):
                return fn(*args, **kwargs)
        return wrapped

    def _patch(self, owner, attr, name):
        orig = getattr(owner, attr)
        setattr(owner, attr, self._wrap(orig, name))
        self._undo.append(lambda: setattr(owner, attr, orig))

    def _wrap_module(self, mod, name):
        mod.forward = self._wrap(mod.forward, name)
        self._undo.append(lambda: delattr(mod, "forward"))

    def enable(self):
        m = self.model.model
        mods = [(m, "model"), (m.embed_tokens, "embed"), (m.rotary_emb, "rope_table"),
                (m.norm, "final_norm"), (self.model.lm_head, "lm_head")]
        for layer in m.layers:
            attn, mlp = layer.self_attn, layer.mlp
            mods += [
                (layer, "layer"),
                (attn, "attn"), (attn.q_proj, "qkv_proj"), (attn.k_proj, "qkv_proj"), (attn.v_proj, "qkv_proj"),
                (attn.q_norm, "qk_norm"), (attn.k_norm, "qk_norm"), (attn.o_proj, "o_proj"),
                (layer.post_attention_layernorm, "rmsnorm"), (layer.post_feedforward_layernorm, "rmsnorm"),
                (mlp, "mlp"), (mlp.gate_proj, "gate_up_proj"), (mlp.up_proj, "gate_up_proj"),
                (mlp.act_fn, "act"), (mlp.down_proj, "down_proj"),
            ]
        for mod, name in mods:
            self._wrap_module(mod, name)

        # plain functions: the attention module looks these up at call time
        self._patch(olmo3, "apply_rotary_pos_emb", "rope")
        self._patch(cache_utils.Cache, "update", "kv_cache")
        fns = olmo3.ALL_ATTENTION_FUNCTIONS
        impl = self.model.config._attn_implementation
        fns[impl] = self._wrap(fns.get_interface(impl, olmo3.eager_attention_forward), "attn_core")
        self._undo.append(lambda: fns.__delitem__(impl))

    def disable(self):
        while self._undo:
            self._undo.pop()()
