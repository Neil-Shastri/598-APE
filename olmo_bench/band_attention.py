import torch
import torch.nn.functional as F
from transformers.integrations.sdpa_attention import repeat_kv
from transformers.models.olmo3 import modeling_olmo3 as olmo3

# Banded attention for sliding-window layers in long prefill. HF lets query i see keys
# (i - window, i], i.e. the current token plus window - 1 before it. Assumes no padding.

_state = None  # (impl, local override it replaced, or None)


def band_mask(window, chunk, dtype, device):
    # query i of a chunk starting at s vs keys s - window + j: visible iff i < j <= i + window
    i = torch.arange(chunk, device=device)[:, None]
    j = torch.arange(chunk + window, device=device)[None, :]
    mask = torch.zeros(chunk, chunk + window, dtype=dtype, device=device)
    return mask.masked_fill_((j <= i) | (j > i + window), float("-inf"))


def banded_sdpa(query, key, value, window, scale, band, chunk):
    s = query.shape[2]
    outs = [F.scaled_dot_product_attention(query[:, :, :window], key[:, :, :window], value[:, :, :window],
                                           is_causal=True, scale=scale)]
    for start in range(window, s, chunk):
        end = min(start + chunk, s)
        n = end - start
        outs.append(F.scaled_dot_product_attention(query[:, :, start:end], key[:, :, start - window:end],
                                                   value[:, :, start - window:end],
                                                   attn_mask=band[:n, :n + window], scale=scale))
    return torch.cat(outs, dim=2).transpose(1, 2).contiguous()


def _wrap(orig, band, chunk):
    def attention(module, query, key, value, attention_mask, scaling=None, **kwargs):
        window = module.sliding_window
        if window and not module.training and query.shape[2] == key.shape[2] > window:
            groups = getattr(module, "num_key_value_groups", 1)
            key, value = repeat_kv(key, groups), repeat_kv(value, groups)
            return banded_sdpa(query, key, value, window, scaling, band, chunk), None
        return orig(module, query, key, value, attention_mask, scaling=scaling, **kwargs)
    return attention


def enable(model, chunk=1024):
    global _state
    assert _state is None, "band attention is already enabled"
    impl = model.config._attn_implementation
    assert impl == "sdpa", impl
    window = model.config.sliding_window
    chunk = min(chunk, window)
    band = band_mask(window, chunk, model.dtype, model.device)
    fns = olmo3.ALL_ATTENTION_FUNCTIONS
    _state = (impl, fns._local_mapping.get(impl))
    fns[impl] = _wrap(fns.get_interface(impl, olmo3.eager_attention_forward), band, chunk)


def disable(model=None):
    global _state
    if _state is None:
        return
    impl, saved = _state
    fns = olmo3.ALL_ATTENTION_FUNCTIONS
    if saved is None:
        del fns[impl]
    else:
        fns[impl] = saved
    _state = None


def enabled():
    return _state is not None
