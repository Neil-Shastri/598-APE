"""Merge the q/k/v projections of every layer into one weight (opt-in, used with kernels.enable).

The merged weight is the only copy: the old Linear weights become row views of it,
so memory does not grow and the stock HF forward still works.
"""
import torch
from torch import nn


def _merge(linears):
    assert all(lin.bias is None for lin in linears)
    with torch.no_grad():
        weight = torch.cat([lin.weight for lin in linears])
    merged = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="meta")
    merged.weight = nn.Parameter(weight, requires_grad=False)
    start = 0
    for lin in linears:
        n = lin.out_features
        lin.weight = nn.Parameter(weight[start:start + n], requires_grad=False)
        start += n
    return merged


def merge_qkv(model):
    for layer in model.model.layers:
        attn = layer.self_attn
        if not hasattr(attn, "qkv_proj"):
            attn.qkv_proj = _merge([attn.q_proj, attn.k_proj, attn.v_proj])
