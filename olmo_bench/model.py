import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/Olmo-3-1025-7B"
REVISION = "a81bae42db3975be1671e27b9c9a56da1a9f980f"


def load_model(device="cuda", dtype=torch.bfloat16, model_id=MODEL_ID, revision=REVISION,
               attn_implementation=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=dtype, device_map=device,
        attn_implementation=attn_implementation,
    )
    model.eval()
    return model, tokenizer


def attn_implementation(model):
    # what transformers actually picked (None resolves to sdpa, or eager if sdpa is unavailable)
    return model.config._attn_implementation
