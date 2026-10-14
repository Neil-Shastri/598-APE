from pathlib import Path

import torch

TEXT_FILE = Path(__file__).parent / "data" / "scandal_in_bohemia.txt"
LONG_TEXT_FILE = Path(__file__).parent / "data" / "long_prompt.txt"


def text_token_ids(tokenizer, text_file=TEXT_FILE):
    return tokenizer(text_file.read_text(), add_special_tokens=False)["input_ids"]


def make_input_ids(tokenizer, batch_size, seq_len, device, text_file=TEXT_FILE):
    # row i takes the i-th seq_len chunk of the passage, tiled if the passage is too short
    ids = torch.tensor(text_token_ids(tokenizer, text_file), dtype=torch.long)
    need = batch_size * seq_len
    ids = ids.repeat(-(-need // len(ids)))[:need]
    return ids.view(batch_size, seq_len).to(device)


@torch.no_grad()
def prefill(model, input_ids):
    out = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    return out.logits[:, -1:].argmax(-1), out.past_key_values


@torch.no_grad()
def decode_step(model, next_token, past_key_values):
    out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
    return out.logits[:, -1:].argmax(-1), out.past_key_values


def greedy_generate(model, input_ids, max_new_tokens):
    # returns prompt + new tokens, like model.generate; no EOS stopping
    token, cache = prefill(model, input_ids)
    tokens = [token]
    for _ in range(max_new_tokens - 1):
        token, cache = decode_step(model, token, cache)
        tokens.append(token)
    return torch.cat([input_ids] + tokens, dim=1)
