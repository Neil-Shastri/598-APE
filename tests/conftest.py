import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Olmo3Config, Olmo3ForCausalLM, PreTrainedTokenizerFast

from olmo_bench.correctness import PROMPTS
from olmo_bench.workload import TEXT_FILE


def tiny_tokenizer():
    words = sorted(set((TEXT_FILE.read_text() + " " + " ".join(PROMPTS)).split()))
    vocab = {"[UNK]": 0, "<|endoftext|>": 1} | {w: i + 2 for i, w in enumerate(words)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", eos_token="<|endoftext|>")


def tiny_config(vocab_size):
    # like the 7B config (yarn on full-attention layers only, 3 sliding + 1 full) but tiny;
    # sliding_window 16 so test prompts run past the window
    return Olmo3Config(
        vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096, rope_theta=500000.0,
        sliding_window=16, rms_norm_eps=1e-6, pad_token_id=None, eos_token_id=1,
        rope_scaling={"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 512},
    )


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory):
    torch.manual_seed(0)
    tok = tiny_tokenizer()
    model = Olmo3ForCausalLM(tiny_config(len(tok))).eval()
    path = tmp_path_factory.mktemp("tiny_olmo3")
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def tiny(tiny_dir):
    from olmo_bench.model import load_model
    return load_model("cpu", torch.float32, str(tiny_dir), None)
