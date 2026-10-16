import torch


def round_up(n, m):
    return -(-n // m) * m


class StaticKV:
    # Preallocated K/V per layer. Sliding layers are ring buffers of `sliding_window` slots,
    # full layers have `max_len` slots. Token at position p lives in slot p % size.
    def __init__(self, config, batch_size, max_len, device, dtype):
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.layer_types = config.layer_types
        self.sizes = [min(config.sliding_window, max_len) if t == "sliding_attention" else max_len
                      for t in self.layer_types]
        self.keys, self.values = [], []
        for n in self.sizes:
            shape = (batch_size, config.num_key_value_heads, n, head_dim)
            self.keys.append(torch.zeros(shape, device=device, dtype=dtype))
            self.values.append(torch.zeros(shape, device=device, dtype=dtype))
        self.pos = torch.zeros(1, dtype=torch.long, device=device)
        self.prefilling = False
        self.slots = {}

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        n = self.sizes[layer_idx]
        if self.prefilling:
            length = key_states.shape[2]
            keep = min(length, n)
            idx = torch.arange(length - keep, length, device=self.pos.device) % n
            self.keys[layer_idx].index_copy_(2, idx, key_states[:, :, length - keep:])
            self.values[layer_idx].index_copy_(2, idx, value_states[:, :, length - keep:])
            return key_states, value_states
        self.keys[layer_idx].index_copy_(2, self.slots[n], key_states)
        self.values[layer_idx].index_copy_(2, self.slots[n], value_states)
        return self.keys[layer_idx], self.values[layer_idx]

    # only used by HF mask code during prefill, when the cache is logically empty
    def get_seq_length(self, layer_idx=0):
        return 0

    def get_query_offset(self, layer_idx=0):
        return 0

    def get_mask_sizes(self, query_length, layer_idx):
        return query_length, 0

    def step_masks(self, sliding_full):
        self.slots = {n: self.pos % n for n in set(self.sizes)}
        masks = {}
        for t in set(self.layer_types):
            n = self.sizes[self.layer_types.index(t)]
            if t == "sliding_attention" and sliding_full:
                masks[t] = None
            else:
                masks[t] = (torch.arange(n, device=self.pos.device) <= self.pos).view(1, 1, 1, n)
        return masks


class GraphDecoder:
    def __init__(self, model, batch_size, max_len, use_graph=True):
        assert model.config._attn_implementation == "sdpa", "GraphDecoder needs sdpa attention"
        param = next(model.parameters())
        self.model = model
        self.batch_size = batch_size
        self.max_len = round_up(max_len, 64)
        self.use_graph = use_graph
        self.kv = StaticKV(model.config, batch_size, self.max_len, param.device, param.dtype)
        self.token = torch.zeros(batch_size, 1, dtype=torch.long, device=param.device)
        self.sliding_size = min(self.kv.sizes)
        self.sliding_full = False
        self.length = 0
        self.graphs = {}

    @torch.no_grad()
    def prefill(self, input_ids):
        assert input_ids.shape[0] == self.batch_size and input_ids.shape[1] < self.max_len
        length = input_ids.shape[1]
        self.kv.pos.zero_()
        self.kv.prefilling = True
        try:
            out = self.model(input_ids=input_ids, past_key_values=self.kv, use_cache=True, logits_to_keep=1)
        finally:
            self.kv.prefilling = False
        self.kv.pos.fill_(length)
        self.length = length
        self.sliding_full = length >= self.sliding_size
        self.token.copy_(out.logits[:, -1:].argmax(-1))
        return self.token.clone()

    @torch.no_grad()
    def _step(self):
        masks = self.kv.step_masks(self.sliding_full)
        logits = self.model(input_ids=self.token, position_ids=self.kv.pos.view(1, 1),
                            attention_mask=masks, past_key_values=self.kv, use_cache=True).logits
        self.token.copy_(logits[:, -1:].argmax(-1))
        self.kv.pos.add_(1)

    def capture(self):
        # Warmup steps only write the slot at the current position, which the next real step
        # rewrites with the same values, so restoring pos and token leaves the cache intact.
        pos, token = self.kv.pos.clone(), self.token.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._step()
                self.kv.pos.copy_(pos)
                self.token.copy_(token)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._step()
        self.kv.pos.copy_(pos)
        self.token.copy_(token)
        self.graphs[self.sliding_full] = graph

    def step(self, token):
        assert self.length < self.max_len, "static cache is full"
        self.token.copy_(token)
        if self.use_graph:
            if self.sliding_full not in self.graphs:
                self.capture()
            self.graphs[self.sliding_full].replay()
        else:
            self._step()
        self.length += 1
        return self.token.clone()

    def greedy_generate(self, input_ids, max_new_tokens):
        # prompt + new tokens, like model.generate without EOS stopping
        token = self.prefill(input_ids)
        tokens = [token]
        for _ in range(max_new_tokens - 1):
            token = self.step(token)
            tokens.append(token)
        return torch.cat([input_ids] + tokens, dim=1)
