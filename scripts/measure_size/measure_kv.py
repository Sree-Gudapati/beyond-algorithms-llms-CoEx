from transformers import AutoConfig

cfg = AutoConfig.from_pretrained("Qwen/Qwen1.5-14B-Chat")
heads = cfg.num_attention_heads
kv_heads = getattr(cfg, "num_key_value_heads", None) or heads
head_dim = cfg.hidden_size // heads
per_tok = 2 * cfg.num_hidden_layers * kv_heads * head_dim * 2   # K+V, fp16 = 2 bytes

print(f"{per_tok / 1024**2:.3f} MiB per token")
print(f"346128 tokens -> {346128 * per_tok / 1024**3:.1f} GiB")