from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

#establish baseline memory
initial_memory = torch.cuda.memory_allocated()
print(f"Initial GPU Memory Allocated: {initial_memory / 1024**2:.2f} MB")



bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)



tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-14B-Chat")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-14B-Chat", device_map="auto", quantization_config=bnb_config)
messages = [
    {"role": "user", "content": "Who are you?"},
]

after_loading_memory = torch.cuda.memory_allocated()
model_vram_size = after_loading_memory - initial_memory

print(f"VRAM consumed by Model Weights (4-bit): {model_vram_size / 1024**2:.2f} MB")



