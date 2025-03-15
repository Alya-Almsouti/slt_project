import torch
import torch
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9} GB")

torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()
torch.cuda.reset_max_memory_cached()
