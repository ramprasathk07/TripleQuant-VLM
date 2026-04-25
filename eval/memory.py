import torch

def get_vram_usage():
    """Returns current and peak VRAM usage in GB."""
    if not torch.cuda.is_available():
        return 0, 0
    
    current_mem = torch.cuda.memory_allocated() / (1024**3)
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    return current_mem, peak_mem

class MemoryProfiler:
    def __init__(self, device="cuda"):
        self.device = device
        self.reset()

    def reset(self):
        torch.cuda.reset_peak_memory_stats()

    def profile(self, func, *args, **kwargs):
        self.reset()
        start_mem, _ = get_vram_usage()
        
        result = func(*args, **kwargs)
        
        end_mem, peak_mem = get_vram_usage()
        
        return {
            "start_vram_gb": start_mem,
            "end_vram_gb": end_mem,
            "peak_vram_gb": peak_mem,
            "delta_vram_gb": peak_mem - start_mem
        }
