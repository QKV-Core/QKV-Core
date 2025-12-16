import torch
import gc
from typing import Optional, Dict, Any
import os

def cleanup_memory(aggressive: bool = False):
    """
    Çöp toplayıcıyı ve CUDA önbelleğini temizler.
    Aggressive modda daha derin temizlik yapar.
    """
    gc.collect()
    
    if aggressive:
        for _ in range(3):
            gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        if aggressive:
            torch.cuda.reset_peak_memory_stats()

def get_memory_info(device: Optional[torch.device] = None) -> Dict[str, float]:
    
    if not torch.cuda.is_available():
        return {
            'available': False,
            'total': 0.0,
            'allocated': 0.0,
            'reserved': 0.0,
            'free': 0.0,
            'usage_percent': 0.0
        }
    
    if device is None:
        device = torch.cuda.current_device()
    
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    free = total - reserved
    
    return {
        'available': True,
        'total': total,
        'allocated': allocated,
        'reserved': reserved,
        'free': free,
        'usage_percent': (reserved / total) * 100 if total > 0 else 0.0
    }

def print_memory_info(device: Optional[torch.device] = None):
    
    info = get_memory_info(device)
    
    if not info['available']:
        print("CUDA not available")
        return
    
    print(f"GPU Memory Status:")
    print(f"  Total:     {info['total']:.2f} GB")
    print(f"  Allocated: {info['allocated']:.2f} GB")
    print(f"  Reserved:  {info['reserved']:.2f} GB")
    print(f"  Free:      {info['free']:.2f} GB")
    print(f"  Usage:     {info['usage_percent']:.1f}%")

def check_memory_threshold(threshold_percent: float = 85.0, device: Optional[torch.device] = None) -> bool:
    
    info = get_memory_info(device)
    
    if not info['available']:
        return False
    
    return info['usage_percent'] >= threshold_percent

def optimize_for_gtx1050():
    """
    GTX 1050 (4GB) için özel bellek optimizasyonları uygular.
    """
    if not torch.cuda.is_available():
        return
    
    # Bellek fragmantasyonunu önlemek için process başına limit koyabiliriz
    # Ancak 4GB zaten sınırda olduğu için dikkatli kullanılmalı
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        # VRAM'in %90'ını ayır, işletim sistemine pay bırak
        try:
            torch.cuda.set_per_process_memory_fraction(0.9)
        except RuntimeError:
            pass # Zaten ayarlanmışsa veya bellek doluysa geç
    
    cleanup_memory(aggressive=True)
    
    print("GTX 1050 optimizations applied:")
    print_memory_info()

def monitor_memory_usage(interval_steps: int = 100):
    
    class MemoryMonitor:
        def __init__(self, interval: int):
            self.interval = interval
            self.step = 0
            self.initial_memory = None
        
        def __enter__(self):
            if torch.cuda.is_available():
                self.initial_memory = get_memory_info()
                print(f"Initial memory: {self.initial_memory['allocated']:.2f} GB allocated")
            return self
        
        def __exit__(self, *args):
            if torch.cuda.is_available():
                final_memory = get_memory_info()
                print(f"Final memory: {final_memory['allocated']:.2f} GB allocated")
                if self.initial_memory:
                    diff = final_memory['allocated'] - self.initial_memory['allocated']
                    print(f"Memory delta: {diff:+.2f} GB")
        
        def check(self, current_step: int):
            
            if current_step % self.interval == 0:
                if torch.cuda.is_available():
                    info = get_memory_info()
                    if info['usage_percent'] > 85:
                        print(f"⚠️ High memory usage at step {current_step}: {info['usage_percent']:.1f}%")
                        cleanup_memory()
    
    return MemoryMonitor(interval_steps)

def delete_tensors(*tensors):
    """
    Verilen tensörleri güvenli bir şekilde siler ve belleği temizler.
    """
    for tensor in tensors:
        if tensor is not None:
            del tensor
    
    cleanup_memory()

# GTX 1050 Configuration Constants
GTX1050_VRAM_GB = 4.0
GTX1050_SAFE_THRESHOLD = 0.85
GTX1050_RECOMMENDED_BATCH_SIZE = 1
GTX1050_RECOMMENDED_ACCUMULATION_STEPS = 32

GTX1050_RECOMMENDED_CONFIG = {
    'batch_size': 1,
    'accumulation_steps': 32,
    'use_mixed_precision': True, # fp16 kullanımı VRAM tasarrufu için kritik
    'use_gradient_checkpointing': True, # Hızdan feragat edip VRAM kazanmak için
    'memory_cleanup_interval': 100,
    'max_seq_length': 256,
    'd_model': 256,
    'num_layers': 4, # 'num_laplaces' muhtemelen buydu
}