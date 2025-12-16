import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple
import math

class ScalingLawsConfig:
    
    def __init__(
        self,
        model_parameters: int,
        compute_budget_flops: float,
        vocab_size: int = 50000
    ):
        self.model_parameters = model_parameters
        self.compute_budget = compute_budget_flops
        self.vocab_size = vocab_size
        
        self._compute_optimal_config()
    
    def _compute_optimal_config(self):
        
        N = self.model_parameters
        
        # Scaling law approximation for learning rate
        self.learning_rate = 0.003 * (N / 1e9) ** (-0.25)
        self.learning_rate = max(1e-5, min(1e-3, self.learning_rate))
        
        # Scaling law for batch size
        self.batch_size = int(32 * (N / 1e8) ** 0.25)
        self.batch_size = max(8, min(512, self.batch_size))
        
        self.warmup_ratio = 0.01
        
        # Chinchilla optimal tokens
        self.optimal_tokens = 20 * N
        
        self.weight_decay = 0.1
        
        self.max_grad_norm = 1.0
        
        self.beta1 = 0.9
        self.beta2 = 0.95 if N > 1e9 else 0.999
        
        self.eps = 1e-8
    
    def compute_optimal_model_size(self, compute_budget_flops: float) -> int:
        alpha = 0.73
        N_optimal = (compute_budget_flops / 6.0) ** alpha
        return int(N_optimal)
    
    def compute_training_tokens(self, model_parameters: int, compute_budget: float) -> int:
        return 20 * model_parameters
    
    def print_config(self):
        print("\n📊 Scaling Laws Optimal Configuration\n")
        print("=" * 60)
        print(f"Model Parameters: {self.model_parameters:,}")
        print(f"Compute Budget: {self.compute_budget:.2e} FLOPs")
        print()
        print("Optimal Hyperparameters:")
        print(f"  Learning Rate: {self.learning_rate:.2e}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Weight Decay: {self.weight_decay}")
        print(f"  Gradient Clipping: {self.max_grad_norm}")
        print(f"  Beta1: {self.beta1}, Beta2: {self.beta2}")
        print()
        print(f"Recommended Training Tokens: {self.optimal_tokens:,}")
        print(f"  ({self.optimal_tokens / 1e9:.2f}B tokens)")
        print("=" * 60)

class ScalingLawsOptimizer:
    
    def __init__(
        self,
        model: nn.Module,
        config: ScalingLawsConfig,
        total_steps: int,
        use_amp: bool = True
    ):
        self.model = model
        self.config = config
        self.total_steps = total_steps
        self.use_amp = use_amp
        
        self.warmup_steps = int(total_steps * config.warmup_ratio)
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = self._create_scheduler()
        
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        self.current_step = 0
    
    def _create_scheduler(self):
        
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            else:
                progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                # Cosine decay
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def step(
        self,
        loss: torch.Tensor,
        clip_grad: bool = True
    ):
        
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
            
            if clip_grad:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
            
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.current_step += 1
    
    def get_lr(self) -> float:
        return self.scheduler.get_last_lr()[0]
    
    def state_dict(self) -> Dict:
        state = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'current_step': self.current_step
        }
        
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        
        return state
    
    def load_state_dict(self, state: Dict):
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.current_step = state['current_step']
        
        if self.scaler is not None and 'scaler' in state:
            self.scaler.load_state_dict(state['scaler'])

def compute_optimal_model_config(
    compute_budget_flops: float,
    vocab_size: int = 50000
) -> Dict:
    
    alpha = 0.73
    N_params = int((compute_budget_flops / 6.0) ** alpha)
    
    d_model_choices = [256, 512, 768, 1024, 1536, 2048, 4096, 8192]
    
    estimated_d_model = int((N_params / (vocab_size + 12)) ** 0.5)
    
    d_model = min(d_model_choices, key=lambda x: abs(x - estimated_d_model))
    
    if d_model <= 512:
        num_layers = 6
    elif d_model <= 1024:
        num_layers = 12
    elif d_model <= 2048:
        num_layers = 24
    else:
        num_layers = 32
    
    if d_model <= 512:
        num_heads = 8
    elif d_model <= 2048:
        num_heads = 16
    else:
        num_heads = 32
    
    d_ff = 4 * d_model
    
    actual_params = (
        vocab_size * d_model +
        num_layers * (
            4 * d_model * d_model +
            2 * d_model * d_ff
        )
    )
    
    return {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'd_ff': d_ff,
        'estimated_params': actual_params,
        'compute_budget': compute_budget_flops
    }

def print_scaling_recommendations(
    compute_budget_flops: float,
    vocab_size: int = 50000
):
    print("\n🔬 Scaling Laws Recommendations\n")
    print("=" * 60)
    print(f"Compute Budget: {compute_budget_flops:.2e} FLOPs")
    print(f"Vocabulary Size: {vocab_size:,}")
    print()
    
    config = compute_optimal_model_config(compute_budget_flops, vocab_size)
    
    print("Optimal Model Architecture:")
    print(f"  d_model: {config['d_model']}")
    print(f"  num_layers: {config['num_layers']}")
    print(f"  num_heads: {config['num_heads']}")
    print(f"  d_ff: {config['d_ff']}")
    print(f"  Estimated Parameters: {config['estimated_params']:,}")
    print(f"    ({config['estimated_params'] / 1e6:.1f}M or {config['estimated_params'] / 1e9:.2f}B)")
    print()
    
    scaling_config = ScalingLawsConfig(config['estimated_params'], compute_budget_flops, vocab_size)
    scaling_config.print_config()
    
    print("\n📝 Example Configurations:\n")
    
    examples = [
        ("Tiny", 1e16, 10e6),
        ("Small", 1e18, 100e6),
        ("Medium", 1e20, 1e9),
        ("Large", 1e22, 10e9),
        ("XL", 1e24, 100e9)
    ]
    
    for name, compute, params in examples:
        print(f"{name}:")
        print(f"  Compute: {compute:.0e} FLOPs")
        print(f"  Params: {params:.0e}")
        print(f"  Tokens: {20 * params:.0e} ({20 * params / 1e9:.1f}B)")
        print()
    
    print("=" * 60)

if __name__ == "__main__":
    print_scaling_recommendations(
        compute_budget_flops=1e20,
        vocab_size=50000
    )
    
    print("\n" + "=" * 60)
    print("Key Takeaways from Scaling Laws:")
    print("=" * 60)
    print()
    print("=" * 60)