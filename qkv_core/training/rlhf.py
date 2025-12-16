import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class RLHFConfig:
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    kl_penalty: float = 0.02
    gamma: float = 1.0
    lam: float = 0.95
    
    batch_size: int = 8
    learning_rate: float = 1e-5
    max_length: int = 512
    
    reward_model_path: Optional[str] = None

@dataclass
class DPOConfig:
    beta: float = 0.1
    learning_rate: float = 5e-7
    batch_size: int = 4
    max_length: int = 512
    label_smoothing: float = 0.0

class RewardModel(nn.Module):
    
    def __init__(self, base_model: nn.Module, d_model: int):
        super().__init__()
        
        self.base_model = base_model
        
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        if hasattr(self.base_model, 'encode'):
            hidden_states = self.base_model.encode(input_ids)
        else:
            hidden_states = self.base_model(input_ids)
        
        last_hidden = hidden_states[:, -1, :]
        
        reward = self.value_head(last_hidden).squeeze(-1)
        
        return reward

class PreferenceDataset(Dataset):
    
    def __init__(
        self,
        prompts: List[str],
        chosen: List[str],
        rejected: List[str],
        tokenizer
    ):
        assert len(prompts) == len(chosen) == len(rejected)
        
        self.prompts = prompts
        self.chosen = chosen
        self.rejected = rejected
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        chosen_text = self.chosen[idx]
        rejected_text = self.rejected[idx]
        
        chosen_ids = self.tokenizer.encode(prompt + chosen_text)
        rejected_ids = self.tokenizer.encode(prompt + rejected_text)
        prompt_ids = self.tokenizer.encode(prompt)
        
        return {
            'prompt_ids': torch.tensor(prompt_ids),
            'chosen_ids': torch.tensor(chosen_ids),
            'rejected_ids': torch.tensor(rejected_ids),
            'prompt_length': len(prompt_ids)
        }

class RLHFTrainer:
    
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        reward_model: RewardModel,
        config: RLHFConfig,
        device: str = 'cuda'
    ):
        self.policy_model = policy_model.to(device)
        self.ref_model = ref_model.to(device)
        self.reward_model = reward_model.to(device)
        self.config = config
        self.device = device
        
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.Adam(
            self.policy_model.parameters(),
            lr=config.learning_rate
        )
    
    def compute_rewards(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor
    ) -> torch.Tensor:
        
        full_sequence = torch.cat([prompts, responses], dim=1)
        
        with torch.no_grad():
            rewards = self.reward_model(full_sequence)
        
        return rewards
    
    def compute_kl_penalty(
        self,
        policy_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor
    ) -> torch.Tensor:
        
        kl = policy_logprobs - ref_logprobs
        return kl
    
    def ppo_step(
        self,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        
        policy_logits = self.policy_model(prompts, responses)
        policy_logprobs = F.log_softmax(policy_logits, dim=-1)
        
        policy_logprobs = torch.gather(
            policy_logprobs,
            dim=-1,
            index=responses.unsqueeze(-1)
        ).squeeze(-1)
        
        ratio = torch.exp(policy_logprobs - old_logprobs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon
        ) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        with torch.no_grad():
            ref_logits = self.ref_model(prompts, responses)
            ref_logprobs = F.log_softmax(ref_logits, dim=-1)
            ref_logprobs = torch.gather(
                ref_logprobs,
                dim=-1,
                index=responses.unsqueeze(-1)
            ).squeeze(-1)
        
        kl_penalty = self.compute_kl_penalty(policy_logprobs, ref_logprobs)
        kl_loss = self.config.kl_penalty * kl_penalty.mean()
        
        loss = policy_loss + kl_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            max_norm=1.0
        )
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': loss.item(),
            'mean_kl': kl_penalty.mean().item()
        }
    
    def train_step(
        self,
        prompts: List[str],
        tokenizer
    ) -> Dict[str, float]:
        
        with torch.no_grad():
            prompt_ids = torch.tensor([tokenizer.encode(p) for p in prompts]).to(self.device)
            response_ids = self.policy_model.generate(
                prompt_ids,
                max_length=self.config.max_length
            )
        
        rewards = self.compute_rewards(prompt_ids, response_ids)
        
        with torch.no_grad():
            old_logits = self.policy_model(prompt_ids, response_ids)
            old_logprobs = F.log_softmax(old_logits, dim=-1)
            old_logprobs = torch.gather(
                old_logprobs,
                dim=-1,
                index=response_ids.unsqueeze(-1)
            ).squeeze(-1)
        
        advantages = rewards - rewards.mean()
        advantages = advantages / (advantages.std() + 1e-8)
        
        returns = rewards
        
        metrics = []
        for _ in range(self.config.ppo_epochs):
            step_metrics = self.ppo_step(
                prompt_ids,
                response_ids,
                old_logprobs,
                advantages,
                returns
            )
            metrics.append(step_metrics)
        
        avg_metrics = {
            k: np.mean([m[k] for m in metrics])
            for k in metrics[0].keys()
        }
        
        return avg_metrics

class DPOTrainer:
    
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        config: DPOConfig,
        device: str = 'cuda'
    ):
        self.policy_model = policy_model.to(device)
        self.ref_model = ref_model.to(device)
        self.config = config
        self.device = device
        
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.Adam(
            self.policy_model.parameters(),
            lr=config.learning_rate
        )
    
    def compute_log_probs(
        self,
        model: nn.Module,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        prompt_length: int
    ) -> torch.Tensor:
        
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        
        logits = model(full_ids[:, :-1])
        logprobs = F.log_softmax(logits, dim=-1)
        
        response_logprobs = torch.gather(
            logprobs[:, prompt_length-1:, :],
            dim=-1,
            index=response_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        return response_logprobs.sum(dim=1)
    
    def dpo_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        prompt_ids = batch['prompt_ids'].to(self.device)
        chosen_ids = batch['chosen_ids'].to(self.device)
        rejected_ids = batch['rejected_ids'].to(self.device)
        prompt_length = batch['prompt_length']
        
        policy_chosen_logprobs = self.compute_log_probs(
            self.policy_model,
            prompt_ids,
            chosen_ids,
            prompt_length
        )
        
        policy_rejected_logprobs = self.compute_log_probs(
            self.policy_model,
            prompt_ids,
            rejected_ids,
            prompt_length
        )
        
        with torch.no_grad():
            ref_chosen_logprobs = self.compute_log_probs(
                self.ref_model,
                prompt_ids,
                chosen_ids,
                prompt_length
            )
            
            ref_rejected_logprobs = self.compute_log_probs(
                self.ref_model,
                prompt_ids,
                rejected_ids,
                prompt_length
            )
        
        chosen_ratio = policy_chosen_logprobs - ref_chosen_logprobs
        rejected_ratio = policy_rejected_logprobs - ref_rejected_logprobs
        
        logits = self.config.beta * (chosen_ratio - rejected_ratio)
        loss = -F.logsigmoid(logits).mean()
        
        with torch.no_grad():
            accuracy = (logits > 0).float().mean()
            chosen_rewards = self.config.beta * chosen_ratio
            rejected_rewards = self.config.beta * rejected_ratio
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'chosen_reward_mean': chosen_rewards.mean().item(),
            'rejected_reward_mean': rejected_rewards.mean().item(),
            'reward_margin': (chosen_rewards - rejected_rewards).mean().item()
        }
        
        return loss, metrics
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        
        self.optimizer.zero_grad()
        
        loss, metrics = self.dpo_loss(batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            max_norm=1.0
        )
        self.optimizer.step()
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        eval_loader: Optional[DataLoader] = None
    ):
        
        print(f"\n🚀 Starting DPO Training ({num_epochs} epochs)\n")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            self.policy_model.train()
            epoch_metrics = []
            
            for batch_idx, batch in enumerate(train_loader):
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}")
                    print(f"  Loss: {metrics['loss']:.4f}")
                    print(f"  Accuracy: {metrics['accuracy']:.2%}")
                    print(f"  Reward Margin: {metrics['reward_margin']:.4f}")
            
            avg_metrics = {
                k: np.mean([m[k] for m in epoch_metrics])
                for k in epoch_metrics[0].keys()
            }
            
            print(f"\nEpoch {epoch+1} Summary:")
            for k, v in avg_metrics.items():
                print(f"  {k}: {v:.4f}")
            print("=" * 60)
            
            if eval_loader is not None:
                eval_metrics = self.evaluate(eval_loader)
                print(f"\nEvaluation:")
                for k, v in eval_metrics.items():
                    print(f"  {k}: {v:.4f}")
                print("=" * 60)
    
    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        
        self.policy_model.eval()
        
        eval_metrics = []
        for batch in eval_loader:
            _, metrics = self.dpo_loss(batch)
            eval_metrics.append(metrics)
        
        avg_metrics = {
            k: np.mean([m[k] for m in eval_metrics])
            for k in eval_metrics[0].keys()
        }
        
        return avg_metrics

def create_synthetic_preference_data(num_samples: int = 100):
    
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing",
        "How do I learn Python?",
        "What is machine learning?",
        "Tell me about black holes"
    ] * (num_samples // 5)
    
    chosen = [
        "The capital of France is Paris.",
        "Quantum computing uses quantum mechanics for computation.",
        "Start with basics, practice coding daily, build projects.",
        "Machine learning is AI that learns from data.",
        "Black holes are regions of spacetime with extreme gravity."
    ] * (num_samples // 5)
    
    rejected = [
        "Paris is nice.",
        "It's complicated.",
        "Just code.",
        "It's AI stuff.",
        "They're black."
    ] * (num_samples // 5)
    
    return prompts[:num_samples], chosen[:num_samples], rejected[:num_samples]

def demo_dpo():
    
    print("\n🎯 DPO (Direct Preference Optimization) Demo\n")
    print("This is how modern LLMs like ChatGPT are aligned!")
    print("=" * 60)
    
    print("\n✨ Key Concepts:")
    print("  • RLHF: Complex (needs reward model + PPO)")
    print("  • DPO: Simple (direct optimization from preferences)")
    print("  • Both achieve similar results")
    print("  • DPO is easier to implement and more stable")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    demo_dpo()