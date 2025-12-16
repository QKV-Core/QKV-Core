"""
Training Module - Complete Training Loop with Mixed Precision and Checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from torch.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    # Fallback for older PyTorch versions
    from torch.CUDA.amp import autocast, GradScaler
    AMP_AVAILABLE = False
import os
import time
import json
import gc  # Garbage collection for memory cleanup
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from tqdm import tqdm
import math

from qkv_core.core.transformer import TransformerModel
from qkv_core.storage.postgresql_db import PostgreSQLManager
from qkv_core.training.dataset import TextDataset
from qkv_core.utils.memory_utils import cleanup_memory, optimize_for_gtx1050, get_memory_info
from qkv_core.utils.logger import get_logger

logger = get_logger()


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    Helps with model generalization
    """
    
    def __init__(self, vocab_size: int, padding_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (batch_size, seq_len, vocab_size)
            target: (batch_size, seq_len)
        """
        pred = pred.reshape(-1, self.vocab_size)
        target = target.reshape(-1)
        
        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.vocab_size - 2))  # -2 for true class and padding
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0
            
            mask = (target == self.padding_idx)
            if mask.any():
                true_dist[mask] = 0.0
        
        # Compute KL divergence
        pred_log = torch.log_softmax(pred, dim=-1)
        loss = -(true_dist * pred_log).sum(dim=-1)
        
        # Mask padding
        loss = loss.masked_fill(mask, 0.0)
        
        return loss.mean()


class Trainer:
    """
    Complete Trainer for Transformer LLM
    Supports training, evaluation, checkpointing, and incremental learning
    """
    
    def __init__(
        self,
        model: TransformerModel,
        config: Dict[str, Any],
        tokenizer,
        db_manager: PostgreSQLManager,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Trainer
        
        Args:
            model: TransformerModel instance
            config: Training configuration dictionary
            tokenizer: tokenizer instance
            db_manager: PostgreSQLManager instance
            device: Training device (CUDA/CPU)
        """
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.db = db_manager
        
        # Setup device
        if device is None:
            self.device = torch.device('CUDA' if torch.CUDA.is_available() else 'CPU')
        else:
            self.device = device
        
        # Initialize logger (once at the start)
        logger = None
        try:
            from qkv_core.utils.logger import get_logger
            logger = get_logger()
        except:
            pass
        
        self.model.to(self.device)
        
        # GTX 1050 (4GB VRAM) optimizations - skip for speed
        # if torch.CUDA.is_available() and self.device.type == 'CUDA':
        #     optimize_for_gtx1050()
        
        # PyTorch 2.0+ model Compilation
        self.use_compile = config.get('use_model_compile', False) and hasattr(torch, 'compile')
        if self.use_compile:
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead', fullgraph=False)
            except Exception as e:
                self.use_compile = False
                if logger:
                    logger.warning(f"model compilation failed: {e}")
        
        # Loss function
        self.criterion = LabelSmoothingLoss(
            vocab_size=config.get('vocab_size', 10000),
            padding_idx=tokenizer.pad_token_id,
            smoothing=config.get('label_smoothing', 0.1)
        )
        
        # Optimizer - LAZY: Initialize later to avoid 4-minute delay
        # Store config for lazy initialization
        self._optimizer_config = {
            'lr': config.get('learning_rate', 3e-4),
            'betas': (config.get('adam_beta1', 0.9), config.get('adam_beta2', 0.98)),
            'eps': config.get('adam_epdeleteon', 1e-9),
            'weight_decay': config.get('weight_decay', 0.01)
        }
        self.optimizer = None  # Will be initialized on first use
        
        # Learning rate scheduler (warmup + decay)
        self.warmup_steps = config.get('warmup_steps', 4000)
        self.current_step = 0
        
        # Mixed precision training - optimized for GTX 1050 (Pascal)
        # GTX 1050 is Pascal architecture: supports FP16 storage but no Tensor Cores
        # Using FP16 reduces VRAM usage by 50% and prevents swap overflow
        # FP16 computation on Pascal can be slower than FP32 BUT VRAM insufficiency is worse
        # Swap overflow is much slower - using FP16 actually accelerates total training time
        self.use_amp = config.get('use_mixed_precision', True) and torch.CUDA.is_available()
        if self.use_amp:
            try:
                # FAST: Always use old API for GTX 1050 (new API hangs on Windows)
                if AMP_AVAILABLE:
                    # New API hangs on GTX 1050 + Windows, use old API
                    self.scaler = GradScaler()
                else:
                    # GradScaler settings optimized for GTX 1050
                    self.scaler = GradScaler(
                        init_scale=2.**16,
                        growth_factor=2.0,
                        backoff_factor=0.5,
                        growth_interval=2000
                    )
            except Exception as e:
                self.use_amp = False
                if logger:
                    logger.warning(f"Mixed precision setup failed: {e}, disabling...")
        else:
            self.scaler = None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Cache for causal masks (performance optimization) - MUST be in __init__
        self._mask_cache = {}
        
        # Session tracking
        self.session_id = None
        self.model_version_id = None
        self.skip_db = False  # Always log to PostgreSQL (set to False)
        
        # Memory management for GTX 1050 (4GB VRAM)
        self.cleanup_interval = config.get('memory_cleanup_interval', 100)  # Cleanup every N steps
    
    def _ensure_optimizer(self):
        """Lazy initialize optimizer on first use to avoid 4-minute startup delay."""
        if self.optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self._optimizer_config['lr'],
                betas=self._optimizer_config['betas'],
                eps=self._optimizer_config['eps'],
                weight_decay=self._optimizer_config['weight_decay'],
                foreach=False  # Disable foreach for faster init
            )
    
    def get_lr(self) -> float:
        """
        Get learning rate with warmup and decay
        
        For fine-tuning: Use simple fixed LR (ignore complex scheduler)
        For training from scratch: Use Transformer LR schedule
        """
        # Check if this is fine-tuning (indicated by use_simple_lr flag or very small dataset)
        use_simple_lr = self.config.get('use_simple_lr', False)
        
        if use_simple_lr:
            # Simple fixed LR for fine-tuning (recommended: 1e-5 to 5e-5)
            base_lr = self.config.get('learning_rate', 5e-5)
            return base_lr
        
        # Original Transformer LR schedule (for training from scratch)
        d_model = self.config.get('d_model', 512)
        step = max(self.current_step, 1)
        warmup_steps = self.warmup_steps
        
        # If no warmup, just use step-based decay
        if warmup_steps == 0:
            lr = (d_model ** -0.5) * (step ** -0.5)
        else:
            # Standard warmup + decay
            lr = (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
        
        return lr * self.config.get('learning_rate', 1.0)
    
    def update_lr(self):
        """Update learning rate for current step"""
        lr = self.get_lr()
        self._ensure_optimizer()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def cleanup_memory(self):
        """
        Memory cleanup for GTX 1050 (4GB VRAM)
        Clean up unnecessary variables, clear cache, run garbage collection
        This function prevents swap memory overflow
        """
        # Python garbage collection
        gc.collect()
        
        # CUDA cache cleanup
        if torch.cuda.is_available():
            torch.CUDA.empty_cache()
            # CUDA synchronization (reduces memory usage)
            torch.CUDA.synchronize()
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        progress_callback: Optional[callable] = None,
        total_epochs: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            progress_callback: Optional callback(epoch, total_epochs, epoch_progress) for progress updates
            total_epochs: Total number of epochs (for progress calculation)
        Returns:
            Average loss and learning rate
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(train_loader)
        
        if num_batches == 0:
            self._ensure_optimizer()
            return 0.0, self.optimizer.param_groups[0]['lr']
        
        progress_bar = tqdm(train_loader, desc=f"epoch {epoch}", total=num_batches)
        
        batch_times = []
        for batch_idx, (src, tgt_input, tgt_output, src_mask) in enumerate(progress_bar):
            batch_start = time.time()
            # Update progress callback if provided (every 10 batches to avoid overhead)
            if progress_callback and batch_idx % 10 == 0:
                try:
                    epoch_progress = (batch_idx + 1) / num_batches
                    progress_callback(epoch, total_epochs, epoch_progress)
                except:
                    pass
            # Move to device (non_blocking for faster transfer if pin_memory is used)
            non_blocking = torch.CUDA.is_available()
            src = src.to(self.device, non_blocking=non_blocking)
            tgt_input = tgt_input.to(self.device, non_blocking=non_blocking)
            tgt_output = tgt_output.to(self.device, non_blocking=non_blocking)
            src_mask = src_mask.to(self.device, non_blocking=non_blocking)
            
            # Create target mask (causal mask) - use cache for performance
            seq_len = tgt_input.size(1)
            cache_key = (seq_len, str(self.device))
            if cache_key not in self._mask_cache:
                mask = self.model.create_causal_mask(seq_len, self.device)
                self._mask_cache[cache_key] = mask.unsqueeze(0).unsqueeze(1)
            tgt_mask = self._mask_cache[cache_key]
            
            # Update learning rate
            current_lr = self.update_lr()
            self.current_step += 1
            
            # Forward pass with mixed precision
            # Optional: Use gradient checkpointing for memory efficiency (trades compute for memory)
            use_checkpointing = self.config.get('use_gradient_checkpointing', False)
            
            if self.use_amp:
                if AMP_AVAILABLE:
                    with autocast('CUDA'):  # New API
                        if use_checkpointing:
                            # Gradient checkpointing: Recompute activations during backward
                            # Saves 50-70% memory at cost of 20-30% slower backward pass
                            logits = torch.utils.checkpoint.checkpoint(
                                self.model, src, tgt_input, src_mask, tgt_mask,
                                use_reentrant=False
                            )
                        else:
                            logits = self.model(src, tgt_input, src_mask, tgt_mask)
                        loss = self.criterion(logits, tgt_output)
                else:
                    with autocast():  # Old API fallback
                        if use_checkpointing:
                            logits = torch.utils.checkpoint.checkpoint(
                                self.model, src, tgt_input, src_mask, tgt_mask,
                                use_reentrant=False
                            )
                        else:
                            logits = self.model(src, tgt_input, src_mask, tgt_mask)
                        loss = self.criterion(logits, tgt_output)
                
                # Gradient accumulation: Scale loss by accumulation steps
                accumulation_steps = self.config.get('accumulation_steps', 1)
                scaled_loss = loss / accumulation_steps
                
                # Backward pass (accumulate gradients)
                self.scaler.scale(scaled_loss).backward()
                
                # Update weights only after accumulating enough gradients
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('gradient_clip', 1.0)
                    )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self._ensure_optimizer()
                    self.optimizer.zero_grad()
            
            else:
                # Standard training
                use_checkpointing = self.config.get('use_gradient_checkpointing', False)
                
                if use_checkpointing:
                    # Gradient checkpointing for memory efficiency
                    logits = torch.utils.checkpoint.checkpoint(
                        self.model, src, tgt_input, src_mask, tgt_mask,
                        use_reentrant=False
                    )
                else:
                    logits = self.model(src, tgt_input, src_mask, tgt_mask)
                loss = self.criterion(logits, tgt_output)
                
                # Gradient accumulation: Scale loss by accumulation steps
                accumulation_steps = self.config.get('accumulation_steps', 1)
                scaled_loss = loss / accumulation_steps
                
                # Backward pass (accumulate gradients)
                scaled_loss.backward()
                
                # Update weights only after accumulating enough gradients
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('gradient_clip', 1.0)
                    )
                    
                    # Optimizer step
                    self._ensure_optimizer()
                    self.optimizer.step()
                    self._ensure_optimizer()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item()
            self.global_step += 1
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Update progress bar (her 50 step'te bir veya ilk batch'te)
            if batch_idx % 50 == 0 or batch_idx == 0:
                avg_batch_time = sum(batch_times[-10:]) / len(batch_times[-10:]) if batch_times else batch_time
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}',
                    'batch_time': f'{batch_time:.3f}s'
                })
            
            # Log training step to PostgreSQL
            if self.session_id is not None and not self.skip_db:
                try:
                    self.db.log_training_step(
                        session_id=self.session_id,
                        step=self.global_step,
                        epoch=epoch,
                        loss=loss.item(),
                        learning_rate=current_lr
                    )
                except Exception as e:
                    # Don't block training on DB errors
                    if logger:
                        logger.warning(f"Failed to log training step to DB: {e}")
        
        avg_loss = total_loss / num_batches
        return avg_loss, current_lr
    
    def train(
        self,
        train_dataset: TextDataset,
        num_epochs: Optional[int] = None,
        session_name: str = "training",
        model_version_name: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ):
        """
        Complete training loop
        
        Args:
            train_dataset: Training dataset
            num_epochs: Number of epochs (uses config if None)
            session_name: Name for this training session
            model_version_name: Name for model version
            progress_callback: Optional callback(epoch, total_epochs, epoch_progress) for progress updates
        """
        if num_epochs is None:
            num_epochs = self.config.get('max_epochs', 10)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Training: {session_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        logger.info(f"Epochs: {num_epochs}")
        batch_size = self.config.get('batch_size', 1)
        accumulation_steps = self.config.get('accumulation_steps', 32)
        effective_batch = batch_size * accumulation_steps
        logger.info(f"Batch size: {batch_size} (Effective: {effective_batch} with accumulation)")
        logger.info(f"Accumulation steps: {accumulation_steps}")
        logger.info(f"Gradient checkpointing: {self.config.get('use_gradient_checkpointing', True)}")
        logger.info(f"Mixed precision (FP16): {self.use_amp}")
        logger.info(f"Model parameters: {self.model.get_num_parameters():,}")
        logger.info(f"Dataset size: {len(train_dataset)}")
        logger.info(f"{'='*60}\n")
        
        # Always log to PostgreSQL (skip_db removed - everything goes to DB)
        self.skip_db = False  # Always log to PostgreSQL (store as instance variable)
        
        if not self.skip_db:
            if model_version_name is None:
                model_version_name = f"model_{int(time.time())}"
            
            model_path = os.path.join(
                self.config.get('model_weights_dir', 'model_weights'),
                f"{model_version_name}.pt"
            )
            
            self.model_version_id = self.db.create_model_version(
                version_name=model_version_name,
                model_path=model_path,
                config={
                    'vocab_size': self.config.get('vocab_size'),
                    'd_model': self.config.get('d_model'),
                    'num_layers': self.config.get('num_layers'),
                    'num_heads': self.config.get('num_heads'),
                    'total_parameters': self.model.get_num_parameters()
                },
                description=f"Training session: {session_name}"
            )
            
            self.session_id = self.db.create_training_session(
                self.model_version_id,
                session_name
            )
            
            self.db.save_hyperparameters(self.session_id, self.config)
        else:
            self.model_version_id = None
            self.session_id = None
        
        # Create data loader
        if os.name == 'nt':
            num_workers = 0
        else:
            cpu_count = os.cpu_count() or 1
            num_workers = min(4, max(2, cpu_count // 2))
        
        pin_memory = torch.CUDA.is_available()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 1),
            shuffle=True,
            collate_fn=lambda batch: TextDataset.collate_fn(batch, self.tokenizer.pad_token_id),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        # Initialize optimizer BEFORE training starts (avoid 4+ minute delay on first batch)
        logger.info("Initializing optimizer (one-time setup)...")
        init_start = time.time()
        self._ensure_optimizer()
        init_time = time.time() - init_start
        logger.info(f"Optimizer initialized in {init_time:.2f}s")
        
        # Disable cuDNN benchmark for faster first batch (GTX 1050 optimization)
        # cuDNN benchmark can cause 1-2 minute delay on first forward pass
        if torch.cuda.is_available() and self.device.type == 'cuda':
            original_benchmark = torch.backends.cudnn.benchmark
            torch.backends.cudnn.benchmark = False  # Disable for faster first batch
            logger.info("cuDNN benchmark disabled (faster first batch)")
        
        # Training loop
        start_time = time.time()
        logger.info("Training loop starting...")
        logger.info(f"Epochs: {num_epochs}, Dataset size: {len(train_dataset)}, Batches: {len(train_loader)}")
        
        try:
            for epoch in range(1, num_epochs + 1):
                epoch_start = time.time()
                self.epoch = epoch
                
                logger.info(f"\nEpoch {epoch}/{num_epochs}")
                logger.info(f"{'-'*60}")
                
                # Call progress callback at epoch start
                if progress_callback:
                    try:
                        progress_callback(epoch, num_epochs, 0.0)
                    except:
                        pass
                
                # Train for one epoch
                epoch_train_start = time.time()
                avg_loss, current_lr = self.train_epoch(train_loader, epoch, progress_callback, num_epochs)
                epoch_train_time = time.time() - epoch_train_start
                
                epoch_time = time.time() - epoch_start
                logger.info(f"\nEpoch {epoch} Summary:")
                logger.info(f"  Average Loss: {avg_loss:.4f}")
                logger.info(f"  Learning Rate: {current_lr:.2e}")
                logger.info(f"  Steps: {self.global_step}")
                logger.info(f"  Training time: {epoch_train_time:.2f}s, Total epoch time: {epoch_time:.2f}s")
                
                # Update best loss
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    logger.info("  New best loss! Saving model...")
                    save_start = time.time()
                    best_checkpoint_path = self.save_checkpoint(f"best_model.pt", is_best=True, save_optimizer=False)
                    logger.info(f"  Checkpoint saved in {time.time() - save_start:.2f}s")
                    
                    # Save best checkpoint metadata to PostgreSQL
                    if self.model_version_id is not None and not self.skip_db:
                        try:
                            # Determine if checkpoint should be stored in DB based on size
                            checkpoint_size_mb = best_checkpoint_path.stat().st_size / (1024**2)
                            store_in_db = checkpoint_size_mb < 500  # Store in DB if < 500MB (hybrid approach)
                            
                            self.db.save_model_checkpoint(
                                version_name=f"{model_version_name}_best_epoch_{epoch}",
                                checkpoint_path=str(best_checkpoint_path),
                                config=self.config,
                                store_in_db=store_in_db  # Store in DB for smaller checkpoints
                            )
                        except Exception as e:
                            try:
                                from qkv_core.utils.logger import get_logger
                                logger = get_logger()
                                logger.warning(f"Failed to save best checkpoint metadata to DB: {e}")
                            except:
                                pass
                
                # Update session (skip if DB logging disabled)
                if self.session_id is not None:
                    try:
                        self.db.update_training_session(
                            self.session_id,
                            total_steps=self.global_step,
                            total_epochs=epoch,
                            final_loss=avg_loss,
                            best_loss=self.best_loss
                        )
                    except:
                        pass  # Don't block on DB errors
            
            # Training completed
            total_time = time.time() - start_time
            
            logger.info(f"\n{'='*60}")
            logger.info("Training Completed!")
            logger.info(f"{'='*60}")
            logger.info(f"Total time: {total_time:.1f} seconds")
            logger.info(f"Total steps: {self.global_step}")
            logger.info(f"Final loss: {avg_loss:.4f}")
            logger.info(f"Best loss: {self.best_loss:.4f}")
            logger.info(f"{'='*60}\n")
            
            # Log training completion
            try:
                from qkv_core.utils.logger import get_logger
                logger = get_logger()
            except:
                pass
            
            # Save final model (without optimizer state to save disk space)
            logger.info("Saving final checkpoint...")
            final_checkpoint_path = self.save_checkpoint("final_model.pt", save_optimizer=False)
            
            # Save checkpoint metadata to PostgreSQL
            if self.model_version_id is not None and not self.skip_db:
                try:
                    # Determine if checkpoint should be stored in DB based on size
                    checkpoint_size_mb = final_checkpoint_path.stat().st_size / (1024**2)
                    store_in_db = checkpoint_size_mb < 500  # Store in DB if < 500MB (hybrid approach)
                    
                    self.db.save_model_checkpoint(
                        version_name=f"{model_version_name}_final",
                        checkpoint_path=str(final_checkpoint_path),
                        config=self.config,
                        store_in_db=store_in_db  # Store in DB for smaller checkpoints
                    )
                except Exception as e:
                    if logger:
                        logger.warning(f"Failed to save checkpoint metadata to DB: {e}")
            
            # Also save as best_model.pt (overwrite old one if it exists to free disk space)
            best_model_path = Path(self.config.get('model_weights_dir', 'model_weights')) / "best_model.pt"
            if best_model_path.exists():
                old_size_gb = best_model_path.stat().st_size / (1024**3)
                logger.info(f"Removing old best_model.pt ({old_size_gb:.2f} GB) to free space...")
                try:
                    best_model_path.unlink()
                    logger.info("Old best_model.pt removed")
                except Exception as e:
                    logger.warning(f"Could not remove old best_model.pt: {e}")
            
            logger.info("Saving best_model.pt (for compatibility)...")
            best_checkpoint_path = self.save_checkpoint("best_model.pt", is_best=True, save_optimizer=False)
            
            # Save best checkpoint metadata to PostgreSQL
            if self.model_version_id is not None and not self.skip_db:
                try:
                    # Determine if checkpoint should be stored in DB based on size
                    checkpoint_size_mb = best_checkpoint_path.stat().st_size / (1024**2)
                    store_in_db = checkpoint_size_mb < 500  # Store in DB if < 500MB (hybrid approach)
                    
                    self.db.save_model_checkpoint(
                        version_name=f"{model_version_name}_best",
                        checkpoint_path=str(best_checkpoint_path),
                        config=self.config,
                        store_in_db=store_in_db  # Store in DB for smaller checkpoints
                    )
                except Exception as e:
                    if logger:
                        logger.warning(f"Failed to save best checkpoint metadata to DB: {e}")
            
            logger.info("All checkpoints saved successfully!")
            
            # Update session status (skip if DB logging disabled)
            if self.session_id is not None:
                try:
                    self.db.update_training_session(
                        self.session_id,
                        status='completed'
                    )
                except:
                    pass  # Don't block on DB errors
            
            # Flush any remaining metrics in queue
            if hasattr(self.db, 'shutdown'):
                self.db.shutdown()
        
        except KeyboardInterrupt:
            logger.warning("\n\nTraining interrupted by user!")
            self.save_checkpoint("interrupted_model.pt", save_optimizer=False)
            if self.session_id is not None:
                try:
                    self.db.update_training_session(
                        self.session_id,
                        status='interrupted'
                    )
                except:
                    pass  # Don't block on DB errors
            # Flush any remaining metrics in queue
            if hasattr(self.db, 'shutdown'):
                self.db.shutdown()
        
        except Exception as e:
            logger.error(f"\n\nTraining error: {str(e)}")
            try:
                self.save_checkpoint("error_model.pt", save_optimizer=False)
            except Exception as save_error:
                logger.warning(f"Could not save error checkpoint: {save_error}")
            if self.session_id is not None:
                try:
                    self.db.update_training_session(
                        self.session_id,
                        status='error'
                    )
                except:
                    pass  # Don't block on DB errors
            # Flush any remaining metrics in queue
            if hasattr(self.db, 'shutdown'):
                self.db.shutdown()
            raise
    
    def save_checkpoint(self, filename: str, is_best: bool = False, save_optimizer: bool = False):
        """
        Save model checkpoint
        
        Args:
            filename: checkpoint filename
            is_best: Whether this is the best model so far
            save_optimizer: Whether to save optimizer state (default: False to save disk space)
                          Set to True only if you need to resume training from this checkpoint
        """
        checkpoint_dir = Path(self.config.get('model_weights_dir', 'model_weights'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / filename
        
        try:
            # Base checkpoint with model weights and config
            checkpoint = {
                'epoch': self.epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'best_loss': self.best_loss,
                'config': self.config,
                'current_step': self.current_step
            }
            
            # Only save optimizer state if explicitly requested (saves ~2GB disk space)
            if save_optimizer and self.optimizer is not None:
                try:
                    checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
                    logger.info("Optimizer state included (~2GB)")
                except Exception as e:
                    logger.warning(f"Failed to save optimizer state: {e}")
            
            # Only save scaler state if optimizer is saved
            if save_optimizer and self.use_amp and self.scaler is not None:
                try:
                    checkpoint['scaler_state_dict'] = self.scaler.state_dict()
                except Exception as e:
                    logger.warning(f"Failed to save scaler state: {e}")
            
            # Check disk space before saving
            import shutil
            total, used, free = shutil.disk_usage(checkpoint_dir)
            free_gb = free / (1024**3)
            model_size_mb = sum(p.numel() * 4 for p in self.model.parameters()) / 1024**2
            required_gb = (model_size_mb / 1024) * 1.5  # 1.5x safety margin
            
            if free_gb < required_gb:
                logger.warning(f"Low disk space! Free: {free_gb:.2f} GB, Required: ~{required_gb:.2f} GB")
                logger.info("Trying to free space by removing old checkpoints...")
                
                # Try to remove old checkpoints
                old_checkpoints = list(checkpoint_dir.glob("*.pt"))
                old_checkpoints.sort(key=lambda p: p.stat().st_mtime)  # Oldest first
                
                freed = 0
                for old_ckpt in old_checkpoints[:3]:  # Remove 3 oldest
                    if old_ckpt.name != filename and old_ckpt.name != "gpt2.pt":
                        try:
                            size_mb = old_ckpt.stat().st_size / (1024**2)
                            old_ckpt.unlink()
                            freed += size_mb
                            logger.info(f"Removed {old_ckpt.name} ({size_mb:.1f} MB)")
                        except:
                            pass
                
                # Recheck disk space
                total, used, free = shutil.disk_usage(checkpoint_dir)
                free_gb = free / (1024**3)
                if free_gb < required_gb:
                    raise RuntimeError(
                        f"❌ Insufficient disk space! Free: {free_gb:.2f} GB, Required: ~{required_gb:.2f} GB\n"
                        f"   Please free up at least {required_gb:.2f} GB of disk space."
                    )
                else:
                    logger.info(f"Freed {freed:.1f} MB, now have {free_gb:.2f} GB free")
            
            # Save checkpoint with error handling
            logger.info(f"Saving checkpoint to {checkpoint_path}...")
            logger.info(f"Model weights: ~{model_size_mb:.1f} MB")
            if save_optimizer:
                logger.info(f"With optimizer state: ~{model_size_mb * 3:.1f} MB total")
            else:
                logger.info(f"Without optimizer state (inference-only): ~{model_size_mb:.1f} MB total")
            
            # Move model to CPU before saving to avoid GPU memory issues
            original_device = next(self.model.parameters()).device
            if original_device.type == 'cuda':
                logger.info("Moving model to CPU for saving...")
                self.model = self.model.cpu()
                # Also move state_dict to CPU
                checkpoint['model_state_dict'] = {k: v.cpu() for k, v in checkpoint['model_state_dict'].items()}
            
            # Save with atomic write (write to temp file first, then rename)
            temp_path = checkpoint_path.with_suffix('.tmp')
            
            # Try saving with different methods if first attempt fails
            try:
                # Method 1: Standard torch.save
                torch.save(checkpoint, temp_path, _use_new_zipfile_serialization=False)
            except Exception as e1:
                logger.warning(f"Standard save failed: {e1}")
                try:
                    # Method 2: Try with new zipfile serialization
                    torch.save(checkpoint, temp_path, _use_new_zipfile_serialization=True)
                except Exception as e2:
                    logger.warning(f"New zipfile save failed: {e2}")
                    # Method 3: Try saving in chunks (for very large models)
                    try:
                        import pickle
                        import gzip
                        # Save as compressed pickle
                        with gzip.open(str(temp_path) + '.gz', 'wb') as f:
                            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
                        temp_path = Path(str(temp_path) + '.gz')
                        checkpoint_path = checkpoint_path.with_suffix('.pt.gz')
                    except Exception as e3:
                        raise RuntimeError(f"All save methods failed. Last error: {e3}")
            
            # Atomic rename (works on Windows too)
            if temp_path.exists():
                if checkpoint_path.exists():
                    try:
                        checkpoint_path.unlink()  # Delete old file
                    except:
                        pass  # Ignore if file doesn't exist
                try:
                    temp_path.rename(checkpoint_path)
                except Exception as rename_error:
                    # If rename fails, try copy + delete
                    import shutil
                    shutil.copy2(temp_path, checkpoint_path)
                    temp_path.unlink()
            
            # Move model back to original device
            if original_device.type == 'cuda':
                self.model = self.model.to(original_device)
            
            if is_best:
                logger.info(f"Best model saved to {checkpoint_path}")
            else:
                if 'best' in filename or 'final' in filename:
                    logger.info(f"Checkpoint saved to {checkpoint_path}")
            
            # Return checkpoint path for PostgreSQL logging
            return checkpoint_path
                    
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "file write failed" in str(e).lower():
                logger.warning(f"Disk space or write permission issue when saving {filename}")
                logger.info("Trying to save without optimizer state...")
                # Retry without optimizer state
                try:
                    checkpoint_light = {
                        'epoch': self.epoch,
                        'global_step': self.global_step,
                        'model_state_dict': self.model.state_dict(),
                        'best_loss': self.best_loss,
                        'config': self.config,
                        'current_step': self.current_step
                    }
                    temp_path = checkpoint_path.with_suffix('.tmp')
                    torch.save(checkpoint_light, temp_path)
                    if temp_path.exists():
                        if checkpoint_path.exists():
                            checkpoint_path.unlink()
                        temp_path.rename(checkpoint_path)
                    logger.info(f"Checkpoint saved (without optimizer state) to {checkpoint_path}")
                    return checkpoint_path
                except Exception as e2:
                    logger.error(f"Failed to save checkpoint even without optimizer: {e2}")
                    raise
            else:
                raise
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: str, resume_training: bool = True):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            resume_training: Whether to resume training state
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if resume_training:
            # Load training state
            if checkpoint.get('optimizer_state_dict') is not None:
                self._ensure_optimizer()
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('global_step', 0)
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.current_step = checkpoint.get('current_step', 0)
            
            if self.use_amp and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            logger.info(f"Checkpoint loaded (epoch {self.epoch}, Step {self.global_step})")
        else:
            logger.info("Model weights loaded")
        
        return checkpoint
    
    def continue_training(
        self,
        train_dataset: TextDataset,
        checkpoint_path: str,
        additional_epochs: int,
        session_name: str = "continued_training"
    ):
        """
        Continue training from a checkpoint (incremental learning)
        
        Args:
            train_dataset: New or combined training dataset
            checkpoint_path: Path to checkpoint to resume from
            additional_epochs: Number of additional epochs to train
            session_name: Name for this training session
        """
        logger.info(f"\n{'='*60}")
        logger.info("Continuing Training from checkpoint")
        logger.info(f"{'='*60}\n")
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path, resume_training=True)
        
        # Create new training session
        model_version_name = f"model_continued_{int(time.time())}"
        
        model_path = os.path.join(
            self.config.get('model_weights_dir', 'model_weights'),
            f"{model_version_name}.pt"
        )
        
        self.model_version_id = self.db.create_model_version(
            version_name=model_version_name,
            model_path=model_path,
            config={
                'vocab_size': self.config.get('vocab_size'),
                'd_model': self.config.get('d_model'),
                'num_layers': self.config.get('num_layers'),
                'num_heads': self.config.get('num_heads'),
                'total_parameters': self.model.get_num_parameters()
            },
            description=f"Continued from checkpoint: {checkpoint_path}"
        )
        
        self.session_id = self.db.create_training_session(
            self.model_version_id,
            session_name
        )
        
        # Continue training
        self.train(
            train_dataset,
            num_epochs=additional_epochs,
            session_name=session_name,
            model_version_name=model_version_name
        )

