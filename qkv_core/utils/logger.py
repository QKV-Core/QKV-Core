import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json
import traceback

class LLMLogger:
    """
    Logger for QKV Core (Query-Key-Value Core).
    Provides comprehensive logging for training, inference, and system operations.
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_loggers()
    
    def _setup_loggers(self):
        
        # Ana logger
        self.logger = logging.getLogger('qkv_core')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        # Konsol çıktısı
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # Dosya çıktısı (Genel)
        file_handler = logging.FileHandler(
            self.log_dir / 'app.log',
            encoding='utf-8',
            mode='a'
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Hata dosyası
        error_handler = logging.FileHandler(
            self.log_dir / 'errors.log',
            encoding='utf-8',
            mode='a'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_format)
        self.logger.addHandler(error_handler)
        
        # Performans logger'ı
        self.perf_logger = logging.getLogger('qkv_core.performance')
        self.perf_logger.setLevel(logging.INFO)
        perf_handler = logging.FileHandler(
            self.log_dir / 'performance.log',
            encoding='utf-8',
            mode='a'
        )
        perf_handler.setLevel(logging.INFO)
        perf_format = logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        perf_handler.setFormatter(perf_format)
        self.perf_logger.addHandler(perf_handler)
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        self.logger.error(message, exc_info=exc_info, extra=kwargs)
    
    def critical(self, message: str, exc_info: bool = True, **kwargs):
        self.logger.critical(message, exc_info=exc_info, extra=kwargs)
    
    def log_training_start(
        self,
        run_name: str,
        model_config: Dict[str, Any],
        dataset_size: int
    ):
        self.info(f"Training started: {run_name}")
        self.info(f"Model config: {json.dumps(model_config, indent=2)}")
        self.info(f"Dataset size: {dataset_size} samples")
    
    def log_training_step(
        self,
        run_name: str,
        epoch: int,
        step: int,
        loss: float,
        learning_rate: float,
        elapsed_time: float
    ):
        self.debug(
            f"Training step - Run: {run_name}, "
            f"Epoch: {epoch}, Step: {step}, Loss: {loss:.4f}, "
            f"LR: {learning_rate:.6f}, Time: {elapsed_time:.2f}s"
        )
        self.perf_logger.info(
            f"TRAINING_STEP|{run_name}|{epoch}|{step}|{loss:.4f}|{learning_rate:.6f}|{elapsed_time:.2f}"
        )
    
    def log_training_end(
        self,
        run_name: str,
        total_time: float,
        best_loss: float,
        final_loss: float
    ):
        self.info(f"Training completed: {run_name}")
        self.info(f"Total time: {total_time:.2f}s")
        self.info(f"Best loss: {best_loss:.4f}, Final loss: {final_loss:.4f}")
        self.perf_logger.info(
            f"TRAINING_END|{run_name}|{total_time:.2f}|{best_loss:.4f}|{final_loss:.4f}"
        )
    
    def log_inference_start(
        self,
        prompt: str,
        method: str,
        max_length: int
    ):
        self.debug(
            f"Inference start - Method: {method}, "
            f"Max length: {max_length}, Prompt: {prompt[:50]}..."
        )
    
    def log_inference_end(
        self,
        prompt: str,
        response: str,
        method: str,
        elapsed_time: float,
        tokens_generated: int
    ):
        tokens_per_sec = tokens_generated / elapsed_time if elapsed_time > 0 else 0
        
        self.info(
            f"Inference completed - Method: {method}, "
            f"Time: {elapsed_time:.2f}s, Tokens: {tokens_generated}, "
            f"Speed: {tokens_per_sec:.1f} tokens/sec"
        )
        
        self.perf_logger.info(
            f"INFERENCE|{method}|{elapsed_time:.2f}|{tokens_generated}|{tokens_per_sec:.1f}|"
            f"Prompt: {prompt[:50]}...|Response: {response[:50]}..."
        )
        
        if elapsed_time > 5.0:
            self.warning(
                f"Slow inference detected: {elapsed_time:.2f}s for {tokens_generated} tokens "
                f"({tokens_per_sec:.1f} tokens/sec)"
            )
    
    def log_inference_error(
        self,
        prompt: str,
        error: Exception,
        method: str
    ):
        self.error(
            f"Inference error - Method: {method}, Prompt: {prompt[:50]}...",
            exc_info=True
        )
        self.error(f"Error details: {str(error)}")
        self.error(f"Traceback: {traceback.format_exc()}")
    
    def log_model_load(
        self,
        checkpoint_path: str,
        model_parameters: int,
        load_time: float
    ):
        self.info(
            f"Model loaded: {checkpoint_path}, "
            f"Parameters: {model_parameters:,}, Load time: {load_time:.2f}s"
        )
        self.perf_logger.info(
            f"MODEL_LOAD|{checkpoint_path}|{model_parameters}|{load_time:.2f}"
        )
    
    def log_chat_interaction(
        self,
        user_message: str,
        ai_response: str,
        response_time: float,
        method: str
    ):
        self.info(
            f"Chat interaction - Method: {method}, "
            f"Response time: {response_time:.2f}s"
        )
        self.debug(f"User: {user_message}")
        self.debug(f"AI: {ai_response[:100]}...")
        
        self.perf_logger.info(
            f"CHAT|{method}|{response_time:.2f}|"
            f"User: {user_message[:50]}...|AI: {ai_response[:50]}..."
        )
        
        if response_time > 10.0:
            self.warning(f"Slow chat response: {response_time:.2f}s")
    
    def log_error_with_context(
        self,
        operation: str,
        error: Exception,
        context: Dict[str, Any]
    ):
        self.error(
            f"Error in {operation}: {str(error)}",
            exc_info=True
        )
        self.error(f"Context: {json.dumps(context, indent=2, default=str)}")
    
    def log_system_info(self):
        import torch
        import sys
        
        info = {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'gpu_memory': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None
        }
        
        self.info(f"System info: {json.dumps(info, indent=2, default=str)}")

_logger_instance: Optional[LLMLogger] = None

def get_logger() -> LLMLogger:
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = LLMLogger()
    return _logger_instance

def setup_logging(log_dir: str = "logs"):
    global _logger_instance
    _logger_instance = LLMLogger(log_dir)
    return _logger_instance