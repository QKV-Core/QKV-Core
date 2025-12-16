import os
import torch

class ModelConfig:
    MODEL_DIR = "model_weights/"
    DB_URI = os.getenv("DB_URI", "sqlite:///llm_core.db")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEFAULT_CONTEXT_LENGTH = 2048
    GPU_LAYERS = 20