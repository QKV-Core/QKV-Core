"""
Feature availability flags for optional dependencies.
"""
try:
    from core.mamba import MambaModel
    MAMBA_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    MAMBA_AVAILABLE = False
    MambaModel = None

try:
    from core.flash_attention import FlashMultiHeadAttention
    FLASH_ATTN_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FLASH_ATTN_AVAILABLE = False
    FlashMultiHeadAttention = None

try:
    from core.quantization import quantize_model, QuantizationConfig, measure_model_size
    QUANTIZATION_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    QUANTIZATION_AVAILABLE = False
    quantize_model = None
    QuantizationConfig = None
    measure_model_size = None

try:
    from core.lora import add_lora_to_model, freeze_base_model, get_lora_state_dict
    LORA_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    LORA_AVAILABLE = False
    add_lora_to_model = None
    freeze_base_model = None
    get_lora_state_dict = None

try:
    from qkv_core.training.rlhf import DPOTrainer, DPOConfig, PreferenceDataset, create_synthetic_preference_data
    RLHF_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    RLHF_AVAILABLE = False
    DPOTrainer = None
    DPOConfig = None
    PreferenceDataset = None
    create_synthetic_preference_data = None

try:
    from qkv_core.formats.huggingface_converter import HuggingFaceConverter, TRANSFORMERS_AVAILABLE
    HF_CONVERTER_AVAILABLE = TRANSFORMERS_AVAILABLE
except (ImportError, ModuleNotFoundError):
    HF_CONVERTER_AVAILABLE = False
    HuggingFaceConverter = None

try:
    from qkv_core.training.scaling_optimizer import ScalingLawsConfig, ScalingLawsOptimizer, compute_optimal_model_config
    SCALING_LAWS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    SCALING_LAWS_AVAILABLE = False
    ScalingLawsConfig = None
    ScalingLawsOptimizer = None
    compute_optimal_model_config = None

try:
    KV_CACHE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    KV_CACHE_AVAILABLE = False
    KVCache = None
    CacheConfig = None
