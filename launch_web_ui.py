import sys
import os
import traceback
from pathlib import Path

# --- 1. CRITICAL PATH FIX ---
# We must set the project root to sys.path BEFORE importing anything else.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

print(f"\nProject Root: {ROOT_DIR}")

# --- 2. SYSTEM DIAGNOSTICS ---
print("Running File System Diagnostics...")

# Check A: Does 'config' folder exist?
config_dir = os.path.join(ROOT_DIR, "config")
if os.path.exists(config_dir) and os.path.isdir(config_dir):
    print("   [OK] 'config' directory found.")
    
    # Check B: Does 'model_config.py' exist inside?
    files = os.listdir(config_dir)
    if "model_config.py" in files:
        print("   [OK] 'model_config.py' found inside config.")
    else:
        print("   [ERROR] 'config' directory exists, but 'model_config.py' is MISSING!")
        print(f"      Contents of config: {files}")
        input("Press Enter to exit...")
        sys.exit(1)
        
    # Check C: Is __init__.py present? (Recommended)
    if "__init__.py" not in files:
        print("   [WARNING] 'config/__init__.py' is missing. Python might not treat it as a package.")
else:
    print(f"   [ERROR] 'config' directory not found at: {config_dir}")
    sys.exit(1)

# Check D: Namespace Shadowing (Critical)
# If a file named 'config.py' exists in the root, Python imports it instead of the folder!
shadow_file = os.path.join(ROOT_DIR, "config.py")
if os.path.exists(shadow_file):
    print("\n   [WARNING] CRITICAL CONFLICT DETECTED: 'Namespace Shadowing'")
    print(f"      A file named 'config.py' exists in the root directory: {shadow_file}")
    print("      Python is trying to import this FILE instead of your config FOLDER.")
    print("      SOLUTION: Delete or rename the 'config.py' file in the root directory.")
    input("Press Enter to exit...")
    sys.exit(1)

print("[OK] System checks passed.\n")
print("-" * 60)

# --- 3. INITIALIZATION ---

# Create necessary directories
directories = ["tokenizer", "model_weights", "storage", "logs", "data"]
for d in directories:
    Path(d).mkdir(exist_ok=True)

# Check PyTorch
try:
    import torch
    print(f"[OK] PyTorch {torch.__version__}")
except ImportError:
    print("[ERROR] PyTorch not found! Please install: pip install torch")
    sys.exit(1)

# Check Gradio
try:
    import gradio as gr
    print(f"[OK] Gradio {gr.__version__}")
except ImportError:
    print("[ERROR] Gradio not found! Please install: pip install gradio")
    sys.exit(1)

# Check CUDA
if torch.cuda.is_available():
    print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[OK] GPU Memory: {gpu_memory:.1f} GB")
else:
    print("[WARNING] CUDA not available, using CPU")

print("\nWeb interface starting...")
print("-" * 60)

# --- 4. LAUNCH APP ---
try:
    # Attempt to import the app creator
    from web_ui.app import create_app
    app = create_app()

    print("\n" + "=" * 60)
    print("QKV Core - Web Interface Started!")
    print("URL: http://localhost:7861")
    print("=" * 60 + "\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        quiet=False
    )

except ModuleNotFoundError as e:
    print(f"\n[ERROR] IMPORT ERROR: {e}")
    print("   This is usually caused by incorrect folder structure or missing __init__.py files.")
    print("   Please check the diagnostic messages above.")
    input("\nPress Enter to exit...")
except Exception as e:
    print(f"\n[ERROR] UNEXPECTED ERROR: {e}")
    traceback.print_exc()
    input("\nPress Enter to exit...")
