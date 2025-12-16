from pathlib import Path
import shutil
import os

def cleanup_checkpoints(
    checkpoint_dir: str = "model_weights",
    keep_gpt2: bool = True,
    keep_best: bool = True,
    max_size_gb: float = 5.0
):
    
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"❌ Directory not found: {checkpoint_dir}")
        return
    
    checkpoints = []
    # Changed 'if' to 'f'
    for f in checkpoint_path.glob("*.pt"):
        size_mb = f.stat().st_size / (1024**2)
        checkpoints.append((f, size_mb, f.stat().st_mtime))
    
    # Sort by modification time (oldest first)
    checkpoints.sort(key=lambda x: x[2])
    
    total_size_gb = sum(size for _, size, _ in checkpoints) / 1024
    print(f"📊 Current checkpoint size: {total_size_gb:.2f} GB")
    
    if total_size_gb <= max_size_gb:
        print(f"✅ Checkpoint size is within limit ({max_size_gb} GB)")
        return
    
    removed = []
    current_size_gb = total_size_gb
    
    for f, size_mb, _ in checkpoints:
        if current_size_gb <= max_size_gb:
            break
        
        if keep_gpt2 and "gpt2.pt" in f.name:
            print(f"  ⏭️  Skipping {f.name} (protected)")
            continue
        if keep_best and "best_model.pt" in f.name:
            print(f"  ⏭️  Skipping {f.name} (protected)")
            continue
        
        try:
            f.unlink()
            removed.append((f.name, size_mb))
            current_size_gb -= size_mb / 1024
            print(f"  🗑️  Removed {f.name} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ❌ Failed to remove {f.name}: {e}")
    
    if removed:
        freed_gb = sum(size for _, size in removed) / 1024
        print(f"\n✅ Freed {freed_gb:.2f} GB by removing {len(removed)} checkpoints")
        print(f"📊 New total size: {current_size_gb:.2f} GB")
    else:
        print("ℹ️  No checkpoints removed")

if __name__ == "__main__":
    import sys
    max_size = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    cleanup_checkpoints(max_size_gb=max_size)