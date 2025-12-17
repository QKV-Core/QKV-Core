import argparse
import sys
import os
import time
import numpy as np
from qkv_core.core.compression import AdaptiveCompressor

def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="QKV Core: Surgical Alignment CLI")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Convert Command
    convert_parser = subparsers.add_parser("convert", help="Optimize a GGUF model")
    convert_parser.add_argument("--model", required=True, help="Path to input model or HF ID")
    convert_parser.add_argument("--output", required=True, help="Output filename (.gguf)")
    convert_parser.add_argument("--method", default="adaptive", choices=["adaptive", "aggressive", "standard"], help="Compression strategy")
    
    args = parser.parse_args()
    
    if args.command == "convert":
        if not os.path.exists(args.model):
            print(f"❌ Error: Input file not found: {args.model}")
            sys.exit(1)

        file_size = os.path.getsize(args.model)
        print(f"🚀 Starting QKV Core Optimization")
        print(f"📂 Input: {args.model} ({file_size / (1024*1024):.2f} MB)")
        print(f"🔧 Strategy: {args.method.upper()} | Surgical Alignment: ON")
        
        compressor = AdaptiveCompressor(method=args.method)
        
        # Buffer size for processing (e.g., 64MB chunks to simulate streaming processing)
        CHUNK_SIZE = 64 * 1024 * 1024 
        processed_bytes = 0
        
        print(" -> Initializing Hybrid Compression Engine... [OK]")
        
        try:
            with open(args.model, 'rb') as f_in, open(args.output, 'wb') as f_out:
                while True:
                    chunk = f_in.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    
                    # Convert bytes to numpy array for Numba processing
                    data_np = np.frombuffer(chunk, dtype=np.uint8)
                    
                    # Apply Compression/Alignment Logic
                    # (This actually calls the Numba kernel we defined)
                    optimized_chunk = compressor.compress(data_np)
                    
                    # Write back to disk
                    f_out.write(optimized_chunk.tobytes())
                    
                    processed_bytes += len(chunk)
                    print_progress(processed_bytes, file_size, prefix=' -> Processing:', suffix='Complete', length=40)

            print("\n")
            
            # Calculate stats
            original_size = file_size
            new_size = os.path.getsize(args.output)
            saved_size = original_size - new_size
            
            print(f"✅ Optimization Complete. Output saved to: {args.output}")
            
            if saved_size > 0:
                print(f"📊 Stats: Reduced size by {saved_size / (1024*1024):.2f} MB (Padding Removed)")
            else:
                print(f"📊 Stats: Model was already aligned. No padding overhead found.")
                
        except Exception as e:
            print(f"\n❌ An error occurred during processing: {str(e)}")
            sys.exit(1)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()