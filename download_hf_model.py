#!/usr/bin/env python3

import os
import sys
import argparse
from huggingface_hub import snapshot_download

def download_hf_model(model_name, output_dir=None):
    """
    Download HuggingFace model to local directory
    
    Args:
        model_name (str): HuggingFace model name (e.g., "Qwen/Qwen3-1.7B")
        output_dir (str): Output directory (default: model name without org)
    """
    
    # Create output directory name from model name
    if output_dir is None:
        # Extract model name without organization
        # "Qwen/Qwen3-1.8B-Instruct" -> "qwen3-1.8b-instruct"
        model_dir_name = model_name.split('/')[-1].lower()
        output_dir = f"./{model_dir_name}"
    
    print(f"Downloading HuggingFace model: {model_name}")
    print(f"Target directory: {output_dir}")
    
    try:
        # Download model
        snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"\n✅ Download completed successfully!")
        print(f"Model saved to: {output_dir}")
        
        # Show downloaded files
        files = os.listdir(output_dir)
        total_size = 0
        
        print(f"\nDownloaded files ({len(files)} total):")
        for file in sorted(files):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                total_size += size_mb
                print(f"  {file} ({size_mb:.1f} MB)")
        
        print(f"\nTotal size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
        return True
        
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace models")
    parser.add_argument("model_name", help="HuggingFace model name (e.g., Qwen/Qwen3-1.8B-Instruct)")
    parser.add_argument("-o", "--output", help="Output directory (default: auto-generated from model name)")
    
    args = parser.parse_args()
    
    success = download_hf_model(args.model_name, args.output)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    # If no arguments provided, show usage examples
    if len(sys.argv) == 1:
        print("Usage examples:")
        print("  python3 download_hf_model.py Qwen/Qwen3-1.8B-Instruct") # Qwen/Qwen2.5-1.5B-Instruct
        print("  python3 download_hf_model.py microsoft/DialoGPT-small") # Qwen/Qwen2.5-1.5B-Instruct
        print("  python3 download_hf_model.py Qwen/Qwen3-0.5B-Instruct -o custom_dir") # Qwen/Qwen2.5-1.5B-Instruct
        print("\nFor help: python3 download_hf_model.py -h")
        sys.exit(1)
    
    main()
