#!/usr/bin/env python3
"""
TensorRT-LLM Qwen Model Converter - Universal Quantization
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

model_name = "qwen2.5-1.5b-instruct"

def convert_qwen_to_trtllm_fp16():
    """Convert Qwen model to TensorRT-LLM checkpoint with FP16"""
    
    model_dir = f"./{model_name}"
    output_dir = f"./{model_name}_checkpoints_fp16"
    
    # Check if model exists
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} not found!")
        return False
    
    print(f"Converting {model_name} to FP16 checkpoint...")
    print(f"Output: {output_dir}")
    
    cmd = [
        "python3", "./TensorRT-LLM/examples/models/core/qwen/convert_checkpoint.py",
        "--model_dir", model_dir,
        "--output_dir", output_dir,
        "--dtype", "float16",
        "--tp_size", "1",
        "--pp_size", "1",
        "--workers", "1",
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"FP16 checkpoint saved to: {output_dir}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"FP16 conversion failed: {e}")
        return False

def convert_qwen_to_trtllm_fp8():
    """Convert Qwen model to TensorRT-LLM checkpoint with FP8 quantization"""
    
    model_dir = f"./{model_name}"
    output_dir = f"./{model_name}_checkpoints_fp8"
    
    # Check if model exists
    if not os.path.exists(model_dir):
        print(f"Error: Model directory {model_dir} not found!")
        return False
    
    print(f"Converting {model_name} to FP8 quantized checkpoint...")
    print(f"Output: {output_dir}")
    
    cmd = [
        "python3", "./TensorRT-LLM/examples/quantization/quantize.py",
        "--model_dir", model_dir,
        "--output_dir", output_dir,
        "--dtype", "float16",
        "--qformat", "fp8",
        "--kv_cache_dtype", "fp8",
        "--calib_size", "512",
        "--quantize_lm_head",
        "--tp_size", "1",
        "--pp_size", "1",
        "--decoder_type", "llama",
    ]
    
    try:
        print("Starting FP8 quantization (may take 10-20 minutes)...")
        result = subprocess.run(cmd, check=True)
        print(f"FP8 checkpoint saved to: {output_dir}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"FP8 conversion failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="TensorRT-LLM Qwen Model Converter")
    parser.add_argument("--quantization", "-q", type=str, default="fp16", 
                       choices=["fp16", "fp8"], 
                       help="Quantization format (default: fp16)")
    
    args = parser.parse_args()
    
    print(f"TensorRT-LLM Qwen Checkpoint Converter ({args.quantization.upper()})")
    
    if args.quantization == "fp8":
        success = convert_qwen_to_trtllm_fp8()
    else:
        success = convert_qwen_to_trtllm_fp16()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 