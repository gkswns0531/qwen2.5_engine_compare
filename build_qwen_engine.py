#!/usr/bin/env python3
"""
TensorRT-LLM Engine Builder Script - Universal Quantization Support
All parameters included version based on official documentation

Based on TensorRT-LLM 0.21.0rc1 official documentation:
https://nvidia.github.io/TensorRT-LLM/commands/trtllm-build.html
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


model_name = "qwen2.5-1.5b-instruct"

def build_trtllm_engine(quantization="fp16"):
    """Build TensorRT-LLM engine from checkpoint with all available parameters"""
    
    # ==========================================================================
    # PATH CONFIGURATION
    # ==========================================================================
    checkpoint_dir = f"./{model_name}_checkpoints"
    engine_dir = f"./{model_name}_engine"

    if quantization == "fp8":
        checkpoint_dir = f"./{model_name}_checkpoints_fp8"
        engine_dir = f"./{model_name}_engine_fp8"
    elif quantization == "fp16":
        checkpoint_dir = f"./{model_name}_checkpoints_fp16"
        engine_dir = f"./{model_name}_engine_fp16"
    else:
        print(f"Error: Invalid quantization type: {quantization}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory {checkpoint_dir} not found!")
        print("Please run convert_qwen_trtllm.py first")
        return False
    
    # Create engine directory
    os.makedirs(engine_dir, exist_ok=True)
    
    print("Building TensorRT-LLM engine...")
    print(f"Quantization type: {quantization.upper()}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Engine directory: {engine_dir}")
    
    # ==========================================================================
    # TENSORRT-LLM BUILD COMMAND - CORRECTED OPTIONS (BASED ON ACTUAL ERROR)
    # ==========================================================================
    cmd = [
        "trtllm-build",
        
        # ======================================================================
        # MANDATORY PARAMETERS (Required parameters)
        # ======================================================================
        "--checkpoint_dir", checkpoint_dir,  # TensorRT-LLM checkpoint directory path (required)
        "--output_dir", engine_dir,          # Engine and config file save directory (default: engine_outputs)
        
        # ======================================================================
        # MODEL CONFIGURATION (Model settings) - All optional
        # ======================================================================
        # "--model_config", "path/to/config.json",     # TensorRT-LLM checkpoint config file path (default: None)
        # "--build_config", "path/to/build.json",     # TensorRT-LLM build config file path (default: None)
        # "--model_cls_file", "path/to/model.py",     # Custom TensorRT-LLM model file path (default: None)
        # "--model_cls_name", "CustomModel",          # Custom TensorRT-LLM model class name (default: None)
        
        # ======================================================================
        # BATCH AND SEQUENCE CONFIGURATION (Batch and sequence settings)
        # ======================================================================
        "--max_batch_size", "1",            # Maximum number of requests the engine can schedule (default: 2048)
        "--max_input_len", "2048",           # Maximum input length for a single request (default: 1024)
        "--max_seq_len", "3072",             # Maximum total length including prompt+output (default: None, inferred from model config)
        "--max_beam_width", "1",             # Maximum beam count for beam search decoding (default: 1)
        "--max_num_tokens", "8192",         # Maximum tokens per batch after padding removal (default: 8192)
        "--opt_num_tokens", "3072",          # Optimal tokens per batch after padding removal (default: max_batch_size * max_beam_width)
        # "--max_encoder_input_len", "1024", # Maximum encoder input length for encoder-decoder models (default: 1024)
        # "--max_prompt_embedding_table_size", "0", # Maximum size for prompt tuning or multimodal input (default: 0)
        
        # ======================================================================
        # KV CACHE CONFIGURATION (KV cache settings)
        # ======================================================================
        "--kv_cache_type", "paged",          # KV cache type (continuous/paged/disabled, default: None)
        "--tokens_per_block", "128",         # Tokens per block for paged KV cache (default: 32)
        
        # ======================================================================
        # PLUGIN CONFIGURATION - ATTENTION (Plugin settings - Attention)
        # ======================================================================
        "--gpt_attention_plugin", "auto",    # Attention plugin for GPT-style decoder models (auto/float16/float32/bfloat16/int32/disable, default: auto)
        # "--bert_attention_plugin", "auto",  # Attention plugin for BERT-style encoder models (auto/float16/float32/bfloat16/int32/disable, default: auto)
        
        # ======================================================================
        # PLUGIN CONFIGURATION - GEMM (Plugin settings - GEMM)
        # ======================================================================
    ]
    
    # Add quantization-specific GEMM plugin settings
    if quantization == "fp8":
        cmd.extend([
            "--gemm_plugin", "fp8",             # General GEMM plugin for FP8
            "--fp8_rowwise_gemm_plugin", "auto", # Rowwise GEMM plugin for FP8
            # "--gemm_swiglu_plugin", "fp8",       # GEMM + SwiGLU fusion plugin - FP8 only on Hopper, not supported on L4
            # "--low_latency_gemm_plugin", "fp8",  # Low latency GEMM plugin - Hopper architecture only, not supported on L4
            # "--low_latency_gemm_swiglu_plugin", "fp8", # Low latency GEMM + SwiGLU fusion plugin - Hopper architecture only
        ])
    else:  # fp16, bf16, or other quantization
        cmd.extend([
            "--gemm_plugin", "auto",            # General GEMM plugin - auto for non-FP8
            "--fp8_rowwise_gemm_plugin", "disable", # Disable FP8 rowwise for non-FP8 quantization
        ])
    
    cmd.extend([
        # "--gemm_allreduce_plugin", "disable", # GEMM + AllReduce kernel fusion plugin (float16/bfloat16/disable, default: None)
        
        # ======================================================================
        # PLUGIN CONFIGURATION - OTHERS (Plugin settings - Others)
        # ======================================================================
        "--nccl_plugin", "auto",             # NCCL plugin - multi-GPU/node support (auto/float16/float32/bfloat16/int32/disable, default: auto)
        # "--moe_plugin", "auto",             # MoE layer acceleration plugin (auto/float16/float32/bfloat16/int32/disable, default: auto)
        # "--mamba_conv1d_plugin", "auto",    # Mamba conv1d operator acceleration plugin (auto/float16/float32/bfloat16/int32/disable, default: auto)
        # "--lora_plugin", "disable",         # LoRA plugin enable (auto/float16/float32/bfloat16/int32/disable, default: None)
        # "--dora_plugin", "disable",         # DoRA plugin enable (enable/disable, default: disable)
        
        # ======================================================================
        # FUSION OPTIMIZATION CONFIGURATION (Fusion optimization settings)
        # ======================================================================
        "--context_fmha", "enable",           # Enable fused multi-head attention in context phase (enable/disable, default: enable)
        "--use_paged_context_fmha", "enable", # Allow advanced features like KV cache reuse and chunked context (enable/disable, default: enable)
    ])
    
    # Add quantization-specific context FMHA settings
    if quantization == "fp8":
        cmd.extend([
            "--use_fp8_context_fmha", "enable",   # Accelerate attention with FP8 context FMHA when FP8 quantization enabled
        ])
    else:  # fp16, bf16, or other quantization
        cmd.extend([
            "--use_fp8_context_fmha", "disable",  # Disable FP8 context FMHA for non-FP8 quantization
        ])
    
    cmd.extend([
        "--norm_quant_fusion", "enable",      # Fuse LayerNorm and quantization kernels into single kernel - auto disabled on L4 (enable/disable, default: disable)
        "--reduce_fusion", "enable",          # Fuse ResidualAdd and LayerNorm kernels after AllReduce - auto disabled on L4 (enable/disable, default: disable)
        "--use_fused_mlp", "enable",          # Fuse two Matmul operations into one in Gated-MLP (enable/disable, default: enable)
        "--user_buffer", "enable",            # Remove additional copy between local-shared buffers in communication kernels - auto disabled on L4 (enable/disable, default: disable)
        # "--fuse_fp4_quant", "disable",      # Fuse FP4 quantization into attention kernels (enable/disable, default: disable)
        
        # ======================================================================
        # ADVANCED OPTIMIZATION CONFIGURATION (Advanced optimization settings)
        # ======================================================================
        "--remove_input_padding", "enable",   # Pack different tokens together to reduce computation and memory consumption (enable/disable, default: enable)
        # "--multiple_profiles", "enable",     # Enable multiple TensorRT optimization profiles - increases engine build time but improves performance (enable/disable, default: disable)
        # "--paged_state", "enable",          # Paged state for memory-efficient management of RNN states (enable/disable, default: enable)
        # "--streamingllm", "disable",        # StreamingLLM - use windowed attention for long text, LLAMA only (enable/disable, default: disable)
        # "--pp_reduce_scatter", "disable",   # Reduce scatter optimization in pipeline parallelization - Mixtral only (enable/disable, default: disable)
        # "--bert_context_fmha_fp32_acc", "disable", # FP32 accumulator for context FMHA in BERT attention plugin (enable/disable, default: disable)
        
        # ======================================================================
        # BUILD OPTIMIZATION CONFIGURATION (Build optimization settings)
        # ======================================================================
        "--workers", "4",                     # Number of workers for parallel build (default: 1)
        # "--weight_sparsity",                 # Enable weight sparsity (default: disable)
        # "--fast_build",                      # Fast engine build feature - may reduce performance, incompatible with int8/int4 quantization (default: disable)
        # "--weight_streaming",               # Offload weights to CPU and stream load at runtime (default: disable)
        # "--strip_plan",                     # Assume refit weights are identical to build time and remove weights from final engine (default: disable)
        
        # ======================================================================
        # TIMING CACHE AND PROFILING (Timing cache and profiling)
        # ======================================================================
        # "--input_timing_cache", "qwen_1.5b_timing.cache",   # Timing cache file path to read (default: None, ignored if file doesn't exist)
        # "--output_timing_cache", "qwen_1.5b_timing.cache",  # Timing cache file path to write (default: model.cache)
        "--profiling_verbosity", "detailed", # Profiling verbosity for generated TensorRT engine (layer_names_only/detailed/none, default: layer_names_only)
        
        # ======================================================================
        # LOGGING AND DEBUGGING (Logging and debugging)
        # ======================================================================
        "--log_level", "info",                # Logging level (internal_error/error/warning/info/verbose/debug/trace, default: info)
        # "--monitor_memory", "enable",         # Enable memory monitor during engine build (default: disable)
        # "--enable_debug_output",            # Enable debug output (default: disable)
        # "--dry_run",                        # Run build process excluding actual engine build - for debugging (default: disable)
        # "--visualize_network", "path/to/dir", # Directory to export TensorRT network to ONNX for debugging before engine build (default: None)
        
        # ======================================================================
        # LOGITS CONFIGURATION (Logits related settings)
        # ======================================================================
        # "--logits_dtype", "float16",        # Data type for logits (float16/float32, default: None)
        # "--gather_context_logits",          # Enable context logits gathering (default: disable)
        # "--gather_generation_logits",       # Enable generation logits gathering (default: disable)
        # "--gather_all_token_logits",        # Enable all token logits gathering - enables both context and generation logits (default: disable)
        
        # ======================================================================
        # LORA CONFIGURATION (LoRA related settings) - Valid only when lora_plugin enabled
        # ======================================================================
        # "--lora_dir", "path/to/lora1", "path/to/lora2", # LoRA weight directories (default: None, multiple allowed, first one for config)
        # "--lora_ckpt_source", "hf",         # LoRA checkpoint source type (hf/nemo, default: hf)
        # "--lora_target_modules", "attn_qkv", "attn_dense", # Target module names for LoRA application (default: None, choices: attn_qkv/attn_q/attn_k/attn_v/attn_dense/mlp_h_to_4h/mlp_4h_to_h/mlp_gate/cross_attn_qkv/cross_attn_q/cross_attn_k/cross_attn_v/cross_attn_dense/moe_h_to_4h/moe_4h_to_h/moe_gate/moe_router/mlp_router/mlp_gate_up)
        # "--max_lora_rank", "64",            # Maximum LoRA rank for various LoRA modules - for workspace size calculation (default: 64)
        
        # ======================================================================
        # SPECULATIVE DECODING CONFIGURATION (Speculative decoding settings)
        # ======================================================================
        # "--speculative_decoding_mode", "none", # Speculative decoding mode (draft_tokens_external/lookahead_decoding/medusa/explicit_draft_tokens/eagle, default: None)
        # "--max_draft_len", "0",             # Maximum draft token length for speculative decoding target model (default: 0)
        
        # ======================================================================
        # AUTO PARALLELIZATION CONFIGURATION (Auto parallelization settings)
        # ======================================================================
        # "--auto_parallel", "1",             # MPI world size for auto parallelization (default: 1)
        # "--gpus_per_node", "8",             # Number of GPUs per node in multi-node setup - cluster spec (default: 8)
        # "--cluster_key", "L4",              # Unique name for target GPU type (A100-SXM-80GB/A100-SXM-40GB/A100-PCIe-80GB/A100-PCIe-40GB/H100-SXM/H100-PCIe/H20/H200-SXM/H200-NVL/A40/A30/A10/A10G/L40S/L40/L20/L4/L2, default: None, inferred from current GPU type if not specified)
    ])
    
    print("=" * 80)
    print("TRTLLM-BUILD COMMAND:")
    print("=" * 80)
    print(f"Command: {' '.join(cmd[:3])}")
    print("Active Parameters:")
    for i in range(3, len(cmd), 2):
        if i + 1 < len(cmd):
            print(f"  {cmd[i]} = {cmd[i+1]}")
    print("=" * 80)
    
    # Execute build
    result = subprocess.run(cmd, check=True)
    print("\n" + "=" * 80)
    print("ENGINE BUILD COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Engine saved to: {engine_dir}")
    print("Ready for inference!")
    return True

def main():
    parser = argparse.ArgumentParser(description="TensorRT-LLM Engine Builder - Universal Quantization Support")
    parser.add_argument("--quantization", type=str, default="fp16", 
                       choices=["fp16", "fp8", "bf16"],
                       help="Quantization type (default: fp16)")
    parser.add_argument("--model_name", type=str, default="qwen2.5-1.5b-instruct",
                       help="Model name (default: qwen2.5-1.5b-instruct)")
    
    args = parser.parse_args()
    
    global model_name
    model_name = args.model_name
    
    print(f"Building TensorRT-LLM engine with {args.quantization.upper()} quantization")
    print(f"Model: {model_name}")
    print("-" * 50)
    
    success = build_trtllm_engine(quantization=args.quantization)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 