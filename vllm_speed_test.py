#!/usr/bin/env python3
"""
vLLM Qwen2.5-1.5B-Instruct model TTFT & TPS measurement script
Including FP8 native operation verification
"""

import time
import argparse
import json
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
import torch
import asyncio
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import random_uuid
import subprocess
import psutil
import gc

def check_fp8_support():
    """Check FP8 support status in detail"""
    print("Checking FP8 support status:")
    
    # GPU information
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        compute_cap = torch.cuda.get_device_capability(0)
        print(f"   GPU: {gpu_name}")
        print(f"   Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
        
        # Check FP8 support
        fp8_supported = compute_cap[0] >= 8 and compute_cap[1] >= 9  # Ada Lovelace or newer
        if fp8_supported:
            print(f"   FP8 hardware support: YES")
        else:
            print(f"   FP8 hardware support: NO (CC 8.9+ required)")
        
        # Check PyTorch FP8 support
        if hasattr(torch, 'float8_e4m3fn'):
            print(f"   PyTorch FP8 support: YES")
            
            # Simple FP8 operation test
            x = torch.randn(2, 2, device='cuda', dtype=torch.bfloat16)
            # Try converting to FP8
            x_fp8 = x.to(torch.float8_e4m3fn)
            print(f"   FP8 tensor creation: SUCCESS")
        else:
            print(f"   PyTorch FP8 support: NO")
    
    return fp8_supported

def monitor_gpu_kernels():
    """Monitor GPU kernel usage"""
    result = subprocess.run(['nvidia-smi', 'dmon', '-s', 'u', '-c', '1'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            return lines[-1].split()
    return None

def get_memory_info():
    """Get detailed GPU memory usage information"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        return allocated, reserved
    return 0, 0

def verify_fp8_inference(llm, formatted_prompt, sampling_params):
    """Verify if FP8 inference is actually performed"""
    print("\nStarting FP8 inference verification...")
    
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # State before inference
    mem_before_alloc, mem_before_reserved = get_memory_info()
    kernel_before = monitor_gpu_kernels()
    
    print(f"   Memory before inference: {mem_before_alloc:.2f}GB allocated, {mem_before_reserved:.2f}GB reserved")
    
    # Perform actual inference
    start_time = time.time()
    
    # Check vLLM internal state
    if hasattr(llm.llm_engine, 'model_executor'):
        model_executor = llm.llm_engine.model_executor
        if hasattr(model_executor, 'driver_worker'):
            worker = model_executor.driver_worker
            if hasattr(worker, 'model_runner'):
                model_runner = worker.model_runner
                if hasattr(model_runner, 'model'):
                    model = model_runner.model
                    
                    # Check dtype of first layer weights
                    for name, param in model.named_parameters():
                        if 'weight' in name:
                            print(f"   Weight '{name[:50]}...': {param.dtype}")
                            break
                    
                    # Check KV cache dtype
                    if hasattr(model_runner, 'kv_cache'):
                        kv_cache = model_runner.kv_cache
                        if hasattr(kv_cache, 'dtype'):
                            print(f"   KV cache dtype: {kv_cache.dtype}")
    
    # Execute inference
    outputs = llm.generate([formatted_prompt], sampling_params)
    
    inference_time = time.time() - start_time
    
    # State after inference
    mem_after_alloc, mem_after_reserved = get_memory_info()
    kernel_after = monitor_gpu_kernels()
    
    print(f"   Memory after inference: {mem_after_alloc:.2f}GB allocated, {mem_after_reserved:.2f}GB reserved")
    print(f"   Inference time: {inference_time:.3f}s")
    
    # Memory usage analysis
    mem_diff_alloc = mem_after_alloc - mem_before_alloc
    mem_diff_reserved = mem_after_reserved - mem_before_reserved
    
    print(f"   Memory increase: {mem_diff_alloc:.2f}GB allocated, {mem_diff_reserved:.2f}GB reserved")
    
    # FP8 usage estimation
    if mem_diff_alloc < 0.5:  # Low memory usage expected when using FP8
        print(f"   Low memory usage -> High possibility of FP8 optimization")
    else:
        print(f"   High memory usage -> High possibility of BF16 usage")
    
    return outputs, {
        'inference_time': inference_time,
        'memory_before': (mem_before_alloc, mem_before_reserved),
        'memory_after': (mem_after_alloc, mem_after_reserved),
        'memory_diff': (mem_diff_alloc, mem_diff_reserved)
    }

def format_qwen_message(prompt: str) -> str:
    """Format message for Qwen2.5 model"""
    return f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

def run_streaming_test(
    model_path: str,
    prompts: List[str],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    # Additional optimization options
    max_num_batched_tokens: int = 8192,  # Chunked Prefill optimization
    max_num_seqs: int = 64,  # Number of concurrent sequences
    enable_prefix_caching: bool = True,  # Prefix caching
    enable_chunked_prefill: bool = True,  # Enable Chunked Prefill
    use_v2_block_manager: bool = True,  # Use V2 block manager
    kv_cache_dtype: str = "auto",  # KV cache data type
    quantization: str = None,  # Quantization (e.g., "fp8", "awq")
    speculative_model: str = None,  # Inference acceleration (Speculative Decoding)
):
    """Streaming-based accurate TTFT measurement (including FP8 verification)"""
    
    # First check FP8 support status
    fp8_supported = check_fp8_support()
    print("-" * 50)
    
    print(f"Loading vLLM model: {model_path}")
    print(f"Test settings:")
    print(f"   - Max tokens: {max_tokens}")
    print(f"   - Max batched tokens: {max_num_batched_tokens}")
    print(f"   - Max sequences: {max_num_seqs}")
    print(f"   - Temperature: {temperature}")
    print(f"   - Top-p: {top_p}")
    print(f"   - Top-k: {top_k}")
    print(f"   - Tensor parallel: {tensor_parallel_size}")
    print(f"   - GPU memory utilization: {gpu_memory_utilization}")
    print(f"   - Prefix caching: {enable_prefix_caching}")
    print(f"   - Chunked Prefill: {enable_chunked_prefill}")
    print(f"   - V2 block manager: {use_v2_block_manager}")
    print(f"   - KV cache type: {kv_cache_dtype}")
    if quantization:
        print(f"   - Quantization: {quantization}")
        if quantization == "fp8" and not fp8_supported:
            print(f"   Warning: FP8 quantization requested but hardware support uncertain")
    if speculative_model:
        print(f"   - Speculative Decoding: {speculative_model}")
    print("-" * 50)
    
    # Initialize vLLM model (with all optimization options)
    start_time = time.time()
    
    # Auto-adjust dtype for FP8 quantization
    if quantization == "fp8":
        model_dtype = "auto"  # Auto selection for FP8 quantization
        print(f"FP8 quantization detected: setting dtype to 'auto'")
    else:
        model_dtype = torch.bfloat16  # Default value
    
    llm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
        "dtype": model_dtype,
        "max_model_len": 3072,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "enable_prefix_caching": enable_prefix_caching,
        "enable_chunked_prefill": enable_chunked_prefill,
        "use_v2_block_manager": use_v2_block_manager,
        "kv_cache_dtype": kv_cache_dtype,
    }
    
    # Add optional parameters
    if quantization:
        llm_kwargs["quantization"] = quantization
    if speculative_model:
        llm_kwargs["speculative_model"] = speculative_model
    
    llm = LLM(**llm_kwargs)
    load_time = time.time() - start_time
    print(f"Model loading completed ({load_time:.2f}s)")
    
    # Output optimization information
    print(f"Applied optimizations:")
    print(f"   PagedAttention (default)")
    print(f"   Continuous Batching (default)")
    print(f"   Optimized CUDA Kernels (default)")
    
    # Model precision information
    if quantization == "fp8":
        print(f"   Model Weights: FP8 (quantized)")
        print(f"   Model Compute: Mixed FP8/BF16 (auto upcasting)")
    else:
        print(f"   Model Precision: BF16 (weights/activations)")
    
    # KV cache information
    if kv_cache_dtype != "auto":
        print(f"   KV Cache: {kv_cache_dtype.upper()} (memory optimized)")
    else:
        print(f"   KV Cache: AUTO (default)")
    
    if enable_chunked_prefill:
        print(f"   Chunked Prefill (batch tokens: {max_num_batched_tokens})")
    if enable_prefix_caching:
        print(f"   Prefix Caching")
    if use_v2_block_manager:
        print(f"   V2 Block Manager")
    if quantization and quantization != "fp8":
        print(f"   Additional Quantization: {quantization.upper()}")
    if speculative_model:
        print(f"   Speculative Decoding")
    print("-" * 50)
    
    # Add warm-up step (simple and fast)
    print("Running warm-up inference...")
    warmup_prompt = "Hello"
    warmup_formatted = format_qwen_message(warmup_prompt)
    warmup_sampling_params = SamplingParams(
        temperature=0.01,
        top_p=1.0,
        top_k=1,
        max_tokens=1024,
        repetition_penalty=1.0,
        stop=["<|im_end|>", "<|endoftext|>"]
    )
    
    warmup_start = time.time()
    warmup_outputs = llm.generate([warmup_formatted], warmup_sampling_params)
    warmup_end = time.time()
    print(f"Warm-up completed ({warmup_end - warmup_start:.2f}s)")
    
    print("-" * 50)
    
    formatted_prompts = [format_qwen_message(prompt) for prompt in prompts]
    results = []
    
    print(f"Starting accurate TTFT & TPS measurement ({len(prompts)} prompts)")
    
    # Perform FP8 verification on first prompt
    fp8_verification_done = False
    
    for i, (prompt, formatted_prompt) in enumerate(zip(prompts, formatted_prompts)):
        print(f"Processing prompt {i+1}/{len(prompts)}...")
        
        # Set sampling parameters (unified with other engines)
        sampling_params = SamplingParams(
            temperature=temperature if temperature > 0 else 0.01,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            repetition_penalty=1.05,  # Unified with others
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        # FP8 verification on first prompt
        if i == 0 and quantization == "fp8" and not fp8_verification_done:
            outputs, verification_info = verify_fp8_inference(llm, formatted_prompt, sampling_params)
            fp8_verification_done = True
            request_start_time = time.time() - verification_info['inference_time']
            request_end_time = time.time()
            actual_ttft = verification_info['inference_time'] * 1000  # Use verification time as TTFT
        else:
            # Accurate TTFT calculation: generate only first token (same method as others)
            ttft_sampling_params = SamplingParams(
                temperature=temperature if temperature > 0 else 0.01,
                top_p=top_p,
                top_k=top_k,
                max_tokens=1,  # Only first token
                repetition_penalty=1.05,  # Unified with others
                stop=["<|im_end|>", "<|endoftext|>"]
            )
            
            # Accurate TTFT measurement
            ttft_start = time.time()
            first_output = llm.generate([formatted_prompt], ttft_sampling_params)
            ttft_end = time.time()
            
            actual_ttft = (ttft_end - ttft_start) * 1000  # ms
            
            # Full text generation
            request_start_time = time.time()
            outputs = llm.generate([formatted_prompt], sampling_params)
            request_end_time = time.time()
        
        # Process output results
        output = outputs[0]
        input_tokens = len(output.prompt_token_ids)
        output_tokens = len(output.outputs[0].token_ids)
        generated_text = output.outputs[0].text
        
        # Total time
        total_time = request_end_time - request_start_time
        
        # TPS calculation (total output tokens / total time)
        tps = output_tokens / total_time if total_time > 0 else 0
        
        result = {
            "prompt_index": i,
            "original_prompt": prompt,
            "generated_text": generated_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": actual_ttft,  # Accurate TTFT
            "tps": tps,
            "total_time": total_time,
            "start_time": request_start_time,
            "end_time": request_end_time
        }
        results.append(result)
    
    # Result analysis (same format as TensorRT-LLM)
    # Exclude first 2 results for warmup, use remaining 10 for statistics
    warm_results = results[2:] if len(results) > 2 else results
    
    total_input_tokens = sum(r['input_tokens'] for r in results)  # Total for all results
    total_output_tokens = sum(r['output_tokens'] for r in results)  # Total for all results
    avg_ttft = sum(r['ttft_ms'] for r in warm_results) / len(warm_results) if warm_results else 0
    valid_tps_results = [r['tps'] for r in warm_results if r['tps'] > 0]
    avg_tps = sum(valid_tps_results) / len(valid_tps_results) if valid_tps_results else 0
    
    # Output results (same format as TensorRT-LLM)
    print("=" * 70)
    print("vLLM Accurate TTFT & TPS Measurement Results (with optimizations)")
    print("=" * 70)
    print(f"Total input tokens: {total_input_tokens:,}")
    print(f"Total output tokens: {total_output_tokens:,}")
    print(f"Average TTFT: {avg_ttft:.1f}ms")
    print(f"Average TPS: {avg_tps:.1f} tokens/s")
    print("=" * 70)
    
    # Detailed results for each prompt (same format as TensorRT-LLM)
    for i, result in enumerate(results):
        print(f"\nPrompt {i+1}:")
        print(f"   Input: {result['original_prompt'][:60]}...")
        print(f"   Output: {result['generated_text'][:100]}...")
        print(f"   Performance:")
        print(f"      - Input tokens: {result['input_tokens']}")
        print(f"      - Output tokens: {result['output_tokens']}")
        print(f"      - TTFT: {result['ttft_ms']:.1f}ms")
        print(f"      - TPS: {result['tps']:.1f} tokens/s")
        print(f"      - Total time: {result['total_time']:.2f}s")
    
    return {
        "engine_type": "vLLM",  # For engine type identification
        "model_path": model_path,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "avg_ttft_ms": avg_ttft,
        "avg_tps": avg_tps,
        "load_time": load_time,  # Add same as TensorRT-LLM
        "optimization_info": {  # Add optimization information
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_num_seqs": max_num_seqs,
            "enable_prefix_caching": enable_prefix_caching,
            "enable_chunked_prefill": enable_chunked_prefill,
            "use_v2_block_manager": use_v2_block_manager,
            "kv_cache_dtype": kv_cache_dtype,
            "quantization": quantization,
            "speculative_model": speculative_model
        },
        "results": results
    }

def main():
    parser = argparse.ArgumentParser(description="vLLM Qwen2.5-1.5B TTFT & TPS measurement")
    parser.add_argument("--model_path", type=str, default="./qwen2.5-1.5b-instruct", 
                       help="Model path (default: ./qwen2.5-1.5b-instruct)")
    parser.add_argument("--max_tokens", type=int, default=1024,
                       help="Maximum number of tokens to generate (default: 1024)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature (default: 0.0)")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="Top-p sampling (default: 1.0)")
    parser.add_argument("--top_k", type=int, default=1,
                       help="Top-k sampling (default: 1)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size (default: 1)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                       help="GPU memory utilization (default: 0.9)")
    
    # Additional optimization options
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192,
                       help="Chunked Prefill max batch tokens (default: 8192)")
    parser.add_argument("--max_num_seqs", type=int, default=64,
                       help="Number of concurrent sequences (default: 64)")
    parser.add_argument("--disable_prefix_caching", action="store_true",
                       help="Disable prefix caching")
    parser.add_argument("--disable_chunked_prefill", action="store_true",
                       help="Disable Chunked Prefill")
    parser.add_argument("--disable_v2_block_manager", action="store_true",
                       help="Disable V2 block manager")
    parser.add_argument("--kv_cache_dtype", type=str, default="auto",
                       help="KV cache data type (default: auto)")
    parser.add_argument("--quantization", type=str, default="fp8",
                       help="Quantization method (e.g., fp8, awq)")
    parser.add_argument("--speculative_model", type=str, default=None,
                       help="Speculative Decoding model path")
    
    parser.add_argument("--custom_prompts", type=str, nargs="+",
                       help="Custom prompts")
    parser.add_argument("--output_file", type=str, default=None,
                       help="JSON file path to save results")
    
    args = parser.parse_args()
    
    # Prepare test prompts
    if args.custom_prompts:
        test_prompts = args.custom_prompts
    else:
        test_prompts = [
        '''
        Provide an extremely comprehensive and detailed technical analysis of modern GPU-accelerated large language model inference optimization techniques and frameworks. Begin with a thorough explanation of the fundamental principles underlying transformer architecture optimization, including the mathematical foundations of self-attention mechanisms, the computational complexity analysis of multi-head attention operations, and the memory access patterns that dominate performance in modern GPU architectures. Discuss in detail the evolution from basic attention implementations to advanced optimizations like flash attention, paged attention, and grouped-query attention, explaining the specific algorithmic improvements and their impact on both memory usage and computational efficiency. Continue with an in-depth analysis of memory management strategies in LLM inference systems. Explain the challenges of managing large model weights, intermediate activations, and KV cache data on GPU memory hierarchies. Detail the implementation of paged KV cache systems, including the mathematical models for optimal block size selection, memory fragmentation mitigation strategies, and the trade-offs between memory efficiency and access latency. Discuss advanced techniques like memory pooling, prefix caching, and dynamic memory allocation algorithms used in production systems. Analyze the role of quantization techniques in modern LLM inference, starting with the theoretical foundations of numerical precision in neural networks. Provide detailed explanations of different quantization approaches including post-training quantization, quantization-aware training, and dynamic quantization. Focus specifically on FP8 quantization, explaining the IEEE 754 floating-point representation, the specific format variations (E4M3 vs E5M2), and the hardware support requirements. Discuss the implementation challenges of mixed-precision arithmetic, automatic scaling techniques, and the accuracy preservation methods used in production quantization systems. Examine the parallelization strategies employed in distributed LLM inference, including tensor parallelism, pipeline parallelism, and sequence parallelism. Explain the mathematical partitioning of transformer layers across multiple GPUs, the communication patterns required for different parallelization schemes, and the optimization of all-reduce operations in multi-GPU environments. Detail the implementation of efficient gradient synchronization, load balancing algorithms, and fault tolerance mechanisms in distributed inference systems. Provide a comprehensive comparison of leading inference frameworks including TensorRT-LLM, vLLM, DeepSpeed-FastGen, FasterTransformer, and text-generation-inference.
        Design and explain a complete production-ready deployment pipeline for large language models, covering every aspect from initial model development through scalable serving infrastructure. Begin with a detailed analysis of model architecture selection criteria, including the evaluation of different transformer variants, the impact of model size on inference performance, and the trade-offs between model complexity and deployment feasibility. Discuss the mathematical relationships between model parameters, memory requirements, and inference latency across different hardware configurations. Provide an exhaustive guide to model optimization techniques for production deployment. Start with training-time optimizations including efficient attention implementations, gradient checkpointing strategies, and mixed-precision training methodologies. Detail the implementation of various quantization approaches, from simple post-training quantization to advanced techniques like QLoRA, GPTQ, and AWQ. Explain the mathematical foundations of each quantization method, the specific algorithms used for weight compression, and the accuracy preservation techniques employed in each approach. Continue with a comprehensive analysis of inference engine compilation and optimization. Explain the process of converting trained models to optimized inference engines, including the role of intermediate representations like ONNX, the graph optimization passes performed by inference engines, and the kernel fusion strategies used to minimize memory bandwidth requirements. Detail the implementation of custom CUDA kernels for specific operations, the optimization of memory access patterns, and the techniques used to maximize GPU utilization. Discuss advanced runtime optimization strategies including dynamic batching algorithms, request scheduling policies, and load balancing techniques. Explain the mathematical models used to predict optimal batch sizes, the algorithms for managing variable-length sequences, and the techniques for minimizing tail latency in production serving systems. Detail the implementation of speculative decoding, parallel sampling, and other advanced inference acceleration techniques. Analyze the infrastructure requirements for scalable LLM serving, including the design of distributed serving architectures, the implementation of auto-scaling systems, and the monitoring and observability requirements for production deployments. Discuss the integration with existing MLOps pipelines, the implementation of A/B testing frameworks for model updates, and the techniques for managing model versioning and rollback procedures.
        Conduct a thorough comparative analysis of attention mechanisms and their implementations in modern large language models, focusing on the computational, memory, and performance characteristics of each approach. Begin with the mathematical foundations of the original transformer attention mechanism, including the detailed derivation of the scaled dot-product attention formula, the role of the scaling factor in numerical stability, and the computational complexity analysis showing the quadratic relationship with sequence length. Explain the evolution of attention mechanisms from the original implementation to modern optimizations. Detail the mathematical formulations and algorithmic improvements in flash attention, including the tiling strategies used to reduce memory usage, the online softmax computation techniques, and the specific CUDA kernel implementations that enable efficient GPU utilization. Analyze the memory access patterns and bandwidth requirements of different attention implementations, explaining how modern approaches minimize the number of memory transactions required. Provide an in-depth analysis of grouped-query attention (GQA) and its variants, including the mathematical reformulation of the attention computation, the impact on model capacity and performance, and the specific implementation challenges in distributed inference systems. Explain how GQA reduces the KV cache memory requirements while maintaining model quality, and detail the optimal grouping strategies for different model architectures and deployment scenarios. Examine sliding window attention mechanisms and their applications in long-context language models. Explain the mathematical foundations of local attention patterns, the algorithms for implementing efficient sliding window computations, and the techniques for combining local and global attention in hybrid architectures. Detail the memory and computational advantages of windowed attention, and analyze the trade-offs between context length and attention quality. Discuss advanced attention variants including sparse attention patterns, learned attention sparsity, and adaptive attention mechanisms. Explain the mathematical models for attention sparsity, the algorithms for efficiently computing sparse attention operations, and the hardware-specific optimizations required for different sparsity patterns. Analyze the impact of attention mechanism choice on overall model performance, including the effects on training stability, inference speed, and memory usage across different hardware platforms and deployment scenarios.
        Analyze the complete ecosystem of batching strategies and request scheduling algorithms used in high-performance LLM inference systems. Begin with the fundamental principles of batch processing in neural network inference, including the mathematical analysis of how batch size affects GPU utilization, memory bandwidth requirements, and overall system throughput. Explain the specific challenges introduced by variable-length sequences in language model inference and the algorithmic approaches used to address these challenges. Detail the implementation of continuous batching systems, starting with the mathematical models for predicting optimal batch composition, the algorithms for managing dynamic sequence lengths, and the techniques for minimizing padding overhead. Explain how continuous batching differs from traditional static batching approaches, including the specific data structures and algorithms required to support dynamic batch modification during inference. Analyze the performance implications of different batching strategies across various hardware configurations and model architectures. Provide a comprehensive analysis of advanced batching techniques including chunked prefill, speculative batching, and priority-based scheduling. Explain the mathematical foundations of chunked prefill, including the optimal chunk size selection algorithms, the memory management strategies required for chunk-based processing, and the techniques for maintaining numerical stability across chunk boundaries. Detail the implementation of speculative batching systems, including the prediction algorithms used to anticipate request characteristics and the fallback mechanisms employed when predictions fail. Examine the role of request scheduling in optimizing system performance, including the algorithms for prioritizing requests based on various criteria such as expected completion time, request priority, and resource requirements. Explain the mathematical models used for load prediction, the techniques for balancing latency and throughput objectives, and the implementation of fair scheduling policies that prevent starvation of long-running requests. Discuss the integration of batching systems with distributed inference architectures, including the challenges of coordinating batching decisions across multiple GPUs and nodes. Detail the implementation of cross-device batching strategies, the algorithms for managing distributed KV cache systems, and the techniques for optimizing communication patterns in multi-node inference deployments.
        Examine the comprehensive landscape of memory optimization techniques employed in large language model inference systems, focusing on the mathematical foundations, algorithmic implementations, and practical deployment considerations of each approach. Begin with a detailed analysis of memory hierarchies in modern GPU architectures, including the characteristics of different memory types (HBM, L2 cache, shared memory, registers), their access patterns, bandwidth limitations, and the impact on inference performance. Provide an in-depth explanation of KV cache management systems, starting with the mathematical analysis of memory requirements for different sequence lengths and batch sizes. Detail the implementation of paged KV cache systems, including the algorithms for optimal page size selection, the data structures used for efficient page management, and the techniques for minimizing memory fragmentation. Explain the mathematical models for predicting memory usage patterns and the algorithms for dynamic memory allocation and deallocation. Analyze advanced memory optimization techniques including memory pooling, prefix caching, and shared memory systems. Explain the implementation of memory pools for different data types and access patterns, including the algorithms for pool size optimization and the techniques for reducing memory allocation overhead. Detail the mathematical foundations of prefix caching, including the algorithms for identifying reusable prefixes, the data structures for efficient prefix storage and retrieval, and the techniques for maintaining cache coherency in multi-request environments. Discuss the implementation of weight streaming and offloading techniques, including the algorithms for determining optimal offloading strategies, the techniques for overlapping computation and data transfer, and the mathematical models for predicting the performance impact of different offloading configurations. Explain the role of CPU memory and storage systems in supporting large model inference, including the implementation of efficient data pipelines and the optimization of memory bandwidth utilization. Examine the integration of memory optimization techniques with quantization and compression methods, including the implementation of compressed KV cache systems, the algorithms for on-the-fly decompression, and the techniques for maintaining numerical accuracy while reducing memory usage. Analyze the performance implications of different memory optimization strategies across various hardware platforms and deployment scenarios, providing detailed guidelines for selecting optimal memory management configurations for specific use cases and resource constraints.
        Please Write Eassy.
        '''
        ] * 12
    
    
    print("vLLM Qwen2.5-1.5B-Instruct TTFT & TPS Measurement (FP8 Verification Version)")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
    # Additional warning for FP8 requests
    if args.quantization == "fp8":
        print("FP8 quantization requested. Verifying actual FP8 native operation.")
    
    print("Accurate TTFT measurement mode (with optimizations)")
    results = run_streaming_test(
        model_path=args.model_path,
        prompts=test_prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=not args.disable_prefix_caching,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        use_v2_block_manager=not args.disable_v2_block_manager,
        kv_cache_dtype=args.kv_cache_dtype,
        quantization=args.quantization,
        speculative_model=args.speculative_model
    )
    
    # Save results to file
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output_file}.")

if __name__ == "__main__":
    main() 