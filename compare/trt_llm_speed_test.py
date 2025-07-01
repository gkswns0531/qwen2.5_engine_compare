#!/usr/bin/env python3
"""
Native TensorRT-LLM Qwen2.5-1.5B-Instruct model TTFT & TPS measurement script
(pip install version, no Docker required)
"""

import time
import argparse
import json
import os
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from pathlib import Path

# TensorRT-LLM related imports (pip version)
import tensorrt_llm
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.runtime import ModelRunner, ModelRunnerCpp, SamplingConfig
from transformers import AutoTokenizer
HAS_TENSORRT_LLM = True
print(f"TensorRT-LLM version: {tensorrt_llm.__version__}")

def format_qwen_message(prompt: str) -> str:
    """Format message for Qwen2.5 model"""
    return f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

class NativeTensorRTLLMRunner:
    """Class to run native TensorRT-LLM engine"""
    
    def __init__(self, engine_dir: str, tokenizer_dir: Optional[str] = None):
        self.engine_dir = Path(engine_dir)
        self.tokenizer_dir = tokenizer_dir or engine_dir
        
        # Load config file
        config_path = self.engine_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        print(f"TensorRT-LLM engine configuration:")
        print(f"   - Version: {self.config['version']}")
        print(f"   - Model: {self.config['pretrained_config']['architecture']}")
        print(f"   - Max input length: {self.config['build_config']['max_input_len']}")
        print(f"   - Max sequence length: {self.config['build_config']['max_seq_len']}")
        print(f"   - Max batch size: {self.config['build_config']['max_batch_size']}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir, 
            trust_remote_code=True,
            use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Tokenizer loading completed")
        
        # Initialize TensorRT-LLM engine
        self._init_engine()
    
    def _init_engine(self):
        """Initialize TensorRT-LLM engine"""
        # Direct initialization with LLM class
        print("Initializing TensorRT-LLM engine...")
        self.llm = LLM(
            model=str(self.engine_dir),
            tokenizer=str(self.tokenizer_dir),  # Explicitly pass tokenizer path
            trust_remote_code=True,
            enable_chunked_context=False,
            enable_trt_overlap=True,
            normalize_log_probs=True
        )
        print("TensorRT-LLM engine initialization completed")
    
    def generate(self, 
                prompts: List[str], 
                max_new_tokens: int = 1024,
                temperature: float = 0.0,
                top_p: float = 1.0,
                top_k: int = 1) -> List[Dict[str, Any]]:
        """Text generation"""
        
        results = []
        
        # Set sampling parameters (unified with other engines)
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 0.01,  # Can't use 0, so use small value
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.05,  # Unified with others
            length_penalty=1.0,
            end_id=151645,  # Explicitly set Qwen2.5 EOS token ID
            pad_id=151643   # Explicitly set Qwen2.5 PAD token ID
        )
        
        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i+1}/{len(prompts)}...")
            
            formatted_prompt = format_qwen_message(prompt)
            
            # 🔧 UNIFIED TTFT MEASUREMENT: 정확한 TTFT 측정
            ttft_sampling_params = SamplingParams(
                max_tokens=1,  # Only first token
                temperature=temperature if temperature > 0 else 0.01,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=1.05,  # Unified with others
                length_penalty=1.0,
                end_id=151645,
                pad_id=151643
            )
            
            ttft_start = time.time()
            first_outputs = self.llm.generate([formatted_prompt], ttft_sampling_params)
            ttft_end = time.time()
            actual_ttft = (ttft_end - ttft_start) * 1000  # ms
            
            request_start_time = time.time()
            full_outputs = self.llm.generate([formatted_prompt], sampling_params)
            request_end_time = time.time()
            
            output = full_outputs[0]
            input_tokens = len(output.prompt_token_ids)
            output_tokens = len(output.outputs[0].token_ids)
            generated_text = output.outputs[0].text
            
            # Total time and TPS calculation (same as vLLM and PyTorch Backend)
            total_time = request_end_time - request_start_time
            tps = output_tokens / total_time if total_time > 0 else 0
            
            result = {
                'prompt_index': i,
                'original_prompt': prompt,
                'generated_text': generated_text,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'ttft_ms': actual_ttft,  # Unified TTFT measurement
                'tps': tps,  # Unified TPS measurement
                'total_time': total_time,
                'start_time': request_start_time,
                'end_time': request_end_time
            }
            
            results.append(result)
        
        return results

def run_native_trt_llm_speed_test(
    engine_dir: str,
    tokenizer_dir: Optional[str] = None,
    prompts: List[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 1
):
    """Run native TensorRT-LLM speed test"""
    
    if not HAS_TENSORRT_LLM:
        print("TensorRT-LLM is not installed.")
        return None
    
    print(f"Loading native TensorRT-LLM engine: {engine_dir}")
    print(f"Test settings:")
    print(f"   - Max tokens: {max_tokens}")
    print(f"   - Temperature: {temperature}")
    print(f"   - Top-p: {top_p}")
    print(f"   - Top-k: {top_k}")
    print("-" * 50)
    
    # Initialize TensorRT-LLM runner
    start_time = time.time()
    runner = NativeTensorRTLLMRunner(engine_dir, tokenizer_dir)
    load_time = time.time() - start_time
    print(f"Model loading completed ({load_time:.2f}s)")
    
    # Add warm-up step (simple and fast)
    print("Running warm-up inference...")
    warmup_prompt = "Hello"
    warmup_formatted = format_qwen_message(warmup_prompt)
    warmup_sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.01,
        top_p=1.0,
        top_k=1,
        repetition_penalty=1.05,  
        length_penalty=1.0,
        end_id=151645,
        pad_id=151643
    )
    
    warmup_start = time.time()
    warmup_outputs = runner.llm.generate([warmup_formatted], warmup_sampling_params)
    warmup_end = time.time()
    print(f"Warm-up completed ({warmup_end - warmup_start:.2f}s)")
    
    print("-" * 50)
    
    print(f"Starting accurate TTFT & TPS measurement ({len(prompts)} prompts)")
    
    # Execute generation
    results = runner.generate(
        prompts=prompts,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )
    
    # Analyze results
    # Exclude first 2 results for warmup, use remaining 10 for statistics
    warm_results = results[2:] if len(results) > 2 else results
    
    total_input_tokens = sum(r['input_tokens'] for r in results)  # Total for all results
    total_output_tokens = sum(r['output_tokens'] for r in results)  # Total for all results
    avg_ttft = sum(r['ttft_ms'] for r in warm_results) / len(warm_results) if warm_results else 0
    valid_tps_results = [r['tps'] for r in warm_results if r['tps'] > 0]
    avg_tps = sum(valid_tps_results) / len(valid_tps_results) if valid_tps_results else 0
    
    # Output results
    print("=" * 70)
    print("TensorRT-LLM Accurate TTFT & TPS Measurement Results")
    print("=" * 70)
    print(f"Total input tokens: {total_input_tokens:,}")
    print(f"Total output tokens: {total_output_tokens:,}")
    print(f"Average TTFT: {avg_ttft:.1f}ms")
    print(f"Average TPS: {avg_tps:.1f} tokens/s")
    print("=" * 70)
    
    # Detailed results for each prompt
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
        "engine_type": "TensorRT-LLM",  # For engine type identification
        "engine_dir": engine_dir,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "avg_ttft_ms": avg_ttft,
        "avg_tps": avg_tps,
        "load_time": load_time,
        "results": results
    }

def main():
    parser = argparse.ArgumentParser(description="Native TensorRT-LLM Qwen2.5-1.5B TTFT & TPS measurement")
    parser.add_argument("--engine_dir", type=str, 
                       default="../qwen2.5-1.5b-instruct_engine_fp16",
                       help="TensorRT-LLM engine directory path (default: FP16 engine)")
    parser.add_argument("--tokenizer_dir", type=str, 
                       default="../qwen2.5-1.5b-instruct",
                       help="Tokenizer directory path (default: original model directory)")
    parser.add_argument("--max_tokens", type=int, default=1024,
                       help="Maximum number of tokens to generate (default: 1024)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature (default: 0.0)")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="Top-p sampling (default: 1.0)")
    parser.add_argument("--top_k", type=int, default=1,
                       help="Top-k sampling (default: 1)")
    parser.add_argument("--custom_prompts", type=str, nargs="+",
                       help="Custom prompts")
    parser.add_argument("--n_samples", type=int, default=12,
                       help="Number of test samples to run (default: 12)")
    parser.add_argument("--output_file", type=str, default=None,
                       help="JSON file path to save results")
    
    args = parser.parse_args()
    
    # Prepare test prompts (optimized for TensorRT-LLM performance)
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
        '''
        ] * args.n_samples
    
    
    print("Native TensorRT-LLM Qwen2.5-1.5B-Instruct TTFT & TPS Measurement")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Run speed test
    results = run_native_trt_llm_speed_test(
        engine_dir=args.engine_dir,
        tokenizer_dir=args.tokenizer_dir,
        prompts=test_prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
    
    if results and args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output_file}.")

if __name__ == "__main__":
    main() 