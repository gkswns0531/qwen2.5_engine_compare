#!/usr/bin/env python3
"""
3-Way Performance Comparison: vLLM vs TensorRT-LLM (TensorRT) vs TensorRT-LLM (PyTorch)
"""

import argparse
import subprocess
import json
import time
from pathlib import Path

def run_vllm_test(model_path: str, max_tokens: int, temperature: float, top_p: float, top_k: int):
    """Run vLLM test"""
    print("Starting vLLM test...")
    
    cmd = [
        "python3", "vllm_speed_test.py",
        "--model_path", model_path,
        "--max_tokens", str(max_tokens),
        "--temperature", str(temperature),
        "--top_p", str(top_p),
        "--top_k", str(top_k),
        "--output_file", "vllm_results.json",
        "--quantization", "fp8"
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    print("vLLM execution log:")
    if result.stdout:
        print(result.stdout[-500:])
    
    if result.returncode == 0 and Path("vllm_results.json").exists():
        with open("vllm_results.json", 'r') as f:
            vllm_results = json.load(f)
        vllm_results['total_test_time'] = end_time - start_time
        vllm_results['engine_type'] = 'vLLM'
        print("vLLM test completed successfully")
        return vllm_results
    else:
        print(f"vLLM test failed:")
        print(f"   Return code: {result.returncode}")
        if result.stderr:
            print(f"   Error: {result.stderr[-500:]}")
        return None

def run_tensorrt_llm_test(engine_dir: str, tokenizer_dir: str, max_tokens: int, temperature: float, top_p: float, top_k: int):
    """Run TensorRT-LLM TensorRT backend test"""
    print("Starting TensorRT-LLM (TensorRT backend) test...")
    
    cmd = [
        "python3", "trt_llm_speed_test.py",
        "--engine_dir", engine_dir,
        "--tokenizer_dir", tokenizer_dir,
        "--max_tokens", str(max_tokens),
        "--temperature", str(temperature),
        "--top_p", str(top_p),
        "--top_k", str(top_k),
        "--output_file", "trt_llm_results.json"
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    print("TensorRT-LLM execution log:")
    if result.stdout:
        print(result.stdout[-500:])
    
    if result.returncode == 0 and Path("trt_llm_results.json").exists():
        with open("trt_llm_results.json", 'r') as f:
            trt_results = json.load(f)
        trt_results['total_test_time'] = end_time - start_time
        trt_results['engine_type'] = 'TensorRT-LLM (TensorRT)'
        print("TensorRT-LLM test completed successfully")
        return trt_results
    else:
        print(f"TensorRT-LLM test failed:")
        print(f"   Return code: {result.returncode}")
        if result.stderr:
            print(f"   Error: {result.stderr[-500:]}")
        return None

def run_pytorch_backend_test(model_path: str, max_tokens: int, temperature: float, top_p: float, top_k: int):
    """Run TensorRT-LLM PyTorch backend test"""
    print("Starting TensorRT-LLM (PyTorch backend) test...")
    
    cmd = [
        "python3", "pytorch_speed_test.py",
        "--model-path", model_path,
        "--max-tokens", "1024",
        "--temperature", "0.0",
        "--top-p", "1.0",
        "--top-k", "1",
        "--tensor-parallel-size", "1",
        "--pipeline-parallel-size", "1",
        "--max-batch-size", "16",
        "--max-num-tokens", "8192",
        "--kv-cache-fraction", "0.6",
        "--attn-backend", "TRTLLM",
        "--quantization", "fp8",
        "--use-cuda-graph",
        "--enable-trtllm-sampler",
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--block-size", "16",
        "--gpu-memory-utilization", "0.85",
        "--swap-space", "4",
        "--output-file", "pytorch_results.json"
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    print("PyTorch backend execution log:")
    if result.stdout:
        print(result.stdout[-500:])
    
    if result.returncode == 0 and Path("pytorch_results.json").exists():
        with open("pytorch_results.json", 'r') as f:
            pytorch_results = json.load(f)
        pytorch_results['total_test_time'] = end_time - start_time
        pytorch_results['engine_type'] = 'TensorRT-LLM (PyTorch Backend)'
        print("PyTorch backend test completed successfully")
        return pytorch_results
    else:
        print(f"PyTorch backend test failed:")
        print(f"   Return code: {result.returncode}")
        if result.stderr:
            print(f"   Error: {result.stderr[-500:]}")
        return None

def compare_three_engines(vllm_results, trt_results, pytorch_results):
    """Compare results from all three engines"""
    
    print("\n" + "=" * 100)
    print("3-WAY PERFORMANCE COMPARISON RESULTS")
    print("   vLLM vs TensorRT-LLM (TensorRT) vs TensorRT-LLM (PyTorch)")
    print("=" * 100)
    
    engines = [
        ("vLLM", vllm_results),
        ("TensorRT-LLM (TensorRT)", trt_results), 
        ("TensorRT-LLM (PyTorch)", pytorch_results)
    ]
    
    # Basic information
    print(f"Test Overview:")
    print(f"   - Total prompts: {len(vllm_results['results'])} prompts")
    
    for name, results in engines:
        print(f"   - {name}: {results['total_input_tokens']:,} input tokens, {results['total_output_tokens']:,} output tokens")
    
    # Loading time comparison
    print(f"\nLoading Time:")
    load_times = [(name, results['load_time']) for name, results in engines]
    load_times.sort(key=lambda x: x[1])
    
    for i, (name, load_time) in enumerate(load_times):
        if i == 0:
            print(f"   1st {name}: {load_time:.3f}s (fastest)")
        else:
            ratio = load_time / load_times[0][1]
            print(f"   {i+1}th {name}: {load_time:.3f}s ({ratio:.2f}x slower)")
    
    # Calculate average performance after warmup (excluding first 2 results)
    engine_stats = {}
    for name, results in engines:
        warm_results = results['results'][2:] if len(results['results']) > 2 else results['results']
        
        ttft_list = [r['ttft_ms'] for r in warm_results]
        tps_list = [r['tps'] for r in warm_results if r['tps'] > 0]
        
        engine_stats[name] = {
            'avg_ttft': sum(ttft_list) / len(ttft_list) if ttft_list else 0,
            'avg_tps': sum(tps_list) / len(tps_list) if tps_list else 0
        }
    
    # TTFT comparison
    print(f"\nTTFT (Time To First Token) - Average:")
    ttft_ranking = sorted(engine_stats.items(), key=lambda x: x[1]['avg_ttft'])
    
    for i, (name, stats) in enumerate(ttft_ranking):
        if i == 0:
            print(f"   1st {name}: {stats['avg_ttft']:.1f}ms (fastest)")
        else:
            ratio = stats['avg_ttft'] / ttft_ranking[0][1]['avg_ttft']
            print(f"   {i+1}th {name}: {stats['avg_ttft']:.1f}ms ({ratio:.2f}x slower)")
    
    # TPS comparison
    print(f"\nTPS (Tokens Per Second) - Average:")
    tps_ranking = sorted(engine_stats.items(), key=lambda x: x[1]['avg_tps'], reverse=True)
    
    for i, (name, stats) in enumerate(tps_ranking):
        if i == 0:
            print(f"   1st {name}: {stats['avg_tps']:.1f} tokens/s (fastest)")
        else:
            ratio = tps_ranking[0][1]['avg_tps'] / stats['avg_tps']
            print(f"   {i+1}th {name}: {stats['avg_tps']:.1f} tokens/s ({ratio:.2f}x slower)")
    
    # Total test time
    print(f"\nTotal Test Time:")
    total_times = [(name, results['total_test_time']) for name, results in engines]
    total_times.sort(key=lambda x: x[1])
    
    for i, (name, total_time) in enumerate(total_times):
        if i == 0:
            print(f"   1st {name}: {total_time:.1f}s (fastest)")
        else:
            ratio = total_time / total_times[0][1]
            print(f"   {i+1}th {name}: {total_time:.1f}s ({ratio:.2f}x slower)")
    
    # Detailed comparison table
    print(f"\nDetailed Comparison by Prompt:")
    print("-" * 90)
    print(f"{'Prompt':<8} {'vLLM TTFT':<12} {'TRT TTFT':<12} {'PyTorch TTFT':<14} {'vLLM TPS':<10} {'TRT TPS':<10} {'PyTorch TPS':<12}")
    print("-" * 90)
    
    for i in range(2, len(vllm_results['results'])):  
        vllm_r = vllm_results['results'][i]
        trt_r = trt_results['results'][i]
        pytorch_r = pytorch_results['results'][i]
        
        print(f"Prompt {i:<2} {vllm_r['ttft_ms']:>7.1f}ms {trt_r['ttft_ms']:>8.1f}ms {pytorch_r['ttft_ms']:>10.1f}ms "
              f"{vllm_r['tps']:>6.1f}t/s {trt_r['tps']:>6.1f}t/s {pytorch_r['tps']:>8.1f}t/s")
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY:")
    
    load_winner = load_times[0][0]
    ttft_winner = ttft_ranking[0][0]
    tps_winner = tps_ranking[0][0]
    total_time_winner = total_times[0][0]
    
    print(f"   - Loading Speed: {load_winner}")
    print(f"   - Response Speed (TTFT): {ttft_winner}")
    print(f"   - Generation Speed (TPS): {tps_winner}")
    print(f"   - Overall Test Speed: {total_time_winner}")
    
    # Score calculation (1st=3 points, 2nd=2 points, 3rd=1 point)
    scores = {name: 0 for name, _ in engines}
    
    categories = [
        ("Loading", load_times),
        ("TTFT", ttft_ranking),
        ("TPS", list(reversed(tps_ranking))),  # TPS higher is better so reverse
        ("Total Time", total_times)
    ]
    
    for category, ranking in categories:
        for i, (name, _) in enumerate(ranking):
            scores[name] += 3 - i
    
    print(f"\nOverall Ranking (higher score = better):")
    final_ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for i, (name, score) in enumerate(final_ranking):
        rank = ["1st", "2nd", "3rd"][i]
        print(f"   {rank} {name}: {score} points")
    
    # Save results
    comparison = {
        "engines": {
            "vllm": vllm_results,
            "tensorrt_llm": trt_results,
            "pytorch_backend": pytorch_results
        },
        "comparison": {
            "load_winner": load_winner,
            "ttft_winner": ttft_winner,
            "tps_winner": tps_winner,
            "total_time_winner": total_time_winner,
            "final_ranking": final_ranking,
            "engine_stats": engine_stats
        }
    }
    
    with open("3way_comparison_results.json", 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed comparison results saved to 3way_comparison_results.json")
    print("=" * 100)

def main():
    parser = argparse.ArgumentParser(description="3-way performance comparison: vLLM vs TensorRT-LLM (TensorRT) vs TensorRT-LLM (PyTorch)")
    parser.add_argument("--vllm_model", type=str, default="./qwen2.5-1.5b-instruct",
                       help="vLLM model path")
    parser.add_argument("--trt_engine", type=str, default="./qwen2.5-1.5b-instruct_engine_fp8",
                       help="TensorRT-LLM engine path")
    parser.add_argument("--trt_tokenizer", type=str, default="./qwen2.5-1.5b-instruct",
                       help="TensorRT-LLM tokenizer path")
    parser.add_argument("--pytorch_model", type=str, default="./qwen2.5-1.5b-instruct",
                       help="TensorRT-LLM PyTorch backend model path")
    parser.add_argument("--max_tokens", type=int, default=1024,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=1,
                       help="Top-k sampling")
    parser.add_argument("--skip_vllm", action="store_true",
                       help="Skip vLLM test")
    parser.add_argument("--skip_tensorrt", action="store_true", 
                       help="Skip TensorRT-LLM TensorRT backend test")
    parser.add_argument("--skip_pytorch", action="store_true",
                       help="Skip TensorRT-LLM PyTorch backend test")
    
    args = parser.parse_args()
    
    print("3-Way Performance Comparison Started!")
    print(f"Test settings:")
    print(f"   - Max tokens: {args.max_tokens}")
    print(f"   - Temperature: {args.temperature}")
    print(f"   - Top-p: {args.top_p}")
    print(f"   - Top-k: {args.top_k}")
    print("=" * 70)
    
    results = {}
    
    # vLLM test
    if not args.skip_vllm:
        vllm_results = run_vllm_test(
            args.vllm_model, args.max_tokens, args.temperature, args.top_p, args.top_k
        )
        if vllm_results:
            results['vllm'] = vllm_results
        else:
            print("vLLM test failed, excluding from comparison.")
    
    print("\n" + "-" * 70)
    
    # TensorRT-LLM TensorRT backend test
    if not args.skip_tensorrt:
        trt_results = run_tensorrt_llm_test(
            args.trt_engine, args.trt_tokenizer, args.max_tokens, args.temperature, args.top_p, args.top_k
        )
        if trt_results:
            results['tensorrt'] = trt_results
        else:
            print("TensorRT-LLM test failed, excluding from comparison.")
    
    print("\n" + "-" * 70)
    
    # TensorRT-LLM PyTorch backend test
    if not args.skip_pytorch:
        pytorch_results = run_pytorch_backend_test(
            args.pytorch_model, args.max_tokens, args.temperature, args.top_p, args.top_k
        )
        if pytorch_results:
            results['pytorch'] = pytorch_results
        else:
            print("PyTorch backend test failed, excluding from comparison.")
    
    # Compare results
    if len(results) >= 2:
        if len(results) == 3:
            compare_three_engines(results['vllm'], results['tensorrt'], results['pytorch'])
        else:
            available_engines = list(results.keys())
            print(f"\nOnly {len(results)} engines succeeded: {', '.join(available_engines)}")
            print("Need at least 3 engines for full comparison.")
    else:
        print("Too few engines succeeded for comparison.")

if __name__ == "__main__":
    main() 