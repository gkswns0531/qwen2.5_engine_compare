# TensorRT-LLM Qwen Model Deployment & Performance Testing

Performance comparison across different inference engines.

## Project Structure

```
./
├── download_hf_model.py                        # Download HuggingFace models
├── convert_qwen_trtllm.py                      # Convert to TensorRT-LLM checkpoints  
├── build_qwen_engine.py                        # Build TensorRT-LLM engines
├── qwen2.5-1.5b-instruct/                      # Downloaded HuggingFace model
├── qwen2.5-1.5b-instruct_checkpoints_fp16/     # FP16 TensorRT-LLM checkpoints
├── qwen2.5-1.5b-instruct_checkpoints_fp8/      # FP8 TensorRT-LLM checkpoints
├── qwen2.5-1.5b-instruct_engine_fp16/          # FP16 TensorRT-LLM engine
├── qwen2.5-1.5b-instruct_engine_fp8/           # FP8 TensorRT-LLM engine
├── compare/                                    # Performance testing scripts
│   ├── vllm_speed_test.py                      # vLLM TTFT & TPS measurement
│   ├── trt_llm_speed_test.py                   # TensorRT-LLM performance test
│   ├── pytorch_speed_test.py                   # TensorRT-LLM PyTorch backend test
│   └── compare_3_engines.py                    # Compare all 3 engines at once
└── README.md
```

## Version Requirements

**Tested Environment:**
- **Python**: 3.10.12
- **TensorRT-LLM**: 0.20.0
- **TensorRT**: 10.10.0.31
- **PyTorch**: 2.7.0
- **vLLM**: 0.9.1
- **CUDA Driver**: 12.8
- **Transformers**: 4.51.1
- **HuggingFace Hub**: 0.33.0

**Docker Alternative:**
- **TensorRT-LLM Docker**: `nvcr.io/nvidia/tensorrt-llm/release:0.21.0rc1`

## Quick Start

### 1. Setup
```bash
pip install huggingface_hub tensorrt-llm vllm
git clone https://github.com/NVIDIA/TensorRT-LLM.git
```

### 2. Model Deployment (3 steps)
```bash
# Step 1: Download model
python3 download_hf_model.py Qwen/Qwen2.5-1.5B-Instruct

# Step 2: Convert to checkpoint
python3 convert_qwen_trtllm.py --quantization fp16  # or fp8

# Step 3: Build engine
python3 build_qwen_engine.py --quantization fp16   # or fp8
```

### 3. Performance Testing
```bash
# Test individual engines
cd compare
python3 vllm_speed_test.py --model_path ../qwen2.5-1.5b-instruct
python3 trt_llm_speed_test.py --engine_dir ../qwen2.5-1.5b-instruct_engine_fp16
python3 pytorch_speed_test.py --model-path ../qwen2.5-1.5b-instruct

# Compare all 3 engines at once
python3 compare_3_engines.py
```

## 📋 Scripts Overview

### Core Deployment Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `download_hf_model.py` | Download HuggingFace models | `python3 download_hf_model.py Qwen/Qwen2.5-1.5B-Instruct` |
| `convert_qwen_trtllm.py` | Convert to TensorRT-LLM format | `python3 convert_qwen_trtllm.py --quantization fp16` |
| `build_qwen_engine.py` | Build optimized engines | `python3 build_qwen_engine.py --quantization fp16` |

### Performance Testing Scripts

| Script | Engine Type | Purpose |
|--------|-------------|---------|
| `vllm_speed_test.py` | vLLM | TTFT & TPS measurement with FP8 support |
| `trt_llm_speed_test.py` | TensorRT-LLM Native | Native engine performance testing |
| `pytorch_speed_test.py` | TensorRT-LLM PyTorch | PyTorch backend performance testing |
| `compare_3_engines.py` | All 3 Engines | Unified comparison of all engines |

## 📊 Output Structure

After deployment:
```
./
├── qwen2.5-1.5b-instruct/                    # Downloaded model
├── qwen2.5-1.5b-instruct_checkpoints_fp16/   # TensorRT-LLM checkpoint
├── qwen2.5-1.5b-instruct_engine_fp16/        # Ready-to-use engine
└── TensorRT-LLM/                             # Required for conversion
```

## ⚡ Performance Testing Details

### Test Metrics
- **TTFT (Time to First Token)**: Latency measurement using `max_tokens=1`
- **TPS (Tokens Per Second)**: Throughput calculation (output_tokens / total_time)
- **Memory Usage**: GPU memory allocation tracking

### Engine Comparison
- **vLLM**: Continuous batching, PagedAttention, FP8 native support
- **TensorRT-LLM Native**: Maximum optimization, compiled engines
- **PyTorch Backend**: Flexible deployment, easier integration

## Common Usage Patterns

```bash
# Quick FP16 deployment
python3 download_hf_model.py Qwen/Qwen2.5-1.5B-Instruct
python3 convert_qwen_trtllm.py --quantization fp16
python3 build_qwen_engine.py --quantization fp16

# FP8 ultra-optimized deployment  
python3 convert_qwen_trtllm.py --quantization fp8
python3 build_qwen_engine.py --quantization fp8

# Performance comparison
cd compare
python3 compare_3_engines.py  # Compare all engines
# or test individually:
python3 vllm_speed_test.py --quantization fp8
python3 trt_llm_speed_test.py --engine_dir ../qwen2.5-1.5b-instruct_engine_fp8
python3 pytorch_speed_test.py --quantization fp8
```

---