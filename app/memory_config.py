#!/usr/bin/env python3
"""
Memory optimization configuration for Higgs Audio TTS
Handles CUDA memory management and quantization settings
"""
import os
import torch
from typing import Dict, Any

def setup_memory_optimization():
    """Configure PyTorch for optimal memory usage"""
    
    # Enable expandable segments to reduce fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Additional memory optimizations
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA operations
    
    # Clear any existing cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("✅ Memory optimization configured")
    print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")

def get_quantization_config(gpu_memory_gb: float = None) -> Dict[str, Any]:
    """Get quantization configuration based on available GPU memory"""
    
    if gpu_memory_gb is None and torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    if gpu_memory_gb is None:
        gpu_memory_gb = 16  # Default assumption
    
    print(f"GPU Memory: {gpu_memory_gb:.1f} GB")
    
    # Configure quantization based on available memory
    if gpu_memory_gb < 10:
        # Use 4-bit quantization for GPUs with less than 10GB
        config = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
        print("🔧 Using 4-bit quantization (aggressive memory saving)")
    elif gpu_memory_gb < 14:
        # Use 6-bit quantization for GPUs with 10-14GB
        config = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "fp4",
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
        print("🔧 Using 6-bit quantization (4-bit FP4 with float16)")
    elif gpu_memory_gb < 20:
        # Use 8-bit quantization for GPUs with 14-20GB
        config = {
            "load_in_8bit": True,
            "device_map": "auto", 
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
        print("🔧 Using 8-bit quantization (balanced)")
    else:
        # Full precision for high-memory GPUs
        config = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
        print("🔧 Using full precision (high memory)")
    
    return config

def optimize_model_loading():
    """Apply model loading optimizations"""
    
    # Set memory fraction to leave some headroom
    if torch.cuda.is_available():
        # Reserve 10% of GPU memory for operations
        total_memory = torch.cuda.get_device_properties(0).total_memory
        memory_fraction = 0.9
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        print(f"🔧 Set GPU memory fraction to {memory_fraction}")

def clear_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("🧹 GPU memory cache cleared")

if __name__ == "__main__":
    setup_memory_optimization()
    config = get_quantization_config()
    print("Quantization config:", config)