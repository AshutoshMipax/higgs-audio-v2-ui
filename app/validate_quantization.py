#!/usr/bin/env python3
"""
Validate quantization setup and bitsandbytes integration
"""
import torch
from loguru import logger

def validate_quantization():
    """Validate that quantization libraries are working correctly"""
    
    print("🔍 Validating quantization setup...")
    
    # Test 1: Import bitsandbytes
    try:
        import bitsandbytes as bnb
        print(f"✅ bitsandbytes imported successfully (version: {bnb.__version__})")
    except ImportError as e:
        print(f"❌ Failed to import bitsandbytes: {e}")
        return False
    
    # Test 2: Import BitsAndBytesConfig
    try:
        from transformers import BitsAndBytesConfig
        print("✅ BitsAndBytesConfig imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import BitsAndBytesConfig: {e}")
        return False
    
    # Test 3: Create quantization configs
    try:
        # Test 4-bit config
        config_4bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print("✅ 4-bit quantization config created successfully")
        
        # Test 8-bit config
        config_8bit = BitsAndBytesConfig(load_in_8bit=True)
        print("✅ 8-bit quantization config created successfully")
        
    except Exception as e:
        print(f"❌ Failed to create quantization configs: {e}")
        return False
    
    # Test 4: Check CUDA availability for quantization
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Test CUDA compute capability (required for some quantization features)
        compute_capability = torch.cuda.get_device_capability()
        print(f"🔧 CUDA Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
        
        if compute_capability[0] >= 7:  # Volta or newer
            print("✅ GPU supports modern quantization features")
        else:
            print("⚠️ GPU may have limited quantization support (older architecture)")
    else:
        print("⚠️ CUDA not available - quantization will fall back to CPU")
    
    # Test 5: Test memory optimization imports
    try:
        from memory_config import setup_memory_optimization, get_quantization_config, clear_memory
        print("✅ Memory optimization modules imported successfully")
        
        # Test memory setup
        setup_memory_optimization()
        print("✅ Memory optimization setup completed")
        
        # Test quantization config generation
        config = get_quantization_config()
        print(f"✅ Quantization config generated: {list(config.keys())}")
        
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
        return False
    
    print("🎉 All quantization validation tests passed!")
    return True

if __name__ == "__main__":
    success = validate_quantization()
    if success:
        print("\n✅ Quantization setup is ready!")
        print("You can now use 4-bit and 8-bit quantization modes.")
    else:
        print("\n❌ Quantization setup has issues.")
        print("Please check the error messages above.")
    
    exit(0 if success else 1)