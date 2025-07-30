#!/usr/bin/env python3
"""
Test script to verify all dependencies are installed correctly
"""

def test_imports():
    """Test all critical imports"""
    try:
        print("Testing imports...")
        
        # Core dependencies
        import torch
        print("✓ torch")
        
        import torchaudio
        print("✓ torchaudio")
        
        import transformers
        print("✓ transformers")
        
        import gradio
        print("✓ gradio")
        
        # Critical missing dependency
        import vector_quantize_pytorch
        print("✓ vector_quantize_pytorch")
        
        # Other dependencies
        import dacite
        print("✓ dacite")
        
        import pydantic
        print("✓ pydantic")
        
        import omegaconf
        print("✓ omegaconf")
        
        import einops
        print("✓ einops")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if not success:
        print("\n❌ Some dependencies are missing. Please run the install process again.")
        exit(1)
    else:
        print("\n✅ All dependencies are properly installed!")