#!/usr/bin/env python3
"""
Optimized model loader for Higgs Audio TTS with memory management
"""
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from memory_config import setup_memory_optimization, get_quantization_config, clear_memory

class OptimizedHiggsAudioLoader:
    def __init__(self, model_path="bosonai/higgs-audio-v2-generation-3B-base", 
                 tokenizer_path="bosonai/higgs-audio-v2-tokenizer"):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.tokenizer = None
        
        # Setup memory optimization
        setup_memory_optimization()
    
    def load_model(self, force_quantization=None):
        """Load model with optimal memory configuration"""
        
        print("🚀 Loading Higgs Audio model with memory optimization...")
        
        # Clear any existing memory
        clear_memory()
        gc.collect()
        
        # Get quantization config
        if force_quantization:
            if force_quantization == "4bit":
                config = {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4",
                    "device_map": "auto",
                    "torch_dtype": torch.bfloat16,
                    "low_cpu_mem_usage": True,
                }
            elif force_quantization == "8bit":
                config = {
                    "load_in_8bit": True,
                    "device_map": "auto",
                    "torch_dtype": torch.bfloat16,
                    "low_cpu_mem_usage": True,
                }
            else:
                config = get_quantization_config()
        else:
            config = get_quantization_config()
        
        try:
            # Load model with quantization
            print(f"📦 Loading model from {self.model_path}")
            print(f"🔧 Config: {config}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **config
            )
            
            # Load tokenizer
            print(f"📝 Loading tokenizer from {self.tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            
            print("✅ Model and tokenizer loaded successfully!")
            
            # Print memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"💾 GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
            
            return self.model, self.tokenizer
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ CUDA OOM Error: {e}")
            print("🔄 Trying with more aggressive quantization...")
            
            # Clear memory and try 4-bit quantization
            clear_memory()
            gc.collect()
            
            config_4bit = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "device_map": "auto",
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
            }
            
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **config_4bit
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
                print("✅ Model loaded with 4-bit quantization!")
                return self.model, self.tokenizer
                
            except Exception as e2:
                print(f"❌ Failed to load with 4-bit quantization: {e2}")
                raise e2
    
    def generate_optimized(self, inputs, **generation_kwargs):
        """Generate with memory optimization"""
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Clear cache before generation
        clear_memory()
        
        # Use no_grad context to save memory
        with torch.no_grad():
            # Set default generation parameters for memory efficiency
            default_kwargs = {
                "max_new_tokens": 512,  # Limit output length
                "do_sample": True,
                "temperature": 0.7,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            default_kwargs.update(generation_kwargs)
            
            try:
                outputs = self.model.generate(inputs, **default_kwargs)
                return outputs
                
            except torch.cuda.OutOfMemoryError:
                print("⚠️ OOM during generation, clearing cache and retrying...")
                clear_memory()
                
                # Retry with smaller max_new_tokens
                default_kwargs["max_new_tokens"] = min(256, default_kwargs.get("max_new_tokens", 512))
                outputs = self.model.generate(inputs, **default_kwargs)
                return outputs

if __name__ == "__main__":
    # Test the loader
    loader = OptimizedHiggsAudioLoader()
    model, tokenizer = loader.load_model()
    print("Model loading test completed!")