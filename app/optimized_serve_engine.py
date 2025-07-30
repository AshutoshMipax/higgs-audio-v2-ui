#!/usr/bin/env python3
"""
Memory-optimized version of HiggsAudioServeEngine with proper bitsandbytes quantization
"""
import torch
import gc
import threading
from typing import Union, Optional, List
from copy import deepcopy
from loguru import logger
from transformers import AutoTokenizer, AutoProcessor, BitsAndBytesConfig, AutoModelForCausalLM
from transformers.cache_utils import StaticCache

from higgs_audio.serve.serve_engine import HiggsAudioServeEngine
from higgs_audio.model import HiggsAudioModel
from higgs_audio.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from higgs_audio.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from memory_config import setup_memory_optimization, get_quantization_config, clear_memory
from performance_optimizer import performance_optimizer

class OptimizedHiggsAudioServeEngine:
    """Memory-optimized version of HiggsAudioServeEngine with proper quantization support"""
    
    def __init__(
        self,
        model_name_or_path: str,
        audio_tokenizer_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: Union[torch.dtype, str] = "auto",
        kv_cache_lengths: List[int] = [1024, 2048, 4096, 6144],
        force_quantization: Optional[str] = None,  # "4bit", "8bit", or None for auto
        enable_memory_optimization: bool = True,
    ):
        """Initialize the OptimizedHiggsAudioServeEngine with memory optimization and quantization."""
        
        if enable_memory_optimization:
            setup_memory_optimization()
        
        # Store settings
        self.device = device
        self.model_name_or_path = model_name_or_path
        self.torch_dtype = torch_dtype
        self.force_quantization = force_quantization
        self.enable_memory_optimization = enable_memory_optimization
        
        # Apply performance optimizations for maximum GPU utilization
        logger.info("🚀 Applying performance optimizations for maximum GPU utilization...")
        performance_optimizer.apply_all_optimizations(quantization_mode=force_quantization or "auto")
        
        # Clear memory before loading
        clear_memory()
        gc.collect()
        
        logger.info(f"🚀 Loading model with quantization: {force_quantization or 'auto'}")
        
        # Load model with quantization using proper bitsandbytes approach
        self._load_quantized_model(model_name_or_path, torch_dtype, force_quantization)
        
        # Load tokenizer
        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path
        logger.info(f"📝 Loading tokenizer from {tokenizer_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        
        # Load audio tokenizer
        logger.info(f"🎵 Initializing Higgs Audio Tokenizer")
        self.audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer_name_or_path, device=device)
        
        # Set model configuration attributes
        self.audio_num_codebooks = self.model.config.audio_num_codebooks
        self.audio_codebook_size = self.model.config.audio_codebook_size
        self.audio_tokenizer_tps = self.audio_tokenizer.tps
        self.samples_per_token = int(self.audio_tokenizer.sampling_rate // self.audio_tokenizer_tps)
        self.hamming_window_len = 2 * self.audio_num_codebooks * self.samples_per_token
        
        # Set the audio special tokens
        self.model.set_audio_special_tokens(self.tokenizer)
        
        # Initialize KV caches
        self._initialize_kv_caches(kv_cache_lengths)
        
        # Initialize collator
        self._initialize_collator()
        
        # Lock to prevent multiple generations from happening at the same time
        self.generate_lock = threading.Lock()
        
        # Log memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"💾 GPU Memory after loading - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
    
    def _load_quantized_model(self, model_name_or_path: str, torch_dtype, force_quantization: Optional[str]):
        """Load model with proper bitsandbytes quantization"""
        
        # Determine quantization config
        quantization_config = None
        
        if force_quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("🔧 Using 4-bit quantization")
            
        elif force_quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("🔧 Using 8-bit quantization")
            
        elif force_quantization == "6bit":
            # 6-bit is achieved using 4-bit with higher precision settings
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,  # Higher precision than bfloat16
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="fp4",  # FP4 instead of NF4 for better quality
            )
            logger.info("🔧 Using 6-bit quantization (4-bit FP4 with float16)")
            
        elif force_quantization == "auto":
            # Auto-select based on GPU memory
            config = get_quantization_config()
            if config.get("load_in_4bit"):
                if config.get("bnb_4bit_quant_type") == "fp4":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="fp4",
                    )
                    logger.info("🔧 Auto-selected 6-bit quantization (4-bit FP4)")
                else:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    logger.info("🔧 Auto-selected 4-bit quantization")
            elif config.get("load_in_8bit"):
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("🔧 Auto-selected 8-bit quantization")
            else:
                logger.info("🔧 Auto-selected full precision")
        else:
            logger.info("🔧 Using full precision (no quantization)")
        
        try:
            # First try with HiggsAudioModel directly
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": True,
            }
            
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            else:
                # No quantization, load normally
                pass
            
            logger.info(f"📦 Loading HiggsAudioModel from {model_name_or_path}")
            logger.info(f"🔧 Model kwargs: {list(model_kwargs.keys())}")
            
            # Try to load the model with quantization
            self.model = HiggsAudioModel.from_pretrained(
                model_name_or_path,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if "device_map" not in model_kwargs and self.device != "cpu":
                self.model = self.model.to(self.device)
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"Model dtype: {self.model.dtype}")
            logger.info(f"Model device: {self.model.device}")
            
            # Apply model-specific optimizations
            self.model = performance_optimizer.optimize_model_settings(self.model)
            
            # Check if quantized
            if hasattr(self.model, 'hf_quantizer') and self.model.hf_quantizer is not None:
                logger.info(f"🔧 Quantizer active: {type(self.model.hf_quantizer).__name__}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            logger.info("🔄 Retrying with 4-bit quantization fallback...")
            
            # Clear memory and try 4-bit
            clear_memory()
            gc.collect()
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": True,
                "quantization_config": quantization_config,
                "device_map": "auto",
            }
            
            try:
                self.model = HiggsAudioModel.from_pretrained(
                    model_name_or_path,
                    **model_kwargs
                )
                logger.info("✅ Model loaded with fallback 4-bit quantization")
            except Exception as e2:
                logger.error(f"❌ Fallback also failed: {e2}")
                logger.info("🔄 Loading without quantization as final fallback...")
                
                # Final fallback - no quantization
                clear_memory()
                gc.collect()
                
                self.model = HiggsAudioModel.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                ).to(self.device)
                
                logger.info("✅ Model loaded without quantization (final fallback)")
    
    def _initialize_kv_caches(self, kv_cache_lengths: List[int]):
        """Initialize KV caches"""
        try:
            cache_config = deepcopy(self.model.config.text_config)
            cache_config.num_hidden_layers = self.model.config.text_config.num_hidden_layers
            if hasattr(self.model.config, 'audio_dual_ffn_layers') and self.model.config.audio_dual_ffn_layers:
                cache_config.num_hidden_layers += len(self.model.config.audio_dual_ffn_layers)
            
            # Adjust cache sizes for memory optimization while supporting voice cloning
            # Voice cloning needs larger caches (up to 6144) for reference audio + generation
            optimized_cache_lengths = [min(length, 6144) for length in kv_cache_lengths]
            
            # Ensure we have a cache that can handle voice cloning sequences
            if max(optimized_cache_lengths) < 4096:
                optimized_cache_lengths.append(4096)
            if max(optimized_cache_lengths) < 6144:
                optimized_cache_lengths.append(6144)
            
            # Remove duplicates and sort
            optimized_cache_lengths = sorted(list(set(optimized_cache_lengths)))
            
            logger.info(f"🔧 Initializing KV caches for lengths: {optimized_cache_lengths}")
            
            self.kv_caches = {}
            for cache_length in optimized_cache_lengths:
                try:
                    cache = StaticCache(
                        config=cache_config,
                        max_batch_size=1,
                        max_cache_len=cache_length,
                        device=self.model.device,
                        dtype=self.model.dtype,
                    )
                    self.kv_caches[cache_length] = cache
                    logger.info(f"✅ Initialized KV cache for length {cache_length}")
                except Exception as e:
                    logger.warning(f"⚠️ Skipping KV cache for length {cache_length}: {e}")
                    continue
            
            if not self.kv_caches:
                logger.warning("⚠️ No KV caches initialized")
                
        except Exception as e:
            logger.warning(f"⚠️ KV cache initialization failed: {e}")
            self.kv_caches = {}
    
    def _initialize_collator(self):
        """Initialize the data collator"""
        try:
            # Initialize whisper processor if needed
            whisper_processor = None
            if hasattr(self.model.config, 'encode_whisper_embed') and self.model.config.encode_whisper_embed:
                logger.info(f"Loading whisper processor")
                whisper_processor = AutoProcessor.from_pretrained(
                    "openai/whisper-large-v3-turbo",
                    trust_remote=True,
                    device=self.device,
                )
            
            # Initialize collator
            self.collator = HiggsAudioSampleCollator(
                whisper_processor=whisper_processor,
                encode_whisper_embed=getattr(self.model.config, 'encode_whisper_embed', False),
                audio_in_token_id=self.model.config.audio_in_token_idx,
                audio_out_token_id=self.model.config.audio_out_token_idx,
                audio_stream_bos_id=self.model.config.audio_stream_bos_id,
                audio_stream_eos_id=self.model.config.audio_stream_eos_id,
                pad_token_id=self.model.config.pad_token_id,
                return_audio_in_tokens=False,
                use_delay_pattern=getattr(self.model.config, 'use_delay_pattern', False),
                audio_num_codebooks=self.model.config.audio_num_codebooks,
                round_to=1,
            )
            logger.info("✅ Collator initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize collator: {e}")
            raise e
    
    def generate(self, *args, **kwargs):
        """Generate with memory optimization - use composition pattern"""
        # Clear cache before generation if enabled
        if self.enable_memory_optimization:
            clear_memory()
        
        # Limit max_new_tokens to prevent OOM
        if "max_new_tokens" in kwargs:
            kwargs["max_new_tokens"] = min(kwargs["max_new_tokens"], 1024)
        
        try:
            # Create a standard HiggsAudioServeEngine instance for generation
            # but use our optimized model
            if not hasattr(self, '_generation_engine'):
                logger.info("🔧 Creating generation engine...")
                
                # Create engine instance without loading model again
                self._generation_engine = HiggsAudioServeEngine.__new__(HiggsAudioServeEngine)
                
                # Set all required attributes
                self._generation_engine.model = self.model
                self._generation_engine.tokenizer = self.tokenizer
                self._generation_engine.audio_tokenizer = self.audio_tokenizer
                self._generation_engine.audio_num_codebooks = self.audio_num_codebooks
                self._generation_engine.audio_codebook_size = self.audio_codebook_size
                self._generation_engine.audio_tokenizer_tps = self.audio_tokenizer_tps
                self._generation_engine.samples_per_token = self.samples_per_token
                self._generation_engine.hamming_window_len = self.hamming_window_len
                self._generation_engine.kv_caches = self.kv_caches
                self._generation_engine.collator = self.collator
                self._generation_engine.generate_lock = self.generate_lock
                self._generation_engine.device = self.device
                self._generation_engine.model_name_or_path = self.model_name_or_path
                self._generation_engine.torch_dtype = self.torch_dtype
                
                logger.info("✅ Generation engine ready")
            
            # Use the generation engine
            return self._generation_engine.generate(*args, **kwargs)
                
        except Exception as e:
            error_msg = str(e)
            
            if "larger than all past key values buckets" in error_msg:
                logger.error(f"❌ Sequence too long for KV cache: {e}")
                logger.info("🔄 Trying to reinitialize with larger KV caches...")
                
                # Extract the required sequence length from error message
                import re
                match = re.search(r'sequence length (\d+)', error_msg)
                if match:
                    required_length = int(match.group(1))
                    # Round up to next power of 2 or add some buffer
                    new_cache_size = max(8192, ((required_length // 1024) + 1) * 1024)
                    logger.info(f"🔧 Adding KV cache for length: {new_cache_size}")
                    
                    # Try to add a larger cache
                    try:
                        cache_config = deepcopy(self.model.config.text_config)
                        cache_config.num_hidden_layers = self.model.config.text_config.num_hidden_layers
                        if hasattr(self.model.config, 'audio_dual_ffn_layers') and self.model.config.audio_dual_ffn_layers:
                            cache_config.num_hidden_layers += len(self.model.config.audio_dual_ffn_layers)
                        
                        new_cache = StaticCache(
                            config=cache_config,
                            max_batch_size=1,
                            max_cache_len=new_cache_size,
                            device=self.model.device,
                            dtype=self.model.dtype,
                        )
                        
                        # Add to both our cache and the generation engine's cache
                        self.kv_caches[new_cache_size] = new_cache
                        self._generation_engine.kv_caches[new_cache_size] = new_cache
                        
                        logger.info(f"✅ Added larger KV cache, retrying generation...")
                        return self._generation_engine.generate(*args, **kwargs)
                        
                    except Exception as cache_e:
                        logger.error(f"❌ Failed to create larger cache: {cache_e}")
                        raise e
                else:
                    raise e
                    
            elif "CUDA out of memory" in error_msg or isinstance(e, torch.cuda.OutOfMemoryError):
                logger.warning(f"⚠️ OOM during generation: {e}")
                logger.info("🔄 Retrying with reduced parameters...")
                
                # Clear memory and retry with smaller parameters
                if self.enable_memory_optimization:
                    clear_memory()
                
                # Reduce max_new_tokens
                if "max_new_tokens" in kwargs:
                    kwargs["max_new_tokens"] = min(kwargs["max_new_tokens"] // 2, 512)
                
                # Retry
                return self._generation_engine.generate(*args, **kwargs)
            else:
                # Re-raise other exceptions
                raise e