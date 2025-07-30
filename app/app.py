"""
Gradio UI for Text-to-Speech using HiggsAudioServeEngine
"""

import argparse
import base64
import os
import uuid
import json
from typing import Optional
import gradio as gr
from loguru import logger
import numpy as np
import time
from functools import lru_cache
import re
import torch

# Set CUDA memory management for 8-bit quantization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logger.info("🔧 Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for 8-bit quantization")

# Import HiggsAudio components
from higgs_audio.serve.serve_engine import HiggsAudioServeEngine
from higgs_audio.data_types import ChatMLSample, AudioContent, Message

# Import optimized components
try:
    from optimized_serve_engine import OptimizedHiggsAudioServeEngine
    from memory_config import setup_memory_optimization, clear_memory
    OPTIMIZATION_AVAILABLE = True
    logger.info("✅ Memory optimization modules loaded")
except ImportError as e:
    logger.warning(f"⚠️ Memory optimization not available: {e}")
    OPTIMIZATION_AVAILABLE = False

# Import numpy for audio processing
import numpy as np

# Global engine instance
engine = None
# Global device setting
DEVICE_ARG = None
# Global quantization setting
QUANTIZATION_MODE = "auto"
# Global memory optimization setting
MEMORY_OPTIMIZATION_ENABLED = True

# Default model configuration
DEFAULT_MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
DEFAULT_AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
SAMPLE_RATE = 24000

DEFAULT_SYSTEM_PROMPT = (
    "Generate audio following instruction.\n\n"
    "<|scene_desc_start|>\n"
    "Audio is recorded from a quiet room.\n"
    "<|scene_desc_end|>"
)

DEFAULT_STOP_STRINGS = ["<|end_of_text|>", "<|eot_id|>"]

# Predefined examples for system and input messages
PREDEFINED_EXAMPLES = {
    "voice-clone": {
        "system_prompt": "",
        "input_text": "Hey there! I'm your friendly voice twin in the making. Pick a voice preset below or upload your own audio - let's clone some vocals and bring your voice to life! ",
        "description": "Voice clone to clone the reference audio. Leave the system prompt empty.",
    },
    "smart-voice": {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "input_text": "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.",
        "description": "Smart voice to generate speech based on the context",
    },
    "multispeaker-voice-description": {
        "system_prompt": "You are an AI assistant designed to convert text into speech.\n"
        "If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.\n"
        "If no speaker tag is present, select a suitable voice on your own.\n\n"
        "<|scene_desc_start|>\n"
        "SPEAKER0: feminine\n"
        "SPEAKER1: masculine\n"
        "<|scene_desc_end|>",
        "input_text": "[SPEAKER0] I can't believe you did that without even asking me first!\n"
        "[SPEAKER1] Oh, come on! It wasn't a big deal, and I knew you would overreact like this.\n"
        "[SPEAKER0] Overreact? You made a decision that affects both of us without even considering my opinion!\n"
        "[SPEAKER1] Because I didn't have time to sit around waiting for you to make up your mind! Someone had to act.",
        "description": "Multispeaker with different voice descriptions in the system prompt",
    },
    "single-speaker-voice-description": {
        "system_prompt": "Generate audio following instruction.\n\n"
        "<|scene_desc_start|>\n"
        "SPEAKER0: He speaks with a clear British accent and a conversational, inquisitive tone. His delivery is articulate and at a moderate pace, and very clear audio.\n"
        "<|scene_desc_end|>",
        "input_text": "Hey, everyone! Welcome back to Tech Talk Tuesdays.\n"
        "It's your host, Alex, and today, we're diving into a topic that's become absolutely crucial in the tech world — deep learning.\n"
        "And let's be honest, if you've been even remotely connected to tech, AI, or machine learning lately, you know that deep learning is everywhere.\n"
        "\n"
        "So here's the big question: Do you want to understand how deep learning works?\n",
        "description": "Single speaker with voice description in the system prompt",
    },
    "single-speaker-zh": {
        "system_prompt": "Generate audio following instruction.\n\n"
        "<|scene_desc_start|>\n"
        "Audio is recorded from a quiet room.\n"
        "<|scene_desc_end|>",
        "input_text": "大家好, 欢迎收听本期的跟李沐学AI. 今天沐哥在忙着洗数据, 所以由我, 希格斯主播代替他讲这期视频.\n"
        "今天我们要聊的是一个你绝对不能忽视的话题: 多模态学习.\n"
        "那么, 问题来了, 你真的了解多模态吗? 你知道如何自己动手构建多模态大模型吗.\n"
        "或者说, 你能察觉到我其实是个机器人吗?",
        "description": "Single speaker speaking Chinese",
    },
    "single-speaker-bgm": {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "input_text": "[music start] I will remember this, thought Ender, when I am defeated. To keep dignity, and give honor where it's due, so that defeat is not disgrace. And I hope I don't have to do it often. [music end]",
        "description": "Single speaker with BGM using music tag. This is an experimental feature and you may need to try multiple times to get the best result.",
    },
    "long-form-story": {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "input_text": "Once upon a time, in a small village nestled between rolling hills and a sparkling river, there lived a young girl named Luna who possessed an extraordinary gift. She could hear the whispers of the wind and understand the songs of the birds. Every morning, Luna would wake before dawn and venture into the forest that bordered her village. The ancient trees seemed to recognize her presence, their branches swaying gently as if greeting an old friend. As she walked deeper into the woods, the sounds of the village faded away, replaced by the symphony of nature that only she could truly comprehend. The wind would tell her stories of distant lands, of mountains that touched the clouds and oceans that stretched beyond the horizon. The birds would share gossip from their travels, speaking of the changing seasons and the migration patterns of their cousins. Luna treasured these moments of connection with the natural world, knowing that her gift was both rare and precious.",
        "description": "Long-form story generation with smart chunking for consistency",
    },
    "multi-speaker-debate": {
        "system_prompt": "You are an AI assistant designed to convert text into speech.\n"
        "Generate speech for multiple speakers in a debate format.\n\n"
        "<|scene_desc_start|>\n"
        "SPEAKER0: Professional debater with confident tone\n"
        "SPEAKER1: Academic with thoughtful, measured speech\n"
        "SPEAKER2: Moderator with neutral, authoritative voice\n"
        "<|scene_desc_end|>",
        "input_text": "[SPEAKER2] Welcome to today's debate on artificial intelligence and creativity. Our first speaker will argue that AI can be truly creative.\n\n[SPEAKER0] Thank you, moderator. I firmly believe that artificial intelligence has already demonstrated genuine creativity. Look at AI-generated art, music, and literature that moves people emotionally. Creativity isn't just about human experience—it's about generating novel, valuable ideas.\n\n[SPEAKER1] I respectfully disagree. While AI can produce impressive outputs, true creativity requires consciousness, intentionality, and lived experience. AI systems are sophisticated pattern matching tools, but they lack the subjective experience that drives authentic creative expression.\n\n[SPEAKER0] But isn't human creativity also based on pattern recognition and recombination of existing ideas? We learn from what came before and create new combinations. AI does the same thing, just more efficiently.\n\n[SPEAKER2] Interesting points from both sides. Let's explore this further.",
        "description": "Multi-speaker debate with automatic speaker detection and voice assignment",
    },
}

def encode_audio_file(file_path):
    """Encode an audio file to base64."""
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


def get_current_device(device_arg=None):
    """Get the current device."""
    if device_arg:
        # If user explicitly wants CUDA, verify it's actually working
        if device_arg == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
            # Test if we can actually create a tensor on CUDA
            try:
                test_tensor = torch.tensor([1.0], device="cuda")
                del test_tensor
                return "cuda"
            except Exception as e:
                logger.warning(f"CUDA requested but failed to create tensor: {e}, falling back to CPU")
                return "cpu"
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_voice_presets():
    """Load the voice presets from the voice_examples directory."""
    try:
        with open(
            os.path.join(os.path.dirname(__file__), "voice_examples", "config.json"),
            "r",
            encoding="utf-8",
        ) as f:
            voice_dict = json.load(f)
        voice_presets = {k: v["transcript"] for k, v in voice_dict.items()}
        voice_presets["EMPTY"] = "No reference voice"
        logger.info(f"Loaded voice presets: {list(voice_presets.keys())}")
        return voice_presets
    except FileNotFoundError:
        logger.warning("Voice examples config file not found. Using empty voice presets.")
        return {"EMPTY": "No reference voice"}
    except Exception as e:
        logger.error(f"Error loading voice presets: {e}")
        return {"EMPTY": "No reference voice"}


def get_voice_preset(voice_preset):
    """Get the voice path and text for a given voice preset."""
    voice_path = os.path.join(os.path.dirname(__file__), "voice_examples", f"{voice_preset}.wav")
    if not os.path.exists(voice_path):
        logger.warning(f"Voice preset file not found: {voice_path}")
        return None, "Voice preset not found"

    text = VOICE_PRESETS.get(voice_preset, "No transcript available")
    return voice_path, text


def normalize_chinese_punctuation(text):
    """
    Convert Chinese (full-width) punctuation marks to English (half-width) equivalents.
    """
    # Mapping of Chinese punctuation to English punctuation
    chinese_to_english_punct = {
        "，": ", ",  # comma
        "。": ".",  # period
        "：": ":",  # colon
        "；": ";",  # semicolon
        "？": "?",  # question mark
        "！": "!",  # exclamation mark
        "（": "(",  # left parenthesis
        "）": ")",  # right parenthesis
        "【": "[",  # left square bracket
        "】": "]",  # right square bracket
        "《": "<",  # left angle quote
        "》": ">",  # right angle quote
        "“": '"',  # left double quotation
        "”": '"',  # right double quotation
        "‘": "'",  # left single quotation
        "’": "'",  # right single quotation
        "、": ",",  # enumeration comma
        "—": "-",  # em dash
        "…": "...",  # ellipsis
        "·": ".",  # middle dot
        "「": '"',  # left corner bracket
        "」": '"',  # right corner bracket
        "『": '"',  # left double corner bracket
        "』": '"',  # right double corner bracket
    }

    # Replace each Chinese punctuation with its English counterpart
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)

    return text


def normalize_text(transcript: str):
    transcript = normalize_chinese_punctuation(transcript)
    # Other normalizations (e.g., parentheses and other symbols. Will be improved in the future)
    transcript = transcript.replace("(", " ")
    transcript = transcript.replace(")", " ")
    transcript = transcript.replace("°F", " degrees Fahrenheit")
    transcript = transcript.replace("°C", " degrees Celsius")

    for tag, replacement in [
        ("[laugh]", "<SE>[Laughter]</SE>"),
        ("[humming start]", "<SE>[Humming]</SE>"),
        ("[humming end]", "<SE_e>[Humming]</SE_e>"),
        ("[music start]", "<SE_s>[Music]</SE_s>"),
        ("[music end]", "<SE_e>[Music]</SE_e>"),
        ("[music]", "<SE>[Music]</SE>"),
        ("[sing start]", "<SE_s>[Singing]</SE_s>"),
        ("[sing end]", "<SE_e>[Singing]</SE_e>"),
        ("[applause]", "<SE>[Applause]</SE>"),
        ("[cheering]", "<SE>[Cheering]</SE>"),
        ("[cough]", "<SE>[Cough]</SE>"),
    ]:
        transcript = transcript.replace(tag, replacement)

    lines = transcript.split("\n")
    transcript = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
    transcript = transcript.strip()

    if not any([transcript.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
        transcript += "."

    return transcript


def initialize_engine(model_path, audio_tokenizer_path, device_arg=None, quantization_mode="auto", memory_optimization=True) -> bool:
    """Initialize the HiggsAudioServeEngine with memory optimization."""
    global engine
    
    # Setup memory optimization if available
    if OPTIMIZATION_AVAILABLE:
        setup_memory_optimization()
        clear_memory()
    
    try:
        device = get_current_device(device_arg)
        logger.info(f"Initializing engine with model: {model_path} and audio tokenizer: {audio_tokenizer_path} on device: {device}")
        
        # Log additional device information
        if device == "cuda":
            logger.info(f"CUDA Device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Use optimized engine if available
        if OPTIMIZATION_AVAILABLE and device == "cuda":
            logger.info(f"🚀 Using OptimizedHiggsAudioServeEngine with quantization: {quantization_mode}")
            
            # Convert quantization mode
            force_quantization = None if quantization_mode in ["auto", "none"] else quantization_mode
            
            try:
                engine = OptimizedHiggsAudioServeEngine(
                    model_name_or_path=model_path,
                    audio_tokenizer_name_or_path=audio_tokenizer_path,
                    device=device,
                    force_quantization=force_quantization,
                    enable_memory_optimization=memory_optimization,
                )
                
                # Verify engine has all required attributes
                required_attrs = ['generate_lock', 'model', 'tokenizer', 'audio_tokenizer']
                missing_attrs = [attr for attr in required_attrs if not hasattr(engine, attr)]
                if missing_attrs:
                    raise AttributeError(f"OptimizedHiggsAudioServeEngine missing attributes: {missing_attrs}")
                    
            except Exception as opt_e:
                logger.error(f"❌ OptimizedHiggsAudioServeEngine failed: {opt_e}")
                logger.info("🔄 Falling back to standard HiggsAudioServeEngine")
                engine = HiggsAudioServeEngine(
                    model_name_or_path=model_path,
                    audio_tokenizer_name_or_path=audio_tokenizer_path,
                    device=device,
                )
        else:
            logger.info("📦 Using standard HiggsAudioServeEngine")
            engine = HiggsAudioServeEngine(
                model_name_or_path=model_path,
                audio_tokenizer_name_or_path=audio_tokenizer_path,
                device=device,
            )
        
        logger.info(f"✅ Successfully initialized HiggsAudioServeEngine with model: {model_path} on device: {device}")
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"❌ CUDA OOM Error during initialization: {e}")
        
        # Try with optimized engine and forced quantization
        if OPTIMIZATION_AVAILABLE and device == "cuda":
            logger.info("🔄 Retrying with aggressive 4-bit quantization...")
            try:
                if OPTIMIZATION_AVAILABLE:
                    clear_memory()
                
                engine = OptimizedHiggsAudioServeEngine(
                    model_name_or_path=model_path,
                    audio_tokenizer_name_or_path=audio_tokenizer_path,
                    device=device,
                    force_quantization="4bit",
                    enable_memory_optimization=memory_optimization,
                )
                
                # Verify engine has all required attributes
                required_attrs = ['generate_lock', 'model', 'tokenizer', 'audio_tokenizer']
                missing_attrs = [attr for attr in required_attrs if not hasattr(engine, attr)]
                if missing_attrs:
                    raise AttributeError(f"OptimizedHiggsAudioServeEngine missing attributes: {missing_attrs}")
                
                logger.info("✅ Successfully initialized with 4-bit quantization!")
                return True
                
            except Exception as quant_e:
                logger.error(f"❌ Failed with quantization: {quant_e}")
                logger.info("🔄 Falling back to standard engine on CPU")
        
        # Final fallback to CPU
        if device == "cuda":
            logger.info("🔄 Final fallback to CPU...")
            try:
                if OPTIMIZATION_AVAILABLE:
                    clear_memory()
                
                engine = HiggsAudioServeEngine(
                    model_name_or_path=model_path,
                    audio_tokenizer_name_or_path=audio_tokenizer_path,
                    device="cpu",
                )
                logger.info("✅ Successfully initialized on CPU (fallback)")
                return True
                
            except Exception as cpu_e:
                logger.error(f"❌ Failed to initialize on CPU as well: {cpu_e}")
                return False
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize engine on {device}: {e}")
        return False


def check_return_audio(audio_wv: np.ndarray):
    # check if the audio returned is all silent
    if np.all(audio_wv == 0):
        logger.warning("Audio is silent, returning None")


def process_text_output(text_output: str):
    # remove all the continuous <|AUDIO_OUT|> tokens with a single <|AUDIO_OUT|>
    text_output = re.sub(r"(<\|AUDIO_OUT\|>)+", r"<|AUDIO_OUT|>", text_output)
    return text_output


def prepare_chatml_sample(
    voice_preset: str,
    text: str,
    reference_audio: Optional[str] = None,
    reference_text: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
):
    """Prepare a ChatMLSample for the HiggsAudioServeEngine."""
    messages = []

    # Add system message if provided
    if len(system_prompt) > 0:
        messages.append(Message(role="system", content=system_prompt))

    # Add reference audio if provided
    audio_base64 = None
    ref_text = ""

    if reference_audio:
        # Custom reference audio
        audio_base64 = encode_audio_file(reference_audio)
        ref_text = reference_text or ""
    elif voice_preset != "EMPTY":
        # Voice preset
        voice_path, ref_text = get_voice_preset(voice_preset)
        if voice_path is None:
            logger.warning(f"Voice preset {voice_preset} not found, skipping reference audio")
        else:
            audio_base64 = encode_audio_file(voice_path)

    # Only add reference audio if we have it
    if audio_base64 is not None:
        # Add user message with reference text
        messages.append(Message(role="user", content=ref_text))

        # Add assistant message with audio content
        audio_content = AudioContent(raw_audio=audio_base64, audio_url="")
        messages.append(Message(role="assistant", content=[audio_content]))

    # Add the main user message
    text = normalize_text(text)
    messages.append(Message(role="user", content=text))

    return ChatMLSample(messages=messages)


def text_to_speech(
    text,
    voice_preset,
    reference_audio=None,
    reference_text=None,
    max_completion_tokens=1024,
    temperature=1.0,
    top_p=0.95,
    top_k=50,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    stop_strings=None,
    ras_win_len=7,
    ras_win_max_num_repeat=2,
    # Advanced features
    enable_smart_chunking=True,
    chunk_method="auto",
    enable_multi_speaker=True,
    long_form_consistency=True,
):
    """Convert text to speech using HiggsAudioServeEngine."""
    global engine, DEVICE_ARG, QUANTIZATION_MODE, MEMORY_OPTIMIZATION_ENABLED

    if engine is None:
        initialize_engine(DEFAULT_MODEL_PATH, DEFAULT_AUDIO_TOKENIZER_PATH, DEVICE_ARG, QUANTIZATION_MODE, MEMORY_OPTIMIZATION_ENABLED)

    try:
        # Generate request ID first
        request_id = f"tts-playground-{str(uuid.uuid4())}"
        
        # Check if we should use advanced generation
        # Only use advanced for explicit multi-speaker content
        should_use_advanced = (
            enable_smart_chunking and 
            enable_multi_speaker and 
            ("[SPEAKER" in text or ":" in text and len(text.split(":")) > 2)
        )
        
        if should_use_advanced and OPTIMIZATION_AVAILABLE:
            logger.info("🚀 Using advanced generation with smart chunking")
            return _generate_with_advanced_features(
                text, voice_preset, reference_audio, reference_text,
                max_completion_tokens, temperature, top_p, top_k,
                system_prompt, ras_win_len, ras_win_max_num_repeat,
                chunk_method, enable_multi_speaker, long_form_consistency, request_id
            )
        
        # Check if this is long text that needs chunking (>100 words) or voice cloning
        word_count = len(text.split())
        needs_chunking = word_count > 100 or voice_preset != "EMPTY" or reference_audio is not None
        
        if needs_chunking:
            logger.info("🎵 Using SEQUENTIAL VOICE PROCESSING - Exactly as requested!")
            logger.info("   1. Smart chunk into 70-80 words")
            logger.info("   2. Save chunks separately")
            logger.info("   3. Process ONE chunk at a time")
            logger.info("   4. Same settings/ref audio/seed for each")
            logger.info("   5. Save each received audio chunk")
            logger.info("   6. Combine all chunks into complete audio")
            
            # Log voice cloning setup
            if reference_audio:
                logger.info(f"🎤 CUSTOM REFERENCE AUDIO: {reference_audio}")
                logger.info(f"📝 Reference text: {reference_text or 'None'}")
            elif voice_preset != "EMPTY":
                logger.info(f"🎵 VOICE PRESET: {voice_preset}")
            
            logger.info(f"📊 Text stats: {word_count} words → ~{(word_count + 74) // 75} chunks expected")
            
            try:
                from sequential_voice_processor import SequentialVoiceProcessor
                
                processor = SequentialVoiceProcessor(engine)
                
                # Process text exactly as you described with reference audio support
                audio, status_message = processor.process_text_sequentially(
                    text=text,
                    voice_preset=voice_preset,
                    system_prompt=system_prompt,
                    reference_audio=reference_audio,  # Pass custom reference audio
                    reference_text=reference_text     # Pass custom reference text
                )
                
                if audio is not None:
                    # Convert to int16 for Gradio
                    audio_data = (audio * 32767).astype(np.int16)
                    duration = len(audio) / 24000
                    input_words = len(text.split())
                    
                    logger.info(f"🎉 SEQUENTIAL VOICE PROCESSING SUCCESS:")
                    logger.info(f"   Final audio: {duration:.2f}s")
                    logger.info(f"   Input words: {input_words}")
                    logger.info(f"   Speech rate: {input_words/duration:.1f} words/sec")
                    logger.info(f"   Method: Sequential chunk processing")
                    
                    return status_message, (24000, audio_data)
                else:
                    error_msg = status_message or "Sequential processing failed"
                    logger.error(f"❌ Sequential processing failed: {error_msg}")
                    return error_msg, None
                    
            except Exception as e:
                logger.error(f"❌ Sequential processing EXCEPTION: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                
                # NO FALLBACK - prevents sequence length errors
                error_msg = f"Sequential voice processing failed: {str(e)}"
                logger.error("❌ Returning error - NO fallback to prevent token overflow")
                return error_msg, None
        
        # Standard generation (no voice cloning or fallback)
        logger.info("📦 Using standard generation")
        # Prepare ChatML sample
        chatml_sample = prepare_chatml_sample(voice_preset, text, reference_audio, reference_text, system_prompt)

        # Convert stop strings format
        if stop_strings is None:
            stop_list = DEFAULT_STOP_STRINGS
        else:
            stop_list = [s for s in stop_strings["stops"] if s.strip()]

        logger.info(
            f"{request_id}: Generating speech for text: {text[:100]}..., \n"
            f"with parameters: temperature={temperature}, top_p={top_p}, top_k={top_k}, stop_list={stop_list}, "
            f"ras_win_len={ras_win_len}, ras_win_max_num_repeat={ras_win_max_num_repeat}"
        )
        start_time = time.time()

        # Log current device before generation
        if torch.cuda.is_available():
            logger.info(f"{request_id}: Using CUDA device: {torch.cuda.get_device_name()}")
            logger.info(f"{request_id}: CUDA memory before generation: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        # Clear memory before generation if optimization is enabled
        if OPTIMIZATION_AVAILABLE and MEMORY_OPTIMIZATION_ENABLED:
            clear_memory()
        
        # Generate using the engine
        try:
            response = engine.generate(
                chat_ml_sample=chatml_sample,
                max_new_tokens=max_completion_tokens,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
                top_p=top_p,
                stop_strings=stop_list,
                ras_win_len=ras_win_len if ras_win_len > 0 else None,
                ras_win_max_num_repeat=max(ras_win_len, ras_win_max_num_repeat),
            )
        except torch.cuda.OutOfMemoryError as oom_e:
            logger.error(f"{request_id}: CUDA OOM during generation: {oom_e}")
            
            # Try with reduced parameters
            if OPTIMIZATION_AVAILABLE and MEMORY_OPTIMIZATION_ENABLED:
                clear_memory()
            
            logger.info(f"{request_id}: Retrying with reduced max_new_tokens...")
            reduced_tokens = min(max_completion_tokens // 2, 512)
            
            try:
                response = engine.generate(
                    chat_ml_sample=chatml_sample,
                    max_new_tokens=reduced_tokens,
                    temperature=temperature,
                    top_k=top_k if top_k > 0 else None,
                    top_p=top_p,
                    stop_strings=stop_list,
                    ras_win_len=ras_win_len if ras_win_len > 0 else None,
                    ras_win_max_num_repeat=max(ras_win_len, ras_win_max_num_repeat),
                )
                logger.info(f"{request_id}: Successfully generated with reduced tokens ({reduced_tokens})")
            except Exception as retry_e:
                raise Exception(f"Generation failed even with reduced parameters: {retry_e}") from oom_e
        
        # Log memory usage after generation
        if torch.cuda.is_available():
            logger.info(f"{request_id}: CUDA memory after generation: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

        generation_time = time.time() - start_time
        logger.info(f"{request_id}: Generated audio in {generation_time:.3f} seconds")
        gr.Info(f"Generated audio in {generation_time:.3f} seconds")

        # Process the response
        text_output = process_text_output(response.generated_text)

        if response.audio is not None:
            # Convert to int16 for Gradio
            audio_data = (response.audio * 32767).astype(np.int16)
            check_return_audio(audio_data)
            return text_output, (response.sampling_rate, audio_data)
        else:
            logger.warning("No audio generated")
            return text_output, None

    except Exception as e:
        error_msg = f"Error generating speech: {e}"
        logger.error(error_msg)
        gr.Error(error_msg)
        return f"❌ {error_msg}", None


def reload_model_with_settings(quantization_mode, memory_optimization):
    """Reload the model with new quantization and memory settings"""
    global engine, QUANTIZATION_MODE, MEMORY_OPTIMIZATION_ENABLED
    
    try:
        # Update global settings
        QUANTIZATION_MODE = quantization_mode
        MEMORY_OPTIMIZATION_ENABLED = memory_optimization
        
        # Clear existing engine
        if engine is not None:
            del engine
            engine = None
        
        # Clear memory if optimization is available
        if OPTIMIZATION_AVAILABLE:
            clear_memory()
        
        # Initialize with new settings
        success = initialize_engine(
            DEFAULT_MODEL_PATH, 
            DEFAULT_AUDIO_TOKENIZER_PATH, 
            DEVICE_ARG,
            quantization_mode,
            memory_optimization
        )
        
        if success:
            # Get model info
            if hasattr(engine, 'model') and hasattr(engine.model, 'dtype'):
                model_dtype = str(engine.model.dtype)
            else:
                model_dtype = "unknown"
            
            # Get quantization info
            quant_info = "Full Precision"
            if quantization_mode == "4bit":
                quant_info = "4-bit Quantized"
            elif quantization_mode == "8bit":
                quant_info = "8-bit Quantized"
            elif quantization_mode == "auto":
                quant_info = "Auto-selected"
            
            # Get memory info
            memory_info = ""
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                memory_info = f" | GPU Memory: {allocated:.1f}GB"
            
            status_html = f'<p style="font-size: 0.85em; color: var(--success-text-color); margin: 5px 0;"> ✅ Model loaded successfully<br>Quantization: {quant_info} | Dtype: {model_dtype}{memory_info}</p>'
            
            gr.Info(f"Model reloaded successfully with {quant_info}")
            return status_html
        else:
            error_html = '<p style="font-size: 0.85em; color: var(--error-text-color); margin: 5px 0;"> ❌ Failed to load model</p>'
            gr.Error("Failed to reload model")
            return error_html
            
    except Exception as e:
        error_html = f'<p style="font-size: 0.85em; color: var(--error-text-color); margin: 5px 0;"> ❌ Error: {str(e)}</p>'
        gr.Error(f"Error reloading model: {e}")
        return error_html

def update_gpu_memory_status():
    """Update GPU memory status display"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        reserved_memory = torch.cuda.memory_reserved() / 1024**3
        
        return f'<p style="font-size: 0.9em; color: var(--body-text-color); margin: 5px 0; padding: 8px; background: var(--background-fill-secondary); border-radius: 6px;"> 🖥️ GPU: {torch.cuda.get_device_name()} | Total: {total_memory:.1f}GB | Used: {allocated_memory:.1f}GB | Reserved: {reserved_memory:.1f}GB</p>'
    else:
        return '<p style="font-size: 0.9em; color: var(--body-text-color); margin: 5px 0; padding: 8px; background: var(--background-fill-secondary); border-radius: 6px;"> 🖥️ GPU: Not available (using CPU)</p>'

def update_quantization_info(quantization_mode):
    """Update quantization information display"""
    info_map = {
        "auto": "🔧 Auto mode will select optimal quantization based on your GPU memory",
        "none": "🎯 Full precision mode uses ~18GB GPU memory but provides highest quality",
        "8bit": "⚖️ 8-bit quantization uses ~12GB GPU memory with minimal quality loss (~2%)",
        "6bit": "🎨 6-bit quantization uses ~10GB GPU memory with good quality (~3% loss)",
        "4bit": "💾 4-bit quantization uses ~8GB GPU memory with small quality loss (~5%)"
    }
    
    info_text = info_map.get(quantization_mode, "Unknown quantization mode")
    return f'<p style="font-size: 0.8em; color: var(--body-text-color-subdued); margin: 5px 0; padding: 6px; background: var(--background-fill-primary); border-radius: 4px;"> {info_text}</p>'

def _generate_with_advanced_features(
    text, voice_preset, reference_audio, reference_text,
    max_completion_tokens, temperature, top_p, top_k,
    system_prompt, ras_win_len, ras_win_max_num_repeat,
    chunk_method, enable_multi_speaker, long_form_consistency, request_id
):
    """Generate audio using advanced features"""
    
    try:
        from advanced_generation_engine import AdvancedGenerationEngine, AdvancedGenerationConfig
        
        # Create advanced engine
        advanced_engine = AdvancedGenerationEngine(engine)
        
        # Configure generation
        config = AdvancedGenerationConfig(
            chunk_method=chunk_method if chunk_method != "auto" else "sentence",
            max_new_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            system_prompt=system_prompt,
            reference_audio=voice_preset if voice_preset != "EMPTY" else None,
            cross_chunk_consistency=long_form_consistency,
            clear_cache_between_chunks=MEMORY_OPTIMIZATION_ENABLED,
            quantization_mode=QUANTIZATION_MODE
        )
        
        # Handle custom reference audio
        if reference_audio and reference_text:
            # For custom reference, we'll use the standard method for now
            # Advanced custom reference handling could be added later
            logger.info("🎵 Custom reference audio detected, using standard generation")
            chatml_sample = prepare_chatml_sample(voice_preset, text, reference_audio, reference_text, system_prompt)
            
            response = engine.generate(
                chat_ml_sample=chatml_sample,
                max_new_tokens=max_completion_tokens,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
                top_p=top_p,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                ras_win_len=ras_win_len if ras_win_len > 0 else None,
                ras_win_max_num_repeat=max(ras_win_len, ras_win_max_num_repeat),
            )
            
            text_output = process_text_output(response.generated_text)
            
            if response.audio is not None:
                audio_data = (response.audio * 32767).astype(np.int16)
                return text_output, (response.sampling_rate, audio_data)
            else:
                return text_output, None
        
        # Use advanced generation
        if enable_multi_speaker:
            # Check if text contains multiple speakers
            from advanced_text_processor import AdvancedTextProcessor
            processor = AdvancedTextProcessor(engine.tokenizer, QUANTIZATION_MODE)
            speakers_content = processor.detect_speakers(text)
            
            if len(speakers_content) > 1:
                logger.info(f"🎭 Multi-speaker content detected: {len(speakers_content)} speakers")
                
                # For now, use the same voice for all speakers
                # Could be enhanced to support different voices per speaker
                speaker_voices = {}
                for speaker, _ in speakers_content:
                    speaker_voices[speaker] = voice_preset if voice_preset != "EMPTY" else "belinda"
                
                result = advanced_engine.generate_multi_speaker(text, speaker_voices, config)
            else:
                result = advanced_engine.generate_advanced(text, config)
        else:
            # Single speaker advanced generation
            if len(text) > 1000:
                logger.info("📚 Long-form content detected")
                result = advanced_engine.generate_long_form(
                    text, voice_preset if voice_preset != "EMPTY" else "belinda", config
                )
            else:
                result = advanced_engine.generate_advanced(text, config)
        
        # Process result
        if result.audio is not None:
            logger.info(f"✅ Advanced generation complete: {result.total_duration:.2f}s, {result.chunks_processed} chunks")
            audio_data = (result.audio * 32767).astype(np.int16)
            
            # Add generation info to text output
            info_text = f"Generated {result.chunks_processed} chunks, {result.total_duration:.2f}s total"
            if result.speakers_used and len(result.speakers_used) > 1:
                info_text += f", {len(result.speakers_used)} speakers"
            
            text_output = result.generated_text + f"\n\n[{info_text}]"
            
            return text_output, (result.sampling_rate, audio_data)
        else:
            logger.error("❌ Advanced generation failed")
            return "❌ Advanced generation failed", None
            
    except Exception as e:
        logger.error(f"❌ Advanced generation error: {e}")
        # Fallback to standard generation
        logger.info("🔄 Falling back to standard generation")
        
        chatml_sample = prepare_chatml_sample(voice_preset, text, reference_audio, reference_text, system_prompt)
        
        response = engine.generate(
            chat_ml_sample=chatml_sample,
            max_new_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
            ras_win_len=ras_win_len if ras_win_len > 0 else None,
            ras_win_max_num_repeat=max(ras_win_len, ras_win_max_num_repeat),
        )
        
        text_output = process_text_output(response.generated_text)
        
        if response.audio is not None:
            audio_data = (response.audio * 32767).astype(np.int16)
            return text_output, (response.sampling_rate, audio_data)
        else:
            return text_output, None

def create_ui():
    # Ensure VOICE_PRESETS is loaded
    global VOICE_PRESETS
    if 'VOICE_PRESETS' not in globals() or VOICE_PRESETS is None:
        VOICE_PRESETS = load_voice_presets()
    
    my_theme = gr.Theme.load("theme.json")

    # Add custom CSS to disable focus highlighting on textboxes
    custom_css = """
    .gradio-container input:focus, 
    .gradio-container textarea:focus, 
    .gradio-container select:focus,
    .gradio-container .gr-input:focus,
    .gradio-container .gr-textarea:focus,
    .gradio-container .gr-textbox:focus,
    .gradio-container .gr-textbox:focus-within,
    .gradio-container .gr-form:focus-within,
    .gradio-container *:focus {
        box-shadow: none !important;
        border-color: var(--border-color-primary) !important;
        outline: none !important;
        background-color: var(--input-background-fill) !important;
    }

    /* Override any hover effects as well */
    .gradio-container input:hover, 
    .gradio-container textarea:hover, 
    .gradio-container select:hover,
    .gradio-container .gr-input:hover,
    .gradio-container .gr-textarea:hover,
    .gradio-container .gr-textbox:hover {
        border-color: var(--border-color-primary) !important;
        background-color: var(--input-background-fill) !important;
    }

    /* Style for checked checkbox */
    .gradio-container input[type="checkbox"]:checked {
        background-color: var(--primary-500) !important;
        border-color: var(--primary-500) !important;
    }
    """

    default_template = "smart-voice"

    """Create the Gradio UI."""
    with gr.Blocks(theme=my_theme, css=custom_css, title="Higgs Audio Text-to-Speech") as demo:
        gr.Markdown("# Higgs Audio Text-to-Speech Playground")
        gr.Markdown("🎵 **TRUE SEQUENTIAL**: Smart 70-80 word chunks → Save separately → Process ONE at a time → Same settings/seed → Save each → Combine complete audio!")

        # Main UI section
        with gr.Row():
            with gr.Column(scale=2):
                # Template selection dropdown
                template_dropdown = gr.Dropdown(
                    label="TTS Template",
                    choices=list(PREDEFINED_EXAMPLES.keys()),
                    value=default_template,
                    info="Select a predefined example for system and input messages.",
                )

                # Template description display
                template_description = gr.HTML(
                    value=f'<p style="font-size: 0.85em; color: var(--body-text-color-subdued); margin: 0; padding: 0;"> {PREDEFINED_EXAMPLES[default_template]["description"]}</p>',
                    visible=True,
                )

                # Memory Optimization Section
                with gr.Accordion("🔧 Memory & Performance Settings", open=False):
                    # GPU Memory Status
                    def get_gpu_memory_info():
                        if torch.cuda.is_available():
                            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                            allocated_memory = torch.cuda.memory_allocated() / 1024**3
                            return f"🖥️ GPU: {torch.cuda.get_device_name()} | Total: {total_memory:.1f}GB | Used: {allocated_memory:.1f}GB"
                        else:
                            return "🖥️ GPU: Not available (using CPU)"
                    
                    with gr.Row():
                        gpu_memory_status = gr.HTML(
                            value=f'<p style="font-size: 0.9em; color: var(--body-text-color); margin: 5px 0; padding: 8px; background: var(--background-fill-secondary); border-radius: 6px;"> {get_gpu_memory_info()}</p>',
                            label="GPU Status"
                        )
                        refresh_memory_btn = gr.Button("🔄", size="sm", scale=0)
                    
                    # Quantization Selection
                    quantization_mode = gr.Dropdown(
                        label="🎛️ Quantization Mode",
                        choices=[
                            ("Auto (Recommended)", "auto"),
                            ("Full Precision (Highest Quality, ~18GB)", "none"),
                            ("8-bit (Balanced, ~12GB)", "8bit"),
                            ("6-bit (Good Quality, ~10GB)", "6bit"),
                            ("4-bit (Memory Saver, ~8GB)", "4bit")
                        ],
                        value="auto",
                        info="Choose quantization level. Auto selects based on your GPU memory. Lower bit = less memory but slightly lower quality.",
                    )
                    
                    # Memory optimization toggle
                    enable_memory_optimization = gr.Checkbox(
                        label="🧹 Enable Memory Optimization",
                        value=True,
                        info="Clear GPU cache before generation to prevent OOM errors"
                    )
                    
                    # Action buttons
                    with gr.Row():
                        reload_model_btn = gr.Button("🔄 Reload Model", variant="secondary", scale=2)
                        clear_memory_btn = gr.Button("🧹 Clear GPU Memory", variant="secondary", scale=1)
                    
                    # Quantization info
                    quantization_info = gr.HTML(
                        value='<p style="font-size: 0.8em; color: var(--body-text-color-subdued); margin: 5px 0; padding: 6px; background: var(--background-fill-primary); border-radius: 4px;"> 🔧 Auto mode will select optimal quantization based on your GPU memory</p>',
                        visible=True
                    )
                    
                    # Model status
                    model_status = gr.HTML(
                        value='<p style="font-size: 0.85em; color: var(--body-text-color-subdued); margin: 5px 0;"> Model not loaded yet</p>',
                        label="Model Status"
                    )

                system_prompt = gr.TextArea(
                    label="System Prompt",
                    placeholder="Enter system prompt to guide the model...",
                    value=PREDEFINED_EXAMPLES[default_template]["system_prompt"],
                    lines=2,
                )

                input_text = gr.TextArea(
                    label="Input Text",
                    placeholder="Type the text you want to convert to speech...",
                    value=PREDEFINED_EXAMPLES[default_template]["input_text"],
                    lines=5,
                )

                voice_preset = gr.Dropdown(
                    label="Voice Preset",
                    choices=list(VOICE_PRESETS.keys()),
                    value="EMPTY",
                    interactive=False,  # Disabled by default since default template is not voice-clone
                    visible=False,
                )

                with gr.Accordion(
                    "Custom Reference (Optional)", open=False, visible=False
                ) as custom_reference_accordion:
                    reference_audio = gr.Audio(label="Reference Audio", type="filepath")
                    reference_text = gr.TextArea(
                        label="Reference Text (transcript of the reference audio)",
                        placeholder="Enter the transcript of your reference audio...",
                        lines=3,
                    )

                # Smart Generation Features
                with gr.Accordion("🧠 Smart Generation Features", open=False):
                    enable_smart_chunking = gr.Checkbox(
                        label="📝 Enable Smart Chunking",
                        value=True,
                        info="✅ ENABLED by default - Optimized for 8-bit quantization and long texts"
                    )
                    
                    chunk_method = gr.Dropdown(
                        label="🎵 Crystal Clear Voice Cloning",
                        choices=[
                            ("Auto (Intelligent Chunking)", "auto"),
                            ("By Word Count (40 words max)", "word"),
                            ("By Sentence (Natural breaks)", "sentence")
                        ],
                        value="auto",
                        info="🎯 NEW: Crystal Clear Voice Cloning with optimal quality settings"
                    )
                    
                    enable_multi_speaker = gr.Checkbox(
                        label="🎭 Enable Multi-Speaker Detection",
                        value=False,
                        info="Enable only for multi-speaker dialogues with [SPEAKER0] tags"
                    )
                    
                    long_form_consistency = gr.Checkbox(
                        label="🔗 Long-Form Consistency",
                        value=True,
                        info="✅ ENABLED by default - Maintains voice consistency with seed-based generation"
                    )

                with gr.Accordion("Advanced Parameters", open=False):
                    max_completion_tokens = gr.Slider(
                        minimum=128,
                        maximum=4096,
                        value=1024,
                        step=10,
                        label="Max Completion Tokens",
                    )
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.5,
                        value=1.0,
                        step=0.1,
                        label="Temperature",
                    )
                    top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top P")
                    top_k = gr.Slider(minimum=-1, maximum=100, value=50, step=1, label="Top K")
                    ras_win_len = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=7,
                        step=1,
                        label="RAS Window Length",
                        info="Window length for repetition avoidance sampling",
                    )
                    ras_win_max_num_repeat = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=2,
                        step=1,
                        label="RAS Max Num Repeat",
                        info="Maximum number of repetitions allowed in the window",
                    )
                    # Add stop strings component
                    stop_strings = gr.Dataframe(
                        label="Stop Strings",
                        headers=["stops"],
                        datatype=["str"],
                        value=[[s] for s in DEFAULT_STOP_STRINGS],
                        interactive=True,
                        col_count=(1, "fixed"),
                    )

                submit_btn = gr.Button("Generate Speech", variant="primary", scale=1)

            with gr.Column(scale=2):
                output_text = gr.TextArea(label="Model Response", lines=2)

                # Audio output
                output_audio = gr.Audio(label="Generated Audio", interactive=False, autoplay=True)

                stop_btn = gr.Button("Stop Playback", variant="primary")

        # Example voice
        with gr.Row(visible=False) as voice_samples_section:
            voice_samples_table = gr.Dataframe(
                headers=["Voice Preset", "Sample Text"],
                datatype=["str", "str"],
                value=[[preset, text] for preset, text in VOICE_PRESETS.items() if preset != "EMPTY"],
                interactive=False,
            )
            sample_audio = gr.Audio(label="Voice Sample")

        # Function to play voice sample when clicking on a row
        def play_voice_sample(evt: gr.SelectData):
            try:
                # Get the preset name from the clicked row
                preset_names = [preset for preset in VOICE_PRESETS.keys() if preset != "EMPTY"]
                if evt.index[0] < len(preset_names):
                    preset = preset_names[evt.index[0]]
                    voice_path, _ = get_voice_preset(preset)
                    if voice_path and os.path.exists(voice_path):
                        return voice_path
                    else:
                        gr.Warning(f"Voice sample file not found for preset: {preset}")
                        return None
                else:
                    gr.Warning("Invalid voice preset selection")
                    return None
            except Exception as e:
                logger.error(f"Error playing voice sample: {e}")
                gr.Error(f"Error playing voice sample: {e}")
                return None

        voice_samples_table.select(fn=play_voice_sample, outputs=[sample_audio])

        # Function to handle template selection
        def apply_template(template_name):
            if template_name in PREDEFINED_EXAMPLES:
                template = PREDEFINED_EXAMPLES[template_name]
                # Enable voice preset and custom reference only for voice-clone template
                is_voice_clone = template_name == "voice-clone"
                voice_preset_value = "belinda" if is_voice_clone else "EMPTY"
                # Set ras_win_len to 0 for single-speaker-bgm, 7 for others
                ras_win_len_value = 0 if template_name == "single-speaker-bgm" else 7
                description_text = f'<p style="font-size: 0.85em; color: var(--body-text-color-subdued); margin: 0; padding: 0;"> {template["description"]}</p>'
                return (
                    template["system_prompt"],  # system_prompt
                    template["input_text"],  # input_text
                    description_text,  # template_description
                    gr.update(
                        value=voice_preset_value, interactive=is_voice_clone, visible=is_voice_clone
                    ),  # voice_preset (value and interactivity)
                    gr.update(visible=is_voice_clone),  # custom reference accordion visibility
                    gr.update(visible=is_voice_clone),  # voice samples section visibility
                    ras_win_len_value,  # ras_win_len
                )
            else:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )  # No change if template not found

        # Set up event handlers

        # Connect template dropdown to handler
        template_dropdown.change(
            fn=apply_template,
            inputs=[template_dropdown],
            outputs=[
                system_prompt,
                input_text,
                template_description,
                voice_preset,
                custom_reference_accordion,
                voice_samples_section,
                ras_win_len,
            ],
        )

        # Connect submit button to the TTS function
        submit_btn.click(
            fn=text_to_speech,
            inputs=[
                input_text,
                voice_preset,
                reference_audio,
                reference_text,
                max_completion_tokens,
                temperature,
                top_p,
                top_k,
                system_prompt,
                stop_strings,
                ras_win_len,
                ras_win_max_num_repeat,
                enable_smart_chunking,
                chunk_method,
                enable_multi_speaker,
                long_form_consistency,
            ],
            outputs=[output_text, output_audio],
            api_name="generate_speech",
        )

        # Stop button functionality
        stop_btn.click(
            fn=lambda: None,
            inputs=[],
            outputs=[output_audio],
            js="() => {const audio = document.querySelector('audio'); if(audio) audio.pause(); return null;}",
        )

        # Memory optimization event handlers
        reload_model_btn.click(
            fn=reload_model_with_settings,
            inputs=[quantization_mode, enable_memory_optimization],
            outputs=[model_status],
        )

        # Update GPU memory status periodically (when UI loads)
        demo.load(
            fn=update_gpu_memory_status,
            inputs=[],
            outputs=[gpu_memory_status],
        )

        # Refresh memory status button
        refresh_memory_btn.click(
            fn=update_gpu_memory_status,
            inputs=[],
            outputs=[gpu_memory_status],
        )

        # Update quantization info when mode changes
        quantization_mode.change(
            fn=update_quantization_info,
            inputs=[quantization_mode],
            outputs=[quantization_info],
        )

        # Clear memory button
        def manual_clear_memory():
            if OPTIMIZATION_AVAILABLE:
                clear_memory()
                return update_gpu_memory_status()
            else:
                return "Memory optimization not available"

        clear_memory_btn.click(
            fn=manual_clear_memory,
            inputs=[],
            outputs=[gpu_memory_status],
        )

    return demo


def main():
    """Main function to parse arguments and launch the UI."""
    global DEFAULT_MODEL_PATH, DEFAULT_AUDIO_TOKENIZER_PATH, VOICE_PRESETS, DEVICE_ARG

    parser = argparse.ArgumentParser(description="Gradio UI for Text-to-Speech using HiggsAudioServeEngine")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run the model on.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for the Gradio interface.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio interface.")

    args = parser.parse_args()

    # Set global device argument
    DEVICE_ARG = args.device
    logger.info(f"Using device: {DEVICE_ARG}")

    # Update default values if provided via command line
    VOICE_PRESETS = load_voice_presets()

    # Create and launch the UI
    demo = create_ui()
    # Try the specified port, if busy, let Gradio find an available one
    try:
        demo.launch(server_name=args.host, server_port=args.port, share=False, inbrowser=False)
    except OSError as e:
        if "Cannot find empty port" in str(e):
            logger.warning(f"Port {args.port} is busy, letting Gradio find an available port...")
            demo.launch(server_name=args.host, server_port=0, share=False, inbrowser=False)  # 0 = auto-find port
        else:
            raise e


if __name__ == "__main__":
    main()
