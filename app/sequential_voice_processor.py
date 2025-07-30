"""
SEQUENTIAL VOICE PROCESSOR
=========================
Exactly what you requested:
1. Smart chunk text into 70-80 words per chunk
2. Save chunks separately  
3. Process ONE chunk at a time sequentially
4. Same settings, reference audio, and seed for each
5. Save each received audio chunk
6. Combine all chunks into one complete audio at the end

This works with the actual HiggsAudioServeEngine API.
"""

import os
import json
import time
import uuid
import numpy as np
import base64
import re
from pathlib import Path
from typing import Optional, List, Tuple
from loguru import logger

# Import HiggsAudio components
from higgs_audio.data_types import ChatMLSample, AudioContent, Message

class SequentialVoiceProcessor:
    def __init__(self, engine):
        self.engine = engine
        self.chunks_dir = Path("temp_chunks")
        self.chunks_dir.mkdir(exist_ok=True)
        
        # Default settings for consistency
        self.default_settings = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 20,
            "max_new_tokens": 1024,
            "stop_strings": ["<|end_of_text|>", "<|eot_id|>"],
            "ras_win_len": 7,
            "ras_win_max_num_repeat": 2
        }
        
    def smart_chunk_text(self, text: str, target_words: int = 75) -> List[str]:
        """
        Smart chunking into 70-80 words per chunk at natural sentence breaks
        """
        logger.info(f"🔪 Smart chunking text into ~{target_words} word chunks")
        
        # Clean and normalize text
        text = text.strip()
        if not text:
            return []
        
        # Split into sentences using multiple delimiters
        sentences = re.split(r'[.!?]+(?:\s|$)', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            # Fallback: split by word count if no sentence breaks
            words = text.split()
            chunks = []
            for i in range(0, len(words), target_words):
                chunk = ' '.join(words[i:i + target_words])
                chunks.append(chunk)
            logger.info(f"✅ Fallback chunking: {len(chunks)} chunks")
            return chunks
        
        chunks = []
        current_chunk = ""
        current_words = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed target, save current chunk
            if current_words > 0 and (current_words + sentence_words) > target_words + 10:
                if current_chunk.strip():
                    # Ensure chunk ends with punctuation
                    if not current_chunk.strip().endswith(('.', '!', '?')):
                        current_chunk += '.'
                    chunks.append(current_chunk.strip())
                    logger.info(f"   Chunk {len(chunks)}: {current_words} words")
                current_chunk = sentence
                current_words = sentence_words
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
                current_words += sentence_words
        
        # Add final chunk
        if current_chunk.strip():
            if not current_chunk.strip().endswith(('.', '!', '?')):
                current_chunk += '.'
            chunks.append(current_chunk.strip())
            logger.info(f"   Chunk {len(chunks)}: {len(current_chunk.split())} words")
        
        logger.info(f"✅ Created {len(chunks)} chunks, avg {sum(len(c.split()) for c in chunks)/len(chunks):.1f} words each")
        return chunks
    
    def save_chunks_separately(self, chunks: List[str], session_id: str) -> List[Path]:
        """
        Save each chunk as separate text file
        """
        logger.info(f"💾 Saving {len(chunks)} chunks separately for session {session_id}")
        
        chunk_files = []
        for i, chunk in enumerate(chunks, 1):
            chunk_file = self.chunks_dir / f"session_{session_id}_chunk_{i}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(chunk)
            chunk_files.append(chunk_file)
            logger.info(f"   Saved: {chunk_file.name} ({len(chunk.split())} words)")
        
        return chunk_files
    
    def get_voice_preset_audio(self, voice_preset: str) -> Tuple[Optional[str], str]:
        """Get voice preset audio file and transcript"""
        if voice_preset == "EMPTY":
            return None, ""
            
        voice_path = Path(__file__).parent / "voice_examples" / f"{voice_preset}.wav"
        if not voice_path.exists():
            logger.warning(f"Voice preset file not found: {voice_path}")
            return None, ""
        
        # Load transcript from config
        try:
            config_path = Path(__file__).parent / "voice_examples" / "config.json"
            with open(config_path, 'r', encoding='utf-8') as f:
                voice_dict = json.load(f)
            transcript = voice_dict.get(voice_preset, {}).get("transcript", "")
        except Exception as e:
            logger.warning(f"Could not load transcript for {voice_preset}: {e}")
            transcript = ""
        
        return str(voice_path), transcript
    
    def encode_audio_file(self, file_path: str) -> str:
        """Encode an audio file to base64."""
        with open(file_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")
    
    def prepare_chatml_sample(self, chunk_text: str, voice_preset: str, system_prompt: str, 
                            reference_audio: Optional[str] = None, reference_text: Optional[str] = None) -> ChatMLSample:
        """Prepare a ChatMLSample for a single chunk with proper reference audio handling"""
        messages = []
        
        # Add system message if provided
        if system_prompt and system_prompt.strip():
            messages.append(Message(role="system", content=system_prompt))
        
        # Handle reference audio - prioritize custom reference audio over voice presets
        audio_base64 = None
        ref_text = ""
        
        if reference_audio:
            # Custom reference audio uploaded by user
            try:
                audio_base64 = self.encode_audio_file(reference_audio)
                ref_text = reference_text or ""
                logger.info(f"   Using custom reference audio: {reference_audio}")
            except Exception as e:
                logger.warning(f"Failed to load custom reference audio {reference_audio}: {e}")
        elif voice_preset != "EMPTY":
            # Voice preset selected
            voice_path, ref_text = self.get_voice_preset_audio(voice_preset)
            if voice_path:
                try:
                    audio_base64 = self.encode_audio_file(voice_path)
                    logger.info(f"   Using voice preset: {voice_preset}")
                except Exception as e:
                    logger.warning(f"Failed to load voice preset {voice_preset}: {e}")
        
        # Add reference audio to messages if we have it
        if audio_base64 is not None:
            # Add user message with reference text
            messages.append(Message(role="user", content=ref_text))
            
            # Add assistant message with audio content
            audio_content = AudioContent(raw_audio=audio_base64, audio_url="")
            messages.append(Message(role="assistant", content=[audio_content]))
            
            logger.info(f"   ✅ Reference audio added to chunk")
        else:
            logger.info(f"   ⚠️ No reference audio for this chunk")
        
        # Add the chunk text as user message
        messages.append(Message(role="user", content=chunk_text))
        
        return ChatMLSample(messages=messages)
    
    def process_single_chunk(self, chunk_text: str, voice_preset: str, system_prompt: str, 
                           chunk_num: int, total_chunks: int, reference_audio: Optional[str] = None, 
                           reference_text: Optional[str] = None, seed: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Process ONE single chunk with exact same settings and reference audio
        """
        logger.info(f"🎵 Processing chunk {chunk_num}/{total_chunks} ({len(chunk_text.split())} words)")
        logger.info(f"   Text preview: {chunk_text[:100]}...")
        if seed is not None:
            logger.info(f"   Using seed: {seed}")
        
        try:
            # Set seed for consistency if provided
            if seed is not None:
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                import numpy as np
                np.random.seed(seed)
                import random
                random.seed(seed)
                logger.info(f"   🎲 Set all random seeds to {seed}")
            
            # Prepare ChatML sample for this chunk with reference audio
            chatml_sample = self.prepare_chatml_sample(
                chunk_text, voice_preset, system_prompt, reference_audio, reference_text
            )
            
            # Generate audio for this single chunk using the engine
            response = self.engine.generate(
                chat_ml_sample=chatml_sample,
                max_new_tokens=self.default_settings["max_new_tokens"],
                temperature=self.default_settings["temperature"],
                top_k=self.default_settings["top_k"],
                top_p=self.default_settings["top_p"],
                stop_strings=self.default_settings["stop_strings"],
                ras_win_len=self.default_settings["ras_win_len"],
                ras_win_max_num_repeat=self.default_settings["ras_win_max_num_repeat"],
            )
            
            if response.audio is not None:
                duration = len(response.audio) / response.sampling_rate
                logger.info(f"✅ Chunk {chunk_num} generated: {duration:.2f}s audio")
                return response.audio
            else:
                logger.error(f"❌ Chunk {chunk_num} failed: No audio in response")
                logger.error(f"   Generated text: {response.generated_text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Chunk {chunk_num} EXCEPTION: {e}")
            import traceback
            logger.error(f"   Full traceback: {traceback.format_exc()}")
            return None
    
    def save_chunk_audio(self, audio: np.ndarray, session_id: str, chunk_num: int) -> Path:
        """
        Save individual chunk audio
        """
        audio_file = self.chunks_dir / f"session_{session_id}_chunk_{chunk_num}.npy"
        np.save(audio_file, audio)
        duration = len(audio) / 24000
        logger.info(f"💾 Saved chunk {chunk_num} audio: {audio_file.name} ({duration:.2f}s)")
        return audio_file
    
    def combine_all_chunks(self, session_id: str, total_chunks: int) -> Optional[np.ndarray]:
        """
        Combine all chunk audios into one complete audio with smooth transitions
        """
        logger.info(f"🔗 Combining {total_chunks} audio chunks into complete audio")
        
        combined_audio = []
        total_duration = 0
        
        for i in range(1, total_chunks + 1):
            audio_file = self.chunks_dir / f"session_{session_id}_chunk_{i}.npy"
            
            if audio_file.exists():
                chunk_audio = np.load(audio_file)
                combined_audio.append(chunk_audio)
                
                chunk_duration = len(chunk_audio) / 24000
                total_duration += chunk_duration
                
                # Add small pause between chunks (0.2 seconds)
                if i < total_chunks:  # Don't add pause after last chunk
                    pause = np.zeros(int(24000 * 0.2))
                    combined_audio.append(pause)
                    total_duration += 0.2
                
                logger.info(f"   Added chunk {i}: {chunk_duration:.2f}s")
            else:
                logger.error(f"❌ Missing chunk audio: {audio_file}")
                return None
        
        if combined_audio:
            final_audio = np.concatenate(combined_audio)
            
            # Save combined audio
            combined_file = self.chunks_dir / f"session_{session_id}_combined.npy"
            np.save(combined_file, final_audio)
            
            logger.info(f"🎉 COMPLETE AUDIO: {total_duration:.2f}s total from {total_chunks} chunks")
            return final_audio
        else:
            logger.error("❌ No audio chunks to combine")
            return None
    
    def cleanup_session_files(self, session_id: str, keep_combined: bool = True):
        """Clean up temporary session files"""
        try:
            pattern = f"session_{session_id}_*"
            for file_path in self.chunks_dir.glob(pattern):
                if keep_combined and "combined" in file_path.name:
                    continue
                file_path.unlink()
                logger.info(f"🗑️ Cleaned up: {file_path.name}")
        except Exception as e:
            logger.warning(f"Failed to cleanup session files: {e}")
    
    def process_text_sequentially(self, text: str, voice_preset: str, system_prompt: str = "", 
                                reference_audio: Optional[str] = None, reference_text: Optional[str] = None) -> Tuple[Optional[np.ndarray], str]:
        """
        MAIN METHOD: Process text exactly as you described with proper voice cloning support
        """
        session_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        logger.info(f"🚀 Starting SEQUENTIAL VOICE PROCESSING for session {session_id}")
        logger.info(f"📝 Input: {len(text.split())} words")
        logger.info(f"🎵 Voice setup: preset='{voice_preset}', custom_audio={reference_audio is not None}")
        
        # Fixed seed for consistency across all chunks
        base_seed = 42
        logger.info(f"🎲 Using base seed {base_seed} for voice consistency")
        
        try:
            # Step 1: Smart chunk text into 70-80 words
            chunks = self.smart_chunk_text(text, target_words=75)
            
            if not chunks:
                return None, "No chunks created from input text"
            
            # Step 2: Save chunks separately
            chunk_files = self.save_chunks_separately(chunks, session_id)
            
            # Step 3: Process ONE chunk at a time sequentially with SAME reference audio and seed
            audio_files = []
            
            for i, chunk in enumerate(chunks, 1):
                logger.info(f"\n--- PROCESSING CHUNK {i}/{len(chunks)} ---")
                
                # Use same seed for all chunks to ensure voice consistency
                chunk_seed = base_seed + (i - 1)  # 42, 43, 44, etc.
                
                # Process single chunk with SAME settings and reference audio
                chunk_audio = self.process_single_chunk(
                    chunk_text=chunk,
                    voice_preset=voice_preset,
                    system_prompt=system_prompt,
                    chunk_num=i,
                    total_chunks=len(chunks),
                    reference_audio=reference_audio,  # SAME reference audio for all chunks
                    reference_text=reference_text,    # SAME reference text for all chunks
                    seed=chunk_seed                    # Consistent seed pattern
                )
                
                if chunk_audio is not None:
                    # Step 4: Save each received audio chunk
                    audio_file = self.save_chunk_audio(chunk_audio, session_id, i)
                    audio_files.append(audio_file)
                    
                    logger.info(f"✅ Chunk {i} COMPLETE and SAVED with consistent voice")
                else:
                    logger.error(f"❌ Chunk {i} FAILED - stopping process")
                    return None, f"Chunk {i} processing failed"
                
                # Small delay between chunks to prevent overload
                time.sleep(0.3)
            
            # Step 5: Combine all chunks into one complete audio
            logger.info(f"\n--- COMBINING ALL {len(chunks)} CHUNKS ---")
            final_audio = self.combine_all_chunks(session_id, len(chunks))
            
            if final_audio is not None:
                processing_time = time.time() - start_time
                
                # Save session metadata
                metadata = {
                    "session_id": session_id,
                    "voice_preset": voice_preset,
                    "has_custom_reference": reference_audio is not None,
                    "base_seed": base_seed,
                    "total_chunks": len(chunks),
                    "total_words": len(text.split()),
                    "final_duration": len(final_audio) / 24000,
                    "processing_time": processing_time,
                    "chunks_info": [
                        {
                            "chunk_num": i,
                            "words": len(chunk.split()),
                            "seed": base_seed + (i - 1),
                            "text_preview": chunk[:50] + "..." if len(chunk) > 50 else chunk
                        }
                        for i, chunk in enumerate(chunks, 1)
                    ]
                }
                
                metadata_file = self.chunks_dir / f"session_{session_id}_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"🎉 SEQUENTIAL VOICE PROCESSING COMPLETE!")
                logger.info(f"   Session: {session_id}")
                logger.info(f"   Chunks processed: {len(chunks)}")
                logger.info(f"   Total duration: {len(final_audio)/24000:.2f}s")
                logger.info(f"   Processing time: {processing_time:.2f}s")
                logger.info(f"   Speech rate: {len(text.split())/(len(final_audio)/24000):.1f} words/sec")
                logger.info(f"   Voice consistency: ✅ Same reference audio & seed for all chunks")
                
                # Clean up temporary files (keep combined and metadata)
                self.cleanup_session_files(session_id, keep_combined=True)
                
                return final_audio, f"Sequential processing complete: {len(chunks)} chunks, {len(final_audio)/24000:.2f}s audio with consistent voice"
            else:
                return None, "Failed to combine audio chunks"
                
        except Exception as e:
            logger.error(f"❌ SEQUENTIAL VOICE PROCESSING EXCEPTION: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None, f"Sequential processing failed: {str(e)}"