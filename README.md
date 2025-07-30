# Higgs Audio v2 UI

A powerful Gradio-based web interface for the Higgs Audio v2 text-to-speech model with advanced features including sequential chunk processing for long-form text generation and voice cloning.

## ✨ Features

### 🎵 **Sequential Voice Processing**
- **Smart chunking**: Automatically splits long text into 70-80 word chunks at natural sentence boundaries
- **Voice consistency**: Maintains the same voice characteristics across all chunks
- **Memory efficient**: Works with 8-bit quantization on 16GB GPUs
- **Perfect for long-form content**: Stories, articles, books, etc.

### 🎤 **Voice Cloning**
- **Custom reference audio**: Upload your own audio file for voice cloning
- **Voice presets**: Choose from built-in voice examples
- **Consistent cloning**: Same reference audio used for all chunks in long text
- **High quality**: Maintains voice characteristics throughout generation

### 🚀 **Advanced Features**
- **8-bit quantization**: Memory-optimized model loading
- **Multi-speaker support**: Handle conversations and dialogues
- **Smart voice**: Context-aware speech generation
- **BGM support**: Background music integration (experimental)
- **Multiple languages**: Support for English, Chinese, and more

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ GPU memory for optimal performance

### Quick Start with Pinokio

1. **Install via Pinokio** (Recommended):
   - Open Pinokio
   - Install this repository
   - Models will be automatically downloaded
   - Launch the interface

2. **Manual Installation**:
   ```bash
   git clone https://github.com/AshutoshMipax/higgs-audio-v2-ui.git
   cd higgs-audio-v2-ui
   pip install -r requirements.txt
   python app/app.py
   ```

## 🎯 Usage

### Basic Text-to-Speech
1. Enter your text in the input field
2. Select generation parameters
3. Click "Generate"
4. Download your audio

### Voice Cloning
1. **Method 1 - Upload Reference Audio**:
   - Upload your reference audio file
   - Enter reference text (optional)
   - Enter your target text
   - Generate with consistent voice

2. **Method 2 - Use Voice Presets**:
   - Select a voice preset from the dropdown
   - Enter your target text
   - Generate with preset voice

### Long-Form Text Generation
- **Automatic**: Text over 100 words automatically uses sequential processing
- **Manual**: Enable for any text by selecting a voice preset
- **Progress**: Watch detailed logs showing chunk processing

## 📊 Sequential Processing

The system automatically handles long text by:

1. **Smart Chunking**: Splits text into 70-80 word chunks at sentence boundaries
2. **Sequential Processing**: Processes one chunk at a time (never parallel)
3. **Voice Consistency**: Uses same reference audio and seed pattern for all chunks
4. **Audio Combination**: Combines all chunks into seamless final audio

### Example Log Output:
```
🚀 Starting SEQUENTIAL VOICE PROCESSING for session a1b2c3d4
🔪 Smart chunking text into ~75 word chunks
   Chunk 1: 77 words
   Chunk 2: 68 words
   Chunk 3: 70 words
✅ Created 3 chunks, avg 71.7 words each

--- PROCESSING CHUNK 1/3 ---
🎵 Processing chunk 1/3 (77 words)
✅ Chunk 1 generated: 3.2s audio

🎉 SEQUENTIAL VOICE PROCESSING COMPLETE!
   Final audio: 9.3s total from 3 chunks
```

## ⚙️ Configuration

### Model Settings
- **Quantization**: Auto, 4-bit, 8-bit, or None
- **Memory Optimization**: Enable for better GPU memory usage
- **Device**: Auto-detect CUDA or CPU

### Generation Parameters
- **Temperature**: Controls randomness (0.1-2.0)
- **Top-p**: Nucleus sampling (0.1-1.0)
- **Top-k**: Top-k sampling (1-100)
- **Max Tokens**: Maximum generation length

## 📁 Project Structure

```
higgs-audio-v2-ui/
├── app/
│   ├── app.py                          # Main Gradio application
│   ├── sequential_voice_processor.py   # Sequential processing engine
│   ├── optimized_serve_engine.py      # Memory-optimized model loader
│   ├── memory_config.py               # Memory optimization settings
│   ├── voice_examples/                # Voice preset files
│   └── higgs_audio/                   # Higgs Audio model code
├── install.json                       # Pinokio installation config
├── start.json                         # Pinokio startup config
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory**:
- Enable 8-bit quantization
- Reduce max_completion_tokens
- Use CPU fallback

**Voice Inconsistency**:
- Ensure reference audio is clear and high-quality
- Use shorter chunks (adjust target_words in processor)
- Check that same reference audio is used for all chunks

**Slow Generation**:
- Enable memory optimization
- Use GPU if available
- Reduce generation parameters

### Debug Logs
The application provides detailed logging for troubleshooting:
- Sequential processing progress
- Memory usage information
- Voice cloning status
- Error messages with solutions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Higgs Audio Team](https://github.com/boson-ai/higgs-audio) for the amazing TTS model
- Gradio team for the excellent web interface framework
- The open-source community for inspiration and support

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Review the logs for error messages
3. Open an issue on GitHub with detailed information

---

**Enjoy creating amazing audio with Higgs Audio v2 UI!** 🎵✨