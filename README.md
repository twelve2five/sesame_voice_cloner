# Sesame CSM Voice Cloning with Whisper ASR

A voice cloning application that combines Sesame AI's Controllable Speech Model (CSM) for high-quality voice synthesis with OpenAI's Whisper model for Automatic Speech Recognition (ASR).

## Demo

[**Watch the Demo Video**](https://drive.google.com/file/d/1Z663i71kxpE8-dFMtSgJvD3xxB4VMdua/view?usp=sharing)

*This demo shows the complete process running in Google Colab.*

## Quick Start

### Google Colab (Recommended)
- **Google Colab Users**: Open and run `VC_collab_file.py` in Google Colab for the complete experience with GPU acceleration.
- The demo video shows the process running in Google Colab.
- This is the easiest way to get started without any local setup.

### Local Installation
To run the application locally:

1. **Clone the repository**:
   ```
   git clone https://github.com/twelve2five/sesame_voice_cloner.git
   cd sesame_voice_cloner
   ```

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Prepare your environment**:
   - Get a HuggingFace token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - A CUDA-capable GPU is highly recommended for reasonable performance

4. **Run the application**:
   ```
   python voice_cloning.py
   ```

5. **Follow the prompts** to:
   - Add voice samples to the `voice_samples` directory
   - Process the samples with Whisper transcription
   - Generate speech with your cloned voice

## Features
- Record and upload voice samples for cloning
- Automatic transcription of voice samples using Whisper ASR
- High-quality voice cloning with Sesame's CSM technology
- Parameter tuning for voice generation control

## Technologies
- **Sesame AI CSM**: State-of-the-art voice cloning model that captures voice characteristics
- **OpenAI Whisper**: Advanced speech recognition for accurate transcription

## System Requirements
- Python 3.7+
- CUDA-capable GPU recommended for faster processing
- Approximately 10GB disk space for models
- At least 8GB RAM (16GB+ recommended)

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details. 
