# Clone the CSM repository if needed
!git clone https://github.com/SesameAILabs/csm.git
%cd csm

# Install dependencies
!pip install -r requirements.txt
!pip install openai-whisper

# Import needed libraries
from huggingface_hub import hf_hub_download, login
import sys
sys.path.append('./') # Ensure CSM modules are in path
from generator import load_csm_1b, Segment
import torchaudio
import torch
import os
from google.colab import files
from IPython.display import display, Audio
import numpy as np
import whisper
import re

# Simplified preprocessing function
def simple_text_preprocessing(text):
    """
    Very simple preprocessing that just removes emojis and replaces problematic characters
    with basic ASCII equivalents.
    """
    # Step 1: Remove all emojis completely (rather than converting them)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Step 2: Simple character replacements (minimal)
    replacements = {
        '—': '-',      # em dash
        '–': '-',      # en dash
        '…': '...',    # ellipsis
        ''': "'",      # smart single quote
        ''': "'",      # smart single quote
        '"': '"',      # smart double quote
        '"': '"',      # smart double quote
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # Step 3: Remove any remaining non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Step 4: Clean up excessive spacing
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Load Whisper large model for maximum transcription accuracy
print("Loading Whisper large model (this may take a minute)...")
whisper_model = whisper.load_model("large")
print("Whisper large model loaded!")

# Authenticate with Hugging Face
login()  # This will prompt for your token

# Set device to A100
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the CSM model
print("Downloading CSM model (this may take a few minutes)...")
model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
print(f"Model downloaded to: {model_path}")

generator = load_csm_1b(model_path, device)
print("Model loaded successfully")

# Create directory for voice samples
!mkdir -p voice_samples

# Phonetically balanced sentences specifically designed for voice cloning
voice_capture_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "The rainbow's vibrant colors stretched across the sky after the storm.",
    "My voice has a unique timbre and natural cadence that defines who I am.",
    "Scientists discovered a new species of deep-sea creatures last month.",
    "The rhythm of my speech includes pauses, emphasis, and tonal variation.",
    "Please call Stella, ask her to bring these things with her from the store.",
    "She sells seashells by the seashore on sunny summer days.",
    "The technology can capture how I pronounce vowels and consonants distinctly.",
    "We were away a year and a day in the sleepy tropical paradise.",
    "My speaking style reflects my personality through subtle vocal characteristics."
]

# Display recording instructions with better guidance
print("\n==== VOICE SAMPLE RECORDING INSTRUCTIONS ====")
print("For best results:")
print("1. Record in a quiet environment with minimal background noise")
print("2. Use a good microphone and position it consistently")
print("3. Speak naturally at your normal pace and tone")
print("4. Maintain consistent volume and distance from the microphone")
print("5. You can speak any content - Whisper will automatically transcribe it")
print("6. For best results, try reading some of the suggested sentences below")
print("\nSuggested sentences (you don't have to use these exactly):")

for i, sentence in enumerate(voice_capture_sentences):
    print(f"\n{i+1}. {sentence}")

print("\nAfter recording, upload each file one by one in the next steps.")
print("===================================\n")

# Initialize transcript dictionary
transcripts = {}

# Function to process audio
def load_audio(audio_path):
    """Load and preprocess audio file for optimal use with CSM"""
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    
    # Convert stereo to mono if needed
    if audio_tensor.shape[0] > 1:
        audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
    
    # Normalize audio volume (improves consistency)
    audio_tensor = audio_tensor / (torch.max(torch.abs(audio_tensor)) + 1e-8)
    
    # Resample to CSM's sample rate
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    
    return audio_tensor

# Process each file individually with Whisper transcription
for i in range(1, 6):  # Upload 5 files
    print(f"\n=== Uploading voice sample {i}/5 ===")
    print("Select an audio file (.wav, .mp3, or .m4a):")
    uploaded = files.upload()
    
    if not uploaded:
        print("No file uploaded. Continuing to next...")
        continue
        
    filename = list(uploaded.keys())[0]
    dest_path = f"voice_samples/{filename}"
    !mv "{filename}" "{dest_path}"
    print(f"Uploaded and moved {filename}")
    
    # Play the uploaded file
    print("\nPlaying your recording:")
    display(Audio(dest_path))
    
    # Automatically transcribe with Whisper large
    print("Transcribing with Whisper large model...")
    result = whisper_model.transcribe(dest_path)
    auto_transcript = result["text"].strip()
    print(f"Whisper transcription: \"{auto_transcript}\"")
    
    # Ask user to accept or modify
    transcript_choice = input(f"\nPress Enter to accept Whisper's transcript, or type a corrected transcript: ")
    
    if not transcript_choice:
        # User accepted the Whisper transcript
        transcripts[filename] = auto_transcript
        print(f"Using Whisper transcript: {auto_transcript}")
    else:
        # User provided a correction
        transcripts[filename] = transcript_choice
        print(f"Using corrected transcript: {transcript_choice}")

# List all files from voice_samples directory (includes all audio formats)
voice_samples = [os.path.join("voice_samples", f) for f in os.listdir("voice_samples") 
                 if f.endswith(('.wav', '.mp3', '.m4a'))]
voice_samples.sort()
print(f"\nFound {len(voice_samples)} voice samples")

# Print transcript summary for verification
print("\n=== Transcript Summary ===")
for sample in voice_samples:
    filename = os.path.basename(sample)
    if filename in transcripts:
        print(f"{filename}: \"{transcripts[filename]}\"")
    else:
        print(f"{filename}: No transcript assigned")

# Create voice context with additional quality checks
target_speaker_id = 0  # Consistent speaker ID

def prepare_voice_context(voice_samples, transcripts, target_speaker_id):
    context_segments = []
    
    for sample in voice_samples:
        filename = os.path.basename(sample)
        if filename in transcripts:
            try:
                audio = load_audio(sample)
                duration = audio.shape[0]/generator.sample_rate
                
                # Check for minimum quality criteria
                if duration < 1.0:
                    print(f"Warning: {filename} is too short ({duration:.2f}s). Should be > 1s")
                elif duration > 10.0:
                    print(f"Warning: {filename} is quite long ({duration:.2f}s). May be truncated")
                
                # Add with detailed stats
                context_segments.append(
                    Segment(
                        text=transcripts[filename],
                        speaker=target_speaker_id,
                        audio=audio
                    )
                )
                print(f"Added sample: {filename} ({duration:.2f}s, {audio.shape[0]} samples)")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return context_segments

# Build the context segments
context_segments = prepare_voice_context(voice_samples, transcripts, target_speaker_id)

# Print stats about the context
print(f"\nUsing {len(context_segments)} voice samples as context")
total_duration = sum([seg.audio.shape[0] for seg in context_segments]) / generator.sample_rate
print(f"Total context duration: {total_duration:.2f} seconds")

if len(context_segments) < 3:
    print("WARNING: For good voice cloning, 5+ samples are recommended")

# Function to generate and play audio with simplified emoji preprocessing
def generate_speech(text_input, temperature=0.9, topk=50):
    # Apply simple preprocessing
    original_text = text_input
    processed_text = simple_text_preprocessing(text_input)
    
    # Log preprocessing if changes were made
    if processed_text != original_text:
        print(f"Original: {original_text}")
        print(f"Simplified: {processed_text}")
    
    print(f"Generating: \"{processed_text}\"")
    print(f"Using temperature={temperature}, topk={topk}")
    
    new_audio = generator.generate(
        text=processed_text,
        speaker=target_speaker_id,
        context=context_segments,
        max_audio_length_ms=90000,  # 90 seconds max
        temperature=temperature,
        topk=topk
    )
    
    # Save the generated audio
    output_filename = f"cloned_voice_{len(os.listdir('.'))}.wav"
    torchaudio.save(output_filename, new_audio.unsqueeze(0).cpu(), generator.sample_rate)
    print(f"Audio saved to {output_filename}")
    
    # Play the audio
    display(Audio(output_filename, autoplay=True))
    
    # Download option
    files.download(output_filename)
    
    return output_filename

# Interactive generation with parameter tuning
print("\n==== VOICE GENERATION ====")
print("You can now generate speech with your cloned voice.")
print("Try different temperatures (0.7-1.0) and topk values (20-100) to find the best quality")
print("Note: This version has improved handling for emojis and special characters")

# First generation with default parameters
text_input = input("Enter text to speak in the cloned voice: ")
generate_speech(text_input)

# Optional: Allow parameter tuning
tune = input("\nWould you like to try different parameters for better quality? (y/n): ")
if tune.lower() == 'y':
    while True:
        text_input = input("\nEnter text (or 'quit' to exit): ")
        if text_input.lower() == 'quit':
            break
            
        temp = input("Temperature (0.7-1.0, default=0.9): ")
        temp = float(temp) if temp else 0.9
        
        topk_val = input("Top-k (20-100, default=50): ")
        topk_val = int(topk_val) if topk_val else 50
        
        generate_speech(text_input, temp, topk_val)