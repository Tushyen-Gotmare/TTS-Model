import streamlit as st
import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import numpy as np
import pandas as pd
import librosa
import base64
from pathlib import Path
import time

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 16px 24px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .upload-box {
        border: 2px dashed #ccc;
        padding: 20px;
        text-align: center;
        border-radius: 8px;
    }
    .success-box {
        padding: 20px;
        background-color: #e8f5e9;
        border-radius: 8px;
        margin: 10px 0;
    }
    .error-box {
        padding: 20px;
        background-color: #ffebee;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'generated_audio' not in st.session_state:
    st.session_state.generated_audio = None
if 'progress' not in st.session_state:
    st.session_state.progress = 0

@st.cache_resource
def load_models():
    """Load TTS models with progress tracking"""
    try:
        with st.spinner("🔄 Loading models..."):
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Load processor
            progress_bar.progress(25)
            processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            
            # Load TTS model
            progress_bar.progress(50)
            model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            
            # Load vocoder
            progress_bar.progress(75)
            vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            
            # Create speaker embedding
            speaker_embedding = torch.randn(1, 512)
            
            progress_bar.progress(100)
            st.success("✅ Models loaded successfully!")
            
            return processor, model, vocoder, speaker_embedding
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None

def generate_speech(text, processor, model, vocoder, speaker_embedding):
    """Generate speech with progress tracking"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Tokenization
        status_text.text("Converting text to tokens...")
        progress_bar.progress(25)
        inputs = processor(text=text, return_tensors="pt")
        
        # Generate speech
        status_text.text("Generating speech...")
        progress_bar.progress(50)
        with torch.no_grad():  # Prevent gradient computation
            speech = model.generate_speech(
                inputs["input_ids"], 
                speaker_embeddings=speaker_embedding
            )
        
        # Convert to waveform
        status_text.text("Converting to audio waveform...")
        progress_bar.progress(75)
        with torch.no_grad():  # Prevent gradient computation
            audio = vocoder(speech)
            audio = audio.detach().cpu().numpy()  # Properly detach and convert to numpy
        
        # Finalize
        progress_bar.progress(100)
        status_text.text("✨ Speech generation complete!")
        time.sleep(0.5)
        status_text.empty()
        
        return audio
    except Exception as e:
        st.error(f"❌ Error generating speech: {str(e)}")
        return None

def main():
    # Header
    st.title("🎤 Fine-tuning Text-to-Speech (TTS)")
    st.markdown("Convert text to natural-sounding speech with fine-tuning capabilities")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🔊 Generate Speech", "⚙️ Fine-tune Model"])
    
    # Load models
    processor, model, vocoder, speaker_embedding = load_models()
    
    with tab1:
        st.header("Generate Speech")
        
        # Text input with character counter
        text = st.text_area(
            "Enter text to convert to speech",
            height=150,
            max_chars=1000,
            help="Type or paste the text you want to convert to speech (max 1000 characters)"
        )
        st.caption(f"Characters: {len(text)}/1000")
        
        # Generation controls
        col1, col2 = st.columns([3, 1])
        with col1:
            speed = st.slider(
                "Speech Rate",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Adjust the speed of the generated speech"
            )
        
        # Generate button
        if st.button("🔊 Generate Speech", type="primary"):
            if not text.strip():
                st.warning("⚠️ Please enter some text first!")
                return
            
            with st.spinner("🎵 Generating speech..."):
                audio = generate_speech(text, processor, model, vocoder, speaker_embedding)
                
                if audio is not None:
                    # Save audio temporarily
                    output_file = "output.wav"
                    sf.write(output_file, audio, samplerate=16000)
                    
                    # Display audio player
                    st.audio(output_file, format="audio/wav")
                    
                    # Download button
                    with open(output_file, "rb") as file:
                        btn = st.download_button(
                            label="⬇️ Download Audio",
                            data=file,
                            file_name="generated_speech.wav",
                            mime="audio/wav"
                        )
    
    with tab2:
        st.header("Fine-tune Model")
        
        # File upload
        st.markdown("### Training Data")
        uploaded_file = st.file_uploader(
            "Upload training data (CSV)",
            type=["csv"],
            help="CSV file must contain 'text' and 'audio_path' columns"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if "text" in df.columns and "audio_path" in df.columns:
                    st.success("✅ Valid training data uploaded!")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Training parameters
                    st.markdown("### Training Parameters")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        epochs = st.number_input(
                            "Number of Epochs",
                            min_value=1,
                            max_value=50,
                            value=10
                        )
                        batch_size = st.selectbox(
                            "Batch Size",
                            options=[1, 2, 4, 8, 16],
                            index=2
                        )
                    
                    with col2:
                        learning_rate = st.selectbox(
                            "Learning Rate",
                            options=[0.00001, 0.0001, 0.001, 0.01],
                            index=1,
                            format_func=lambda x: f"{x:.5f}"
                        )
                    
                    # Training button
                    if st.button("🚀 Start Fine-tuning", type="primary"):
                        st.warning("⚠️ Fine-tuning can take several hours depending on your dataset size and parameters.")
                        st.error("Note: Fine-tuning functionality is currently in development.")
                        
                else:
                    st.error("❌ CSV file must contain 'text' and 'audio_path' columns!")
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)}")

if __name__ == "__main__":
    main()