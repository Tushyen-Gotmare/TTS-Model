# Text-to-Speech Fine-tuning Application

A Streamlit-based web application for generating and fine-tuning text-to-speech models using the SpeechT5 architecture.

![image](https://github.com/user-attachments/assets/5baed9e2-b030-4f10-87c9-be74f8290f88)

![image](https://github.com/user-attachments/assets/3802606a-5681-4126-a454-19fd9935b321)



## Features

- 🎤 Text-to-speech generation
- ⚡ Real-time speech synthesis
- 🎛️ Adjustable speech rate
- 📊 Model fine-tuning capabilities
- 💾 Audio download options
- 📈 Training progress visualization

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)
- 8GB RAM minimum (16GB recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd tts-fine-tuning-app
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/MacOS
python -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

### Requirements.txt
```
streamlit>=1.28.0
torch>=2.0.0
soundfile>=0.12.1
transformers>=4.30.0
numpy>=1.24.0
pandas>=2.0.0
librosa>=0.10.0
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. To generate speech:
   - Navigate to the "Generate Speech" tab
   - Enter your text in the input field
   - Adjust the speech rate if desired
   - Click "Generate Speech"
   - Use the download button to save the generated audio

4. To fine-tune the model:
   - Navigate to the "Fine-tune Model" tab
   - Upload a CSV file containing training data
   - Set training parameters
   - Start the fine-tuning process

## Training Data Format

The CSV file for fine-tuning should contain the following columns:
- `text`: The text content for training
- `audio_path`: Path to corresponding audio files

Example:
```csv
text,audio_path
"Hello world",/path/to/audio1.wav
"Text to speech",/path/to/audio2.wav
```

## Model Information

This application uses the following models from the Hugging Face model hub:
- `microsoft/speecht5_tts`: Main TTS model
- `microsoft/speecht5_hifigan`: Vocoder model

## Known Limitations

- Fine-tuning functionality is currently in development
- Maximum text length is limited to 1000 characters
- Processing time may vary based on hardware capabilities
- Requires significant RAM for model loading

## Troubleshooting

1. If you encounter CUDA out-of-memory errors:
   - Reduce batch size
   - Free up GPU memory
   - Consider using CPU-only mode

2. If models fail to load:
   - Check internet connection
   - Verify CUDA installation
   - Ensure sufficient disk space

3. For audio generation issues:
   - Verify input text is not empty
   - Check system audio settings
   - Ensure required models are properly downloaded

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

