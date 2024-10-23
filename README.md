
# Fine-Tuning TTS Model Using SpeechT5

This project focuses on fine-tuning a Text-to-Speech (TTS) model using Microsoft's SpeechT5. It generates synthetic speech from text input and supports further customization by integrating speaker embeddings. The generated audio is saved as a waveform file (`output.wav`).

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Input Parameters](#input-parameters)
- [Known Issues](#known-issues)
- [Contributing](#contributing)

## Overview

The project leverages the SpeechT5 model from Microsoft's Hugging Face library to fine-tune text-to-speech capabilities. The main focus is on generating high-quality audio with technical vocabulary. The model also supports speaker embeddings, which allow you to use specific voice characteristics.

## Features
- **Text-to-Speech Generation:** Converts technical text input into spoken words.
- **Speaker Embeddings:** Allows custom voice characteristics for generated speech.
- **Audio Export:** Saves generated speech as a `.wav` file for playback.

## Installation

Follow these steps to set up the environment and dependencies:

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/TTSProject.git
   cd TTSProject
   
## Required Dependencies
- transformers
- torch
- torchaudio
- soundfile
- numpy
- datasets

## Usage
To run the fine-tuning process and generate the audio, execute the fine_tune_tts.py script:
```
python fine_tune_tts.py
```
## Input Parameters
- Text: Modify the text variable in fine_tune_tts.py to change the input text.
- Speaker Embeddings: Customize speaker embeddings by modifying the speaker_embeddings loaded from the dataset.

## Known Issues
- Blank Audio: If the audio file is blank or of poor quality, you may need to adjust the spectrogram's n_fft parameter or fine-tune the model further.
- Unsupported File Format: Ensure that the audio file is in a supported format before attempting playback.
## Contributing
I welcome contributions to enhance this project. Feel free to open issues or submit pull requests


