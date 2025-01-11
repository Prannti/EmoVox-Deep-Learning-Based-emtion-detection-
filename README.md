# EmoVox: Deep Learning-Based Emotion Detection

EmoVox is a deep learning-based framework for emotion recognition from audio. It leverages advanced audio processing techniques and neural networks like **Wav2Vec2** for extracting meaningful features, which are then fine-tuned for emotion classification. The project supports data augmentation, GPU acceleration, and multi-fold cross-validation for robust and reliable results.

---

## Features

1. **Speech Emotion Recognition**  
   Classifies a variety of emotions (e.g., happiness, sadness, anger, neutrality, etc.) from audio data.

2. **Wav2Vec2 Feature Extraction**  
   Utilizes `facebook/wav2vec2-base` for robust, pre-trained feature representations of audio waveforms.

3. **Data Augmentation**  
   - **Time Stretching**  
     Extends the audio duration without changing the pitch.
   - **Pitch Shifting**  
     Changes the pitch by semitones.
   - **Noise Addition**  
     Adds Gaussian noise to waveforms to improve robustness against real-world disturbances.

4. **Cross-Validation**  
   Implements `StratifiedKFold` for consistent and reliable performance measurement across different data splits.

5. **GPU Support**  
   Detects and utilizes CUDA for training acceleration if available.

---

## Requirements

- Python 3.8 or above
- Libraries:
  - `torch`
  - `torchaudio`
  - `transformers`
  - `scikit-learn`
  - `tqdm`
  - `matplotlib`
  - `Flask` (for deployment via a Flask API)
  - `PyQt5` (for the optional GUI)

Install all dependencies using:
```bash
pip install torch torchaudio transformers scikit-learn tqdm matplotlib Flask PyQt5


