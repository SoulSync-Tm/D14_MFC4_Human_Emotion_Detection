import librosa
import torch
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

class AudioEmotionAnalyzer:
    def __init__(self):
        print("Loading Audio Emotion Model (Wav2Vec2)...")
        self.model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name)
        self.labels = {0: 'angry', 1: 'calm', 2: 'disgust', 3: 'fearful', 4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprised'}

    def analyze_audio(self, audio_path):
        """
        Input: Path to an audio file.
        Output: The dominant emotion and confidence score.
        """
        try:
            # Load audio (resample to 16kHz for Wav2Vec2)
            speech, sr = librosa.load(audio_path, sr=16000)
            
            # Process chunks if audio is long (simplified: just take first 10s for now)
            max_duration = 10 # seconds
            if len(speech) > max_duration * sr:
                speech = speech[:max_duration * sr]

            # Extract features
            inputs = self.feature_extractor(speech, sampling_rate=sr, return_tensors="pt", padding=True)

            # Predict
            with torch.no_grad():
                logits = self.model(**inputs).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            predicted_label = self.labels[predicted_ids.item()]
            confidence = torch.softmax(logits, dim=-1).max().item()

            return {
                "audio_emotion": predicted_label,
                "audio_confidence": round(confidence, 2)
            }
        except Exception as e:
            print(f"Audio Error: {e}")
            return {"audio_emotion": "Error", "audio_confidence": 0.0}