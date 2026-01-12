import sounddevice as sd
import numpy as np
import librosa
import joblib
from scipy.io.wavfile import write

MODEL_PATH = "model/emotion_model.pkl"
EMOTIONS = ["happy", "sad", "angry", "neutral"]
DURATION = 3
SAMPLE_RATE = 22050
TEMP_FILE = "temp.wav"

print(" Real-Time Speech Emotion Recognition Started")

model = joblib.load(MODEL_PATH)
print("Model loaded")

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
    return np.mean(combined.T, axis=0)

def record_audio():
    print(" Speak now...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    write(TEMP_FILE, SAMPLE_RATE, audio)
    print(" Recording finished")

def predict_emotion():
    features = extract_mfcc(TEMP_FILE)
    prediction = model.predict([features])[0]
    print("Detected Emotion:", EMOTIONS[prediction].upper())

while True:
    input("\nPress ENTER to record (Ctrl+C to exit)...")
    record_audio()
    predict_emotion()
