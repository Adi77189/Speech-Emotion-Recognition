"""
MFCC stands for Mel Frequency Cepstral Coefficients.
MFCCs are numerical features extracted from an audio signal.

These numerical values represent important characteristics of human speech such as tone, pitch, and timbre.

Machine learning models do not learn directly from raw audio files. Instead, they learn patterns from MFCC features to understand and classify emotions present in speech.

"""
import os
import librosa
import numpy as np

DATA_PATH = "data/audio"
EMOTIONS = ["happy", "sad", "angry", "neutral"]

X = []  # features
y = []  # labels

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
    return np.mean(combined.T, axis=0)  # 120 features

for label, emotion in enumerate(EMOTIONS):
    emotion_folder = os.path.join(DATA_PATH, emotion)
    for file in os.listdir(emotion_folder):
        if file.endswith(".wav"):
            file_path = os.path.join(emotion_folder, file)
            try:
                features = extract_mfcc(file_path)
                X.append(features)
                y.append(label)
            except Exception as e:
                print("Error processing:", file_path)

X = np.array(X)
y = np.array(y)

os.makedirs("features", exist_ok=True)
np.save("features/X.npy", X)
np.save("features/y.npy", y)

print("MFCC Feature Extraction Completed")
print("Total samples:", X.shape[0])
print("Feature vector size:", X.shape[1])
