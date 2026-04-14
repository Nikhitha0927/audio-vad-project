import librosa
import numpy as np

print("Program started")

audio, sr = librosa.load("input.wav", sr=16000)

print("Audio loaded successfully")
print("Sample rate:", sr)
print("Audio length:", len(audio))

frame_length = 2048
hop_length = 512

energy = []

for i in range(0, len(audio), hop_length):
    frame = audio[i:i+frame_length]
    energy.append(np.sum(frame**2))

threshold = np.mean(energy) * 0.5

segments = []
start = None

for i, e in enumerate(energy):
    time = i * hop_length / sr

    if e > threshold and start is None:
        start = time

    elif e <= threshold and start is not None:
        end = time
        segments.append((start, end))
        start = None

print("\nSpeech Segments:")
for s, e in segments:
    print(f"Start: {s:.2f}s, End: {e:.2f}s")