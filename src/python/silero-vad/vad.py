import numpy as np
import matplotlib.pyplot as plt
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import librosa
import librosa.display

# Load the VAD model
model = load_silero_vad()

# Read the audio file
wav_path = './en_example.wav'
wav = read_audio(wav_path)

# Get speech timestamps
speech_timestamps = get_speech_timestamps(
    wav,
    model,
    return_seconds=True,  # Return speech timestamps in seconds
)
print(speech_timestamps)

# Load the audio file for visualization
y, sr = librosa.load(wav_path, sr=None)
duration = librosa.get_duration(y=y, sr=sr)

# Create a visualization
plt.figure(figsize=(15, 6))

# Plot waveform
plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr, alpha=0.6)
plt.title('Audio Waveform with Speech Detection')
plt.ylabel('Amplitude')

# Highlight the speech segments
for segment in speech_timestamps:
    start = segment['start']
    end = segment['end']
    plt.axvspan(start, end, color='green', alpha=0.3)
    plt.text(start, 0.8, f"{start:.2f}s", fontsize=8, ha='center')
    plt.text(end, 0.8, f"{end:.2f}s", fontsize=8, ha='center')

# Plot speech activity timeline
plt.subplot(2, 1, 2)
activity = np.zeros(int(duration * 100))  # 100 points per second for visualization
for segment in speech_timestamps:
    start_idx = int(segment['start'] * 100)
    end_idx = int(segment['end'] * 100)
    activity[start_idx:end_idx] = 1

timeline = np.linspace(0, duration, len(activity))
plt.plot(timeline, activity, linewidth=2)
plt.fill_between(timeline, activity, alpha=0.5, color='orange')
plt.yticks([0, 1], ['Silent', 'Speech'])
plt.xlabel('Time (seconds)')
plt.title('Speech Activity Timeline')

plt.tight_layout()
plt.show()

# Print statistical information
total_speech_duration = sum(segment['end'] - segment['start'] for segment in speech_timestamps)
percentage_speech = (total_speech_duration / duration) * 100

print(f"Total audio duration: {duration:.2f} seconds")
print(f"Total speech duration: {total_speech_duration:.2f} seconds")
print(f"Percentage of speech: {percentage_speech:.2f}%")
print(f"Number of speech segments: {len(speech_timestamps)}")