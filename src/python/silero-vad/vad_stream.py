import queue
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from matplotlib.animation import FuncAnimation
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

# Configuration
SAMPLE_RATE = 16000  # Hz
WINDOW_SIZE = 2  # seconds
BUFFER_SIZE = int(SAMPLE_RATE * WINDOW_SIZE)
CHANNELS = 1
BUFFER_BLOCK_SIZE = 512  # Number of samples per block
DEVICE = 0  # Input device (e.g., microphone)
THRESHOLD = 0.5  # VAD threshold
UPDATE_INTERVAL = 30  # Animation update interval in ms

# Initialize the VAD model
model = load_silero_vad()
print("VAD model loaded")

# Create a queue for audio data
audio_queue = queue.Queue()

# Global variables
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
is_speech = np.zeros(BUFFER_SIZE // BUFFER_BLOCK_SIZE, dtype=np.float32)
is_running = True

def audio_callback(indata, frames, time, status):
    """Callback function for the audio stream."""
    if status:
        print(f"Status: {status}")
    
    # Only use the first channel if stereo
    if CHANNELS > 1:
        data = indata[:, 0].copy()
    else:
        data = indata.copy().flatten()
    
    audio_queue.put(data)

def process_audio():
    """Process audio data from the queue and update the buffer."""
    global audio_buffer, is_speech
    
    while is_running:
        try:
            data = audio_queue.get(timeout=1.0)
            
            # Shift the buffer and add new data
            audio_buffer = np.roll(audio_buffer, -len(data))
            audio_buffer[-len(data):] = data
            
            # Convert to float tensor for Silero VAD
            tensor = torch.from_numpy(audio_buffer).float()
            
            # Get speech timestamps
            speech_timestamps = get_speech_timestamps(
                tensor, 
                model, 
                threshold=THRESHOLD,
                sampling_rate=SAMPLE_RATE
            )
            
            # Update speech detection array
            new_is_speech = np.zeros(BUFFER_SIZE // BUFFER_BLOCK_SIZE, dtype=np.float32)
            for timestamp in speech_timestamps:
                start_block = timestamp['start'] // BUFFER_BLOCK_SIZE
                end_block = min(timestamp['end'] // BUFFER_BLOCK_SIZE, len(new_is_speech))
                new_is_speech[start_block:end_block] = 1.0
            
            is_speech = new_is_speech
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in audio processing: {e}")
            break

# Set up the figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
fig.canvas.manager.set_window_title('Real-time Voice Activity Detection')

# Set up the lines
times = np.linspace(0, WINDOW_SIZE, BUFFER_SIZE)
waveform_line, = ax1.plot(times, np.zeros(BUFFER_SIZE), lw=1)
ax1.set_ylim(-1, 1)
ax1.set_xlim(0, WINDOW_SIZE)
ax1.set_title('Audio Waveform')
ax1.set_ylabel('Amplitude')
ax1.grid(True)

block_times = np.linspace(0, WINDOW_SIZE, BUFFER_SIZE // BUFFER_BLOCK_SIZE)
vad_line, = ax2.plot(block_times, np.zeros(BUFFER_SIZE // BUFFER_BLOCK_SIZE), lw=2, color='orange')
ax2.set_ylim(-0.1, 1.1)
ax2.set_xlim(0, WINDOW_SIZE)
ax2.set_title('Voice Activity Detection')
ax2.set_ylabel('Speech Detected')
ax2.set_xlabel('Time (s)')
ax2.grid(True)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['No', 'Yes'])

plt.tight_layout()

# Function to update the plots
def update_plot(frame):
    global audio_buffer, is_speech
    
    # Update waveform
    waveform_line.set_ydata(audio_buffer)
    
    # Update VAD
    vad_line.set_ydata(is_speech)
    
    return waveform_line, vad_line

# Start audio processing thread
audio_thread = threading.Thread(target=process_audio)
audio_thread.daemon = True
audio_thread.start()

# Start audio stream
stream = sd.InputStream(
    channels=CHANNELS,
    samplerate=SAMPLE_RATE,
    callback=audio_callback,
    blocksize=BUFFER_BLOCK_SIZE,
    device=DEVICE
)

print("Starting audio stream - press Ctrl+C to stop")

try:
    with stream:
        # Start animation
        ani = FuncAnimation(
            fig, 
            update_plot, 
            interval=UPDATE_INTERVAL, 
            blit=True
        )
        plt.show()
        
except KeyboardInterrupt:
    print("Stopping...")
except Exception as e:
    print(f"Error: {e}")
finally:
    is_running = False
    if audio_thread.is_alive():
        audio_thread.join(timeout=1.0)
    print("Stopped")