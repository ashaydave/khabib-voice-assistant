#include <stdlib.h>
#include <stdio.h>
#include <portaudio.h>
#include <cstring>
#include <fftw3.h>
#include <cmath>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <sndfile.h>
#include "noise_suppression.cpp" // Include our Wiener filter implementation
#include "beamforming.cpp"       // Include our beamforming implementation

#define SAMPLE_RATE 44100.0
#define FRAMES_PER_BUFFER 512
#define NUM_CHANNELS 2
#define SPECTRO_FREQ_START 20
#define SPECTRO_FREQ_END 20000
#define MIC_SPACING 0.1 // Microphone spacing in meters (adjust based on your setup)

enum ProcessingStage {
    RAW_AUDIO,
    WIENER_FILTERED,
    BEAMFORMED
};

typedef struct {
    WienerFilter* wienerFilter;
    DelayAndSumBeamformer* beamformer;
    double* in;
    double* out;
    fftw_plan p;
    int startIndex;
    int spectroSize;
    float* rawBuffer;          // Raw audio buffer (copy of input)
    float* processedBuffer;    // Buffer after noise reduction
    float* beamformedBuffer;   // Buffer after beamforming
    ProcessingStage currentStage; // Track which processing stage to visualize
    int frameCount;            // Count frames to toggle visualization
} streamCallbackData;

static streamCallbackData* spectroData;

/* ----------------- File Writing Globals ----------------- */
// Structure to hold one audio frame of data for each processing stage.
struct AudioFrame {
    float* raw;
    float* filtered;
    float* beamformed;
    unsigned long frames; // Number of frames stored
};

// Thread‑safe queue and synchronization variables.
std::queue<AudioFrame> audioQueue;
std::mutex queueMutex;
std::condition_variable queueCV;
bool recording = true;

/* ----------------- File Writing Thread ----------------- */
void fileWriterThread() {
    // Setup file info for raw and filtered (stereo) output.
    SF_INFO sfinfo;
    sfinfo.samplerate = SAMPLE_RATE;
    sfinfo.channels = NUM_CHANNELS;
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    SNDFILE* rawFile = sf_open("raw.wav", SFM_WRITE, &sfinfo);
    SNDFILE* filteredFile = sf_open("filtered.wav", SFM_WRITE, &sfinfo);

    // Setup file info for beamformed (mono) output.
    SF_INFO sfinfoMono;
    sfinfoMono.samplerate = SAMPLE_RATE;
    sfinfoMono.channels = 1;
    sfinfoMono.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
    SNDFILE* beamformedFile = sf_open("beamformed.wav", SFM_WRITE, &sfinfoMono);

    if (!rawFile || !filteredFile || !beamformedFile) {
        printf("Error opening output files.\n");
        return;
    }

    while (recording || !audioQueue.empty()) {
        AudioFrame frame;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCV.wait(lock, [] { return !audioQueue.empty() || !recording; });
            if (!audioQueue.empty()) {
                frame = audioQueue.front();
                audioQueue.pop();
            } else {
                continue;
            }
        }
        // Write the data (raw and filtered are stereo; beamformed is mono)
        sf_writef_float(rawFile, frame.raw, frame.frames);
        sf_writef_float(filteredFile, frame.filtered, frame.frames);
        sf_writef_float(beamformedFile, frame.beamformed, frame.frames);

        // Free the allocated buffers for this frame.
        delete[] frame.raw;
        delete[] frame.filtered;
        delete[] frame.beamformed;
    }
    sf_close(rawFile);
    sf_close(filteredFile);
    sf_close(beamformedFile);
}

/* ----------------- Utility Functions ----------------- */
static void checkErr(PaError err) {
    if (err != paNoError) {
        printf("PortAudio Error: %s\n", Pa_GetErrorText(err));
        exit(EXIT_FAILURE);
    }
}

static inline float maxf(float a, float b) {
    return a > b ? a : b;
}

static inline float minf(float a, float b) {
    return a < b ? a : b;
}

void printSignalInfo(const char* label, float* buffer, int length) {
    float sum = 0.0f;
    float max = 0.0f;
    
    for (int i = 0; i < length; i++) {
        sum += fabs(buffer[i]);
        if (fabs(buffer[i]) > max) max = fabs(buffer[i]);
    }
    
    float avg = sum / length;
    printf("%s - Avg: %.6f, Max: %.6f\n", label, avg, max);
}

/* ----------------- Audio Callback ----------------- */
static int streamCallback(const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void* userData) {
    
    float* in = (float*)inputBuffer;
    streamCallbackData* callbackData = (streamCallbackData*)userData;

    // Store the raw audio for visualization.
    memcpy(callbackData->rawBuffer, in, sizeof(float) * framesPerBuffer * NUM_CHANNELS);
    
    // Process with the Wiener filter (assumes interleaved channels).
    callbackData->wienerFilter->processFrame(in, callbackData->processedBuffer);
    
    // Process with the beamformer.
    callbackData->beamformer->processFrame(callbackData->processedBuffer, callbackData->beamformedBuffer);

    // Diagnostic printing every 100 frames.
    callbackData->frameCount++;
    if (callbackData->frameCount % 100 == 0) {
        printf("\n--- DIAGNOSTICS (Frame %d) ---\n", callbackData->frameCount);
        printSignalInfo("RAW SIGNAL", callbackData->rawBuffer, framesPerBuffer * NUM_CHANNELS);
        printSignalInfo("AFTER WIENER FILTER", callbackData->processedBuffer, framesPerBuffer * NUM_CHANNELS);
        printSignalInfo("AFTER BEAMFORMING", callbackData->beamformedBuffer, framesPerBuffer);
        
        if (callbackData->frameCount % 500 == 0) {
            callbackData->currentStage = (ProcessingStage)((callbackData->currentStage + 1) % 3);
            const char* stageNames[] = {"RAW AUDIO", "WIENER FILTERED", "BEAMFORMED"};
            printf("\n>> SWITCHING VISUALIZATION TO: %s <<\n\n", stageNames[callbackData->currentStage]);
        }
    }

    // Select the signal to visualize.
    switch (callbackData->currentStage) {
        case RAW_AUDIO:
            for (unsigned long i = 0; i < framesPerBuffer; i++) {
                callbackData->in[i] = callbackData->rawBuffer[i * NUM_CHANNELS];
            }
            break;
        case WIENER_FILTERED:
            for (unsigned long i = 0; i < framesPerBuffer; i++) {
                callbackData->in[i] = callbackData->processedBuffer[i * NUM_CHANNELS];
            }
            break;
        case BEAMFORMED:
            for (unsigned long i = 0; i < framesPerBuffer; i++) {
                callbackData->in[i] = callbackData->beamformedBuffer[i];
            }
            break;
    }

    // Execute FFT on the selected signal.
    fftw_execute(callbackData->p);

    // Display a simple text-based spectrum.
    int dispSize = 100;
    printf("\r");
    printf("[%s] ", 
        callbackData->currentStage == RAW_AUDIO ? "RAW" : 
        (callbackData->currentStage == WIENER_FILTERED ? "FILTERED" : "BEAMFORMED"));
    for (int i = 0; i < dispSize; i++) {
        double proportion = std::pow(i / (double)dispSize, 2);
        double freq = callbackData->out[(int)(callbackData->startIndex + proportion * callbackData->spectroSize)];
        if (freq < 0.125) {
            printf("▁");
        } else if (freq < 0.25) {
            printf("▂");
        } else if (freq < 0.375) {
            printf("▃");
        } else if (freq < 0.5) {
            printf("▄");
        } else if (freq < 0.625) {
            printf("▅");
        } else if (freq < 0.75) {
            printf("▆");
        } else if (freq < 0.875) {
            printf("▇");
        } else {
            printf("█");
        }
    }
    fflush(stdout);

    // ----- Push the processed frame into the recording queue -----
    {
        AudioFrame frame;
        frame.frames = framesPerBuffer;
        frame.raw = new float[framesPerBuffer * NUM_CHANNELS];
        frame.filtered = new float[framesPerBuffer * NUM_CHANNELS];
        frame.beamformed = new float[framesPerBuffer];
        memcpy(frame.raw, callbackData->rawBuffer, sizeof(float) * framesPerBuffer * NUM_CHANNELS);
        memcpy(frame.filtered, callbackData->processedBuffer, sizeof(float) * framesPerBuffer * NUM_CHANNELS);
        memcpy(frame.beamformed, callbackData->beamformedBuffer, sizeof(float) * framesPerBuffer);
        
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            audioQueue.push(frame);
        }
        queueCV.notify_one();
    }

    return 0;
}

/* ----------------- Main Function ----------------- */
int main() {
    PaError err;
    err = Pa_Initialize();
    checkErr(err);

    // Allocate and initialize our callback data.
    spectroData = (streamCallbackData*)malloc(sizeof(streamCallbackData));
    spectroData->in = (double*)malloc(sizeof(double) * FRAMES_PER_BUFFER);
    spectroData->out = (double*)malloc(sizeof(double) * FRAMES_PER_BUFFER);
    spectroData->rawBuffer = (float*)malloc(sizeof(float) * FRAMES_PER_BUFFER * NUM_CHANNELS);
    spectroData->processedBuffer = (float*)malloc(sizeof(float) * FRAMES_PER_BUFFER * NUM_CHANNELS);
    spectroData->beamformedBuffer = (float*)malloc(sizeof(float) * FRAMES_PER_BUFFER);
    spectroData->currentStage = BEAMFORMED; // Start with beamformed output
    spectroData->frameCount = 0;
    
    if (!spectroData->in || !spectroData->out || 
        !spectroData->rawBuffer || !spectroData->processedBuffer || 
        !spectroData->beamformedBuffer) {
        printf("Could not allocate memory\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize the Wiener filter.
    spectroData->wienerFilter = new WienerFilter(FRAMES_PER_BUFFER * NUM_CHANNELS);
    
    // Initialize the beamformer.
    spectroData->beamformer = new DelayAndSumBeamformer(
        FRAMES_PER_BUFFER, NUM_CHANNELS, SAMPLE_RATE, MIC_SPACING, 343.0, 0.0);
    
    spectroData->p = fftw_plan_r2r_1d(FRAMES_PER_BUFFER, spectroData->in, 
        spectroData->out, FFTW_R2HC, FFTW_ESTIMATE);

    double sampleRatio = FRAMES_PER_BUFFER / SAMPLE_RATE;
    spectroData->startIndex = std::ceil(sampleRatio * SPECTRO_FREQ_START);
    spectroData->spectroSize = minf(std::ceil(sampleRatio * SPECTRO_FREQ_END),
        FRAMES_PER_BUFFER / 2.0) - spectroData->startIndex;

    // Find the default input device.
    PaDeviceIndex defaultInputDevice = Pa_GetDefaultInputDevice();
    if (defaultInputDevice == paNoDevice) {
        printf("No default input device found.\n");
        exit(EXIT_FAILURE);
    }
    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(defaultInputDevice);
    printf("Using default input device: %s\n", deviceInfo->name);
    
    if (deviceInfo->maxInputChannels < NUM_CHANNELS) {
        printf("Warning: Default device only supports %d channels, but %d are needed.\n", 
               deviceInfo->maxInputChannels, NUM_CHANNELS);
        printf("Continuing with available channels, but beamforming may not work correctly.\n");
    }

    PaStreamParameters inputParameters;
    memset(&inputParameters, 0, sizeof(inputParameters));
    inputParameters.channelCount = NUM_CHANNELS;
    inputParameters.device = defaultInputDevice;
    inputParameters.hostApiSpecificStreamInfo = NULL;
    inputParameters.sampleFormat = paFloat32;
    inputParameters.suggestedLatency = deviceInfo->defaultLowInputLatency;

    printf("Starting noise profile collection. Please remain silent for calibration...\n");
    spectroData->wienerFilter->startNoiseCollection();

    // Launch the file writer thread.
    std::thread writer(fileWriterThread);

    PaStream* stream;
    err = Pa_OpenStream(
        &stream,
        &inputParameters,
        NULL,
        SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        paNoFlag,
        streamCallback,
        spectroData
    );
    checkErr(err);

    err = Pa_StartStream(stream);
    checkErr(err);

    printf("\nSystem is running. Available commands:\n");
    printf("  'r' - View raw audio\n");
    printf("  'w' - View Wiener filtered audio\n");
    printf("  'b' - View beamformed audio\n");
    printf("  'q' - Quit\n");
    
    // Process user commands.
    bool running = true;
    while (running) {
        Pa_Sleep(100);
        int c = getchar();
        if (c != EOF) {
            switch (c) {
                case 'r':
                case 'R':
                    spectroData->currentStage = RAW_AUDIO;
                    printf("\nSwitched to RAW AUDIO visualization\n");
                    break;
                case 'w':
                case 'W':
                    spectroData->currentStage = WIENER_FILTERED;
                    printf("\nSwitched to WIENER FILTERED visualization\n");
                    break;
                case 'b':
                case 'B':
                    spectroData->currentStage = BEAMFORMED;
                    printf("\nSwitched to BEAMFORMED visualization\n");
                    break;
                case 'q':
                case 'Q':
                    running = false;
                    break;
            }
        }
    }

    err = Pa_StopStream(stream);
    checkErr(err);
    err = Pa_CloseStream(stream);
    checkErr(err);
    err = Pa_Terminate();
    checkErr(err);

    // Signal the file writer thread to finish.
    recording = false;
    queueCV.notify_one();
    writer.join();

    // Clean up FFTW and allocated memory.
    fftw_destroy_plan(spectroData->p);
    fftw_free(spectroData->in);
    fftw_free(spectroData->out);
    delete spectroData->wienerFilter;
    delete spectroData->beamformer;
    free(spectroData->rawBuffer);
    free(spectroData->processedBuffer);
    free(spectroData->beamformedBuffer);
    free(spectroData);

    printf("\nRecording complete. Exiting.\n");
    return EXIT_SUCCESS;
}
