#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <cmath>
#include <fftw3.h>
#include <vector>

class WienerFilter {
private:
    int frameSize;
    double* inFrame;
    double* outFrame;
    fftw_complex* fftIn;
    fftw_complex* fftOut;
    fftw_plan forwardPlan;
    fftw_plan inversePlan;
    
    // Noise estimation
    std::vector<double> noiseSpectrum;
    std::vector<double> signalSpectrum;
    int noiseFramesCollected;
    int minNoiseFrames;
    bool noiseProfileReady;
    
    // Filter parameters
    double alpha;      // Smoothing factor for noise spectrum update
    double beta;       // Smoothing factor for signal spectrum
    double minGain;    // Minimum gain to apply (avoid musical noise)
    
public:
    WienerFilter(int frameSize) : 
        frameSize(frameSize),
        noiseFramesCollected(0),
        minNoiseFrames(20),
        noiseProfileReady(false),
        alpha(0.98),
        beta(0.98),
        minGain(0.1)
    {
        inFrame = (double*)fftw_malloc(sizeof(double) * frameSize);
        outFrame = (double*)fftw_malloc(sizeof(double) * frameSize);
        fftIn = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (frameSize/2 + 1));
        fftOut = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (frameSize/2 + 1));
        
        // Initialize FFT plans
        forwardPlan = fftw_plan_dft_r2c_1d(frameSize, inFrame, fftIn, FFTW_ESTIMATE);
        inversePlan = fftw_plan_dft_c2r_1d(frameSize, fftOut, outFrame, FFTW_ESTIMATE);
        
        // Initialize spectrum vectors
        noiseSpectrum.resize(frameSize/2 + 1, 0.0);
        signalSpectrum.resize(frameSize/2 + 1, 0.0);
    }
    
    ~WienerFilter() {
        fftw_destroy_plan(forwardPlan);
        fftw_destroy_plan(inversePlan);
        fftw_free(inFrame);
        fftw_free(outFrame);
        fftw_free(fftIn);
        fftw_free(fftOut);
    }
    
    // Process a frame of audio with the Wiener filter
    void processFrame(const float* input, float* output) {
        // Apply window function (Hann window) and copy input
        for (int i = 0; i < frameSize; i++) {
            double window = 0.5 * (1 - cos(2 * M_PI * i / (frameSize - 1)));
            inFrame[i] = input[i] * window;
        }
        
        // Perform forward FFT
        fftw_execute(forwardPlan);
        
        // Compute magnitude spectrum
        std::vector<double> magnitudeSpectrum(frameSize/2 + 1);
        for (int i = 0; i <= frameSize/2; i++) {
            double real = fftIn[i][0];
            double imag = fftIn[i][1];
            magnitudeSpectrum[i] = real*real + imag*imag;
        }
        
        // If we're still collecting noise profile
        if (!noiseProfileReady) {
            collectNoiseProfile(magnitudeSpectrum);
        } else {
            // Update signal spectrum estimation with smoothing
            for (int i = 0; i <= frameSize/2; i++) {
                signalSpectrum[i] = beta * signalSpectrum[i] + (1-beta) * magnitudeSpectrum[i];
            }
            
            // Apply Wiener filter
            applyWienerFilter(magnitudeSpectrum);
        }
        
        // Inverse FFT
        fftw_execute(inversePlan);
        
        // Scale output (FFTW doesn't normalize)
        for (int i = 0; i < frameSize; i++) {
            output[i] = static_cast<float>(outFrame[i] / frameSize);
        }
    }
    
    // Collect noise profile from (assumed) silent portions
    void collectNoiseProfile(const std::vector<double>& magnitudeSpectrum) {
        if (noiseFramesCollected < minNoiseFrames) {
            // Accumulate noise spectrum
            for (int i = 0; i <= frameSize/2; i++) {
                noiseSpectrum[i] += magnitudeSpectrum[i] / minNoiseFrames;
            }
            noiseFramesCollected++;
            
            // Just copy input to output during calibration
            for (int i = 0; i <= frameSize/2; i++) {
                fftOut[i][0] = fftIn[i][0];
                fftOut[i][1] = fftIn[i][1];
            }
            
            if (noiseFramesCollected >= minNoiseFrames) {
                noiseProfileReady = true;
                printf("Noise profile collected successfully.\n");
            }
        }
    }
    
    // Apply the Wiener filter to the spectrum
    void applyWienerFilter(const std::vector<double>& magnitudeSpectrum) {
        for (int i = 0; i <= frameSize/2; i++) {
            // Update noise profile (slow adaptation)
            if (magnitudeSpectrum[i] < noiseSpectrum[i]) {
                noiseSpectrum[i] = alpha * noiseSpectrum[i] + (1-alpha) * magnitudeSpectrum[i];
            }
            
            // Calculate a priori SNR (signal-to-noise ratio)
            double snr = std::max(signalSpectrum[i] / noiseSpectrum[i] - 1.0, 0.0);
            
            // Wiener filter gain
            double gain = snr / (snr + 1.0);
            
            // Apply minimum gain to avoid musical noise
            gain = std::max(gain, minGain);
            
            // Apply gain to the complex spectrum
            fftOut[i][0] = fftIn[i][0] * gain;
            fftOut[i][1] = fftIn[i][1] * gain;
        }
    }
    
    // Force start of noise collection
    void startNoiseCollection() {
        noiseFramesCollected = 0;
        noiseProfileReady = false;
        std::fill(noiseSpectrum.begin(), noiseSpectrum.end(), 0.0);
    }
    
    // Check if noise profile is ready
    bool isNoiseProfileReady() const {
        return noiseProfileReady;
    }
    
    // Set filter parameters
    void setParameters(double noiseSmoothing, double signalSmoothing, double minimumGain) {
        alpha = noiseSmoothing;
        beta = signalSmoothing;
        minGain = minimumGain;
    }
};