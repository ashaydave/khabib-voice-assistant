#ifndef BEAMFORMING_H
#define BEAMFORMING_H

#include <vector>
#include <cmath>
#include <complex>
#include <fftw3.h>

class DelayAndSumBeamformer {
private:
    int frameSize;
    int numChannels;
    double sampleRate;
    double micSpacing;  // Distance between microphones in meters
    double soundSpeed;  // Speed of sound in air (m/s)
    double targetAngle; // Target angle in radians (0 = front)
    
    // FFT related variables
    std::vector<double*> inFrames;
    std::vector<fftw_complex*> fftChannels;
    double* outFrame;
    fftw_complex* fftOut;
    std::vector<fftw_plan> forwardPlans;
    fftw_plan inversePlan;
    
    // Time delays for each channel (in samples)
    std::vector<double> timeDelays;
    
    // Phase shifts for frequency domain beamforming
    std::vector<std::vector<std::complex<double>>> phaseShifts;
    
public:
    DelayAndSumBeamformer(int frameSize, int numChannels, double sampleRate, 
                         double micSpacing = 0.1, double soundSpeed = 343.0, 
                         double targetAngle = 0.0) : 
        frameSize(frameSize),
        numChannels(numChannels),
        sampleRate(sampleRate),
        micSpacing(micSpacing),
        soundSpeed(soundSpeed),
        targetAngle(targetAngle)
    {
        // Allocate memory for each channel
        for (int ch = 0; ch < numChannels; ch++) {
            double* inFrame = (double*)fftw_malloc(sizeof(double) * frameSize);
            fftw_complex* fftChannel = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (frameSize/2 + 1));
            
            inFrames.push_back(inFrame);
            fftChannels.push_back(fftChannel);
            
            // Create forward FFT plan for each channel
            fftw_plan forwardPlan = fftw_plan_dft_r2c_1d(frameSize, inFrame, fftChannel, FFTW_ESTIMATE);
            forwardPlans.push_back(forwardPlan);
        }
        
        // Allocate memory for output
        outFrame = (double*)fftw_malloc(sizeof(double) * frameSize);
        fftOut = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (frameSize/2 + 1));
        
        // Create inverse FFT plan
        inversePlan = fftw_plan_dft_c2r_1d(frameSize, fftOut, outFrame, FFTW_ESTIMATE);
        
        // Calculate time delays for each channel
        calculateTimeDelays();
        
        // Precalculate phase shifts for frequency domain beamforming
        calculatePhaseShifts();
    }
    
    ~DelayAndSumBeamformer() {
        // Clean up all FFT resources
        for (int ch = 0; ch < numChannels; ch++) {
            fftw_destroy_plan(forwardPlans[ch]);
            fftw_free(inFrames[ch]);
            fftw_free(fftChannels[ch]);
        }
        
        fftw_destroy_plan(inversePlan);
        fftw_free(outFrame);
        fftw_free(fftOut);
    }
    
    // Calculate time delays for each microphone based on target angle
    void calculateTimeDelays() {
        timeDelays.resize(numChannels);
        
        // Assume center of array is the reference point (zero delay)
        double arrayCenter = (numChannels - 1) * micSpacing / 2.0;
        
        for (int ch = 0; ch < numChannels; ch++) {
            // Position of this microphone relative to center
            double micPos = ch * micSpacing - arrayCenter;
            
            // Calculate path difference based on target angle
            double pathDiff = micPos * sin(targetAngle);
            
            // Convert to time delay in samples
            timeDelays[ch] = pathDiff / soundSpeed * sampleRate;
        }
    }
    
    // Precalculate phase shifts for all frequencies and channels
    void calculatePhaseShifts() {
        phaseShifts.resize(numChannels);
        
        for (int ch = 0; ch < numChannels; ch++) {
            phaseShifts[ch].resize(frameSize/2 + 1);
            
            for (int k = 0; k <= frameSize/2; k++) {
                // Frequency in Hz
                double freq = k * sampleRate / frameSize;
                
                // Phase shift = -2π * frequency * time_delay
                double phaseShift = -2.0 * M_PI * freq * timeDelays[ch] / sampleRate;
                
                // Store as complex exponential
                phaseShifts[ch][k] = std::polar(1.0, phaseShift);
            }
        }
    }
    
    // Process a multichannel frame to produce a single-channel enhanced output
    void processFrame(const float* input, float* output) {
        // Apply window function and copy input for each channel
        for (int ch = 0; ch < numChannels; ch++) {
            for (int i = 0; i < frameSize; i++) {
                // Apply Hann window
                double window = 0.5 * (1 - cos(2 * M_PI * i / (frameSize - 1)));
                
                // Extract channel data from interleaved input
                inFrames[ch][i] = input[i * numChannels + ch] * window;
            }
            
            // Perform forward FFT for this channel
            fftw_execute(forwardPlans[ch]);
        }
        
        // Perform delay-and-sum beamforming in the frequency domain
        for (int k = 0; k <= frameSize/2; k++) {
            std::complex<double> sum(0.0, 0.0);
            
            // Sum the phase-shifted signals from all channels
            for (int ch = 0; ch < numChannels; ch++) {
                std::complex<double> channelValue(fftChannels[ch][k][0], fftChannels[ch][k][1]);
                sum += channelValue * phaseShifts[ch][k];
            }
            
            // Normalize by the number of channels
            sum /= numChannels;
            
            // Store the result in the output FFT buffer
            fftOut[k][0] = sum.real();
            fftOut[k][1] = sum.imag();
        }
        
        // Perform inverse FFT to get time-domain signal
        fftw_execute(inversePlan);
        
        // Scale output (FFTW doesn't normalize) and copy to output buffer
        for (int i = 0; i < frameSize; i++) {
            output[i] = static_cast<float>(outFrame[i] / frameSize);
        }
    }
    
    // Set a new target angle (in radians)
    void setTargetAngle(double angle) {
        targetAngle = angle;
        calculateTimeDelays();
        calculatePhaseShifts();
    }
    
    // Set microphone spacing
    void setMicSpacing(double spacing) {
        micSpacing = spacing;
        calculateTimeDelays();
        calculatePhaseShifts();
    }
};

#endif // BEAMFORMING_H