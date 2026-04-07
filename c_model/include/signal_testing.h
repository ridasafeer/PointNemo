#include <vector>
#include <string>
#include <complex>

class SignalTesting {
public:
    struct SpectrumResult {
        std::vector<float> freqs_hz;
        std::vector<float> magnitude;
        std::vector<float> power_db;
        std::vector<float> phase_rad;
    };

    struct TransferResult {
        std::vector<float> freqs_hz;
        std::vector<float> magnitude;
        std::vector<float> magnitude_db;
        std::vector<float> phase_rad;
    };

    struct CoherenceResult {
        std::vector<float> freqs_hz;
        std::vector<float> coherence;
    };

    explicit SignalTesting(float sampleRate);

    // some basic stfff
    float computeMean(const std::vector<float>& x) const;
    float computeRMS(const std::vector<float>& x) const;
    float computePeak(const std::vector<float>& x) const;

    // spectral analysis: frequency domain analysis of the output signal
    SpectrumResult computeSpectrum(const std::vector<float>& x, bool applyHann = true) const;

    // ignore?
    TransferResult estimateTransferFunction(
        const std::vector<float>& input,
        const std::vector<float>& output,
        bool applyHann = true) const;

    // frequency response of resulting learnt FIR adaptive filter
    TransferResult computeFIRFrequencyResponse(
        const std::vector<float>& taps) const;

    // Simple one-block coherence estimate
    CoherenceResult estimateCoherence(
        const std::vector<float>& x,
        const std::vector<float>& y,
        bool applyHann = true) const;

    // error surface visualization for a 2-tap filter (for educational purposes)
    void plotErrorSurface(
        const std::vector<float>& x,
        const std::vector<float>& d,
        float w0_min, float w0_max,
        float w1_min, float w1_max,
        int resolution) const;

    // generated
    void plotSpectrum(const SpectrumResult& spec, const std::string& title) const;
    void plotTransfer(const TransferResult& tf, const std::string& title) const;
    void plotCoherence(const CoherenceResult& coh, const std::string& title = "Coherence") const;

private:
    float fs;

    std::vector<float> hannWindow(size_t N) const;
    std::vector<std::complex<float>> dft(const std::vector<float>& x) const;
};