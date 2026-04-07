#include "signal_testing.h"
#include "matplotlibcpp.h"

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace plt = matplotlibcpp;

namespace {
    constexpr float PI_F = 3.14159265358979323846f;
    constexpr float EPS  = 1e-12f;
}

SignalTesting::SignalTesting(float sampleRate) : fs(sampleRate) {
    if (fs <= 0.0f) {
        throw std::invalid_argument("SignalTesting: sample rate must be > 0");
    }
}

float SignalTesting::computeMean(const std::vector<float>& x) const {
    if (x.empty()) return 0.0f;

    double sum = 0.0;
    for (float v : x) sum += v;
    return static_cast<float>(sum / static_cast<double>(x.size()));
}

float SignalTesting::computeRMS(const std::vector<float>& x) const {
    if (x.empty()) return 0.0f;

    double sumSq = 0.0;
    for (float v : x) sumSq += static_cast<double>(v) * static_cast<double>(v);
    return static_cast<float>(std::sqrt(sumSq / static_cast<double>(x.size())));
}

float SignalTesting::computePeak(const std::vector<float>& x) const {
    if (x.empty()) return 0.0f;

    float peak = 0.0f;
    for (float v : x) peak = std::max(peak, std::fabs(v));
    return peak;
}

std::vector<float> SignalTesting::hannWindow(size_t N) const {
    std::vector<float> w(N, 1.0f);
    if (N <= 1) return w;

    for (size_t n = 0; n < N; ++n) {
        w[n] = 0.5f * (1.0f - std::cos((2.0f * PI_F * static_cast<float>(n)) / static_cast<float>(N - 1)));
    }
    return w;
}

// Simple DFT for debugging / visualization
std::vector<std::complex<float>> SignalTesting::dft(const std::vector<float>& x) const {
    const size_t N = x.size();
    std::vector<std::complex<float>> X(N, {0.0f, 0.0f});

    for (size_t k = 0; k < N; ++k) {
        std::complex<float> acc(0.0f, 0.0f);

        for (size_t n = 0; n < N; ++n) {
            float angle = -2.0f * PI_F * static_cast<float>(k * n) / static_cast<float>(N);
            std::complex<float> twiddle(std::cos(angle), std::sin(angle));
            acc += x[n] * twiddle;
        }

        X[k] = acc;
    }

    return X;
}

SignalTesting::SpectrumResult SignalTesting::computeSpectrum(const std::vector<float>& x, bool applyHann) const {
    if (x.empty()) {
        throw std::invalid_argument("computeSpectrum: input signal is empty");
    }

    std::vector<float> xProc = x;

    if (applyHann) {
        auto w = hannWindow(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            xProc[i] *= w[i];
        }
    }

    const auto X = dft(xProc);
    const size_t N = xProc.size();
    const size_t half = N / 2 + 1;

    SpectrumResult out;
    out.freqs_hz.resize(half);
    out.magnitude.resize(half);
    out.power_db.resize(half);
    out.phase_rad.resize(half);

    for (size_t k = 0; k < half; ++k) {
        float mag = std::abs(X[k]) / static_cast<float>(N);
        float pwr = mag * mag;

        out.freqs_hz[k]  = static_cast<float>(k) * fs / static_cast<float>(N);
        out.magnitude[k] = mag;
        out.power_db[k]  = 10.0f * std::log10(pwr + EPS);
        out.phase_rad[k] = std::arg(X[k]);
    }

    return out;
}

SignalTesting::TransferResult SignalTesting::estimateTransferFunction(
    const std::vector<float>& input,
    const std::vector<float>& output,
    bool applyHann) const
{
    if (input.size() != output.size() || input.empty()) {
        throw std::invalid_argument("estimateTransferFunction: input/output size mismatch or empty");
    }

    std::vector<float> xProc = input;
    std::vector<float> yProc = output;

    if (applyHann) {
        auto w = hannWindow(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            xProc[i] *= w[i];
            yProc[i] *= w[i];
        }
    }

    const auto X = dft(xProc);
    const auto Y = dft(yProc);

    const size_t N = input.size();
    const size_t half = N / 2 + 1;

    TransferResult out;
    out.freqs_hz.resize(half);
    out.magnitude.resize(half);
    out.magnitude_db.resize(half);
    out.phase_rad.resize(half);

    for (size_t k = 0; k < half; ++k) {
        std::complex<float> H = Y[k] / (X[k] + std::complex<float>(EPS, 0.0f));
        float mag = std::abs(H);

        out.freqs_hz[k]     = static_cast<float>(k) * fs / static_cast<float>(N);
        out.magnitude[k]    = mag;
        out.magnitude_db[k] = 20.0f * std::log10(mag + EPS);
        out.phase_rad[k]    = std::arg(H);
    }

    return out;
}

SignalTesting::TransferResult SignalTesting::computeFIRFrequencyResponse(
    const std::vector<float>& taps) const
{
    if (taps.empty()) {
        throw std::invalid_argument("computeFIRFrequencyResponse: taps empty");
    }

    size_t N = 1;
    while (N < taps.size() * 8) N <<= 1;

    std::vector<float> h(N, 0.0f);
    for (size_t i = 0; i < taps.size(); ++i) {
        h[i] = taps[i];
    }

    const auto H = dft(h);
    const size_t half = N / 2 + 1;

    TransferResult out;
    out.freqs_hz.resize(half);
    out.magnitude.resize(half);
    out.magnitude_db.resize(half);
    out.phase_rad.resize(half);

    for (size_t k = 0; k < half; ++k) {
        float mag = std::abs(H[k]);

        out.freqs_hz[k]     = static_cast<float>(k) * fs / static_cast<float>(N);
        out.magnitude[k]    = mag;
        out.magnitude_db[k] = 20.0f * std::log10(mag + EPS);
        out.phase_rad[k]    = std::arg(H[k]);
    }

    return out;
}

SignalTesting::CoherenceResult SignalTesting::estimateCoherence(
    const std::vector<float>& x,
    const std::vector<float>& y,
    bool applyHann) const
{
    if (x.size() != y.size() || x.empty()) {
        throw std::invalid_argument("estimateCoherence: x/y size mismatch or empty");
    }

    std::vector<float> xProc = x;
    std::vector<float> yProc = y;

    if (applyHann) {
        auto w = hannWindow(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            xProc[i] *= w[i];
            yProc[i] *= w[i];
        }
    }

    const auto X = dft(xProc);
    const auto Y = dft(yProc);

    const size_t N = x.size();
    const size_t half = N / 2 + 1;

    CoherenceResult out;
    out.freqs_hz.resize(half);
    out.coherence.resize(half);

    for (size_t k = 0; k < half; ++k) {
        std::complex<float> Gxy = X[k] * std::conj(Y[k]);
        float Gxx = std::norm(X[k]);
        float Gyy = std::norm(Y[k]);

        float coh = std::norm(Gxy) / ((Gxx * Gyy) + EPS);
        if (coh > 1.0f) coh = 1.0f;

        out.freqs_hz[k]  = static_cast<float>(k) * fs / static_cast<float>(N);
        out.coherence[k] = coh;
    }

    return out;
}

void SignalTesting::plotSpectrum(const SpectrumResult& spec, const std::string& title) const {
    std::vector<double> x(spec.freqs_hz.begin(), spec.freqs_hz.end());
    std::vector<double> y(spec.power_db.begin(), spec.power_db.end());

    plt::figure();
    plt::plot(x, y);
    plt::title(title);
    plt::xlabel("Frequency (Hz)");
    plt::ylabel("Power (dB)");
    plt::grid(true);
    plt::show();
}

void SignalTesting::plotTransfer(const TransferResult& tf, const std::string& title) const {
    std::vector<double> x(tf.freqs_hz.begin(), tf.freqs_hz.end());
    std::vector<double> mag(tf.magnitude_db.begin(), tf.magnitude_db.end());
    std::vector<double> phase(tf.phase_rad.begin(), tf.phase_rad.end());

    plt::figure();

    plt::subplot(2, 1, 1);
    plt::plot(x, mag);
    plt::title(title);
    plt::ylabel("Magnitude (dB)");
    plt::grid(true);

    plt::subplot(2, 1, 2);
    plt::plot(x, phase);
    plt::xlabel("Frequency (Hz)");
    plt::ylabel("Phase (rad)");
    plt::grid(true);

    plt::show();
}

void SignalTesting::plotCoherence(const CoherenceResult& coh, const std::string& title) const {
    std::vector<double> x(coh.freqs_hz.begin(), coh.freqs_hz.end());
    std::vector<double> y(coh.coherence.begin(), coh.coherence.end());

    plt::figure();
    plt::plot(x, y);
    plt::title(title);
    plt::xlabel("Frequency (Hz)");
    plt::ylabel("Coherence");
    plt::ylim(0.0, 1.05);
    plt::grid(true);
    plt::show();
}

void SignalTesting::plotErrorSurface(
    const std::vector<float>& x,
    const std::vector<float>& d,
    float w0_min, float w0_max,
    float w1_min, float w1_max,
    int resolution) const
{
    if (x.size() != d.size() || x.size() < 2) {
        throw std::invalid_argument("plotErrorSurface: x and d must have same size >= 2");
    }
    if (resolution < 2) {
        throw std::invalid_argument("plotErrorSurface: resolution must be >= 2");
    }

    std::vector<std::vector<double>> Z(resolution, std::vector<double>(resolution));

    float dw0 = (w0_max - w0_min) / static_cast<float>(resolution - 1);
    float dw1 = (w1_max - w1_min) / static_cast<float>(resolution - 1);

    for (int i = 0; i < resolution; i++) {
        float w0 = w0_min + static_cast<float>(i) * dw0;

        for (int j = 0; j < resolution; j++) {
            float w1 = w1_min + static_cast<float>(j) * dw1;

            double J = 0.0;
            for (size_t n = 1; n < x.size(); n++) {
                float yhat = w0 * x[n] + w1 * x[n - 1];
                float e = d[n] - yhat;
                J += static_cast<double>(e) * static_cast<double>(e);
            }
            Z[i][j] = J / static_cast<double>(x.size() - 1);
        }
    }

    plt::figure();
    plt::imshow(Z);
    plt::title("Error Surface (w0 vs w1)");
    plt::colorbar();
    plt::show();
}