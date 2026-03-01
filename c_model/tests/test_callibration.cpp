#include "catch2/catch_amalgamated.hpp"
#include "controller.h"
#include <cmath>
#include <numeric>
#include <random>

using Catch::Approx;

// helpers 

static std::vector<float> convolve(const std::vector<float>& x, const std::vector<float>& h) {
    std::vector<float> y(x.size(), 0.0f);
    for (int n = 0; n < (int)x.size(); ++n)
        for (int k = 0; k < (int)h.size() && k <= n; ++k)
            y[n] += h[k] * x[n - k];
    return y;
}

static std::vector<float> add_noise(const std::vector<float>& x, float std_dev, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, std_dev);
    std::vector<float> noisy = x;
    for (auto& s : noisy) s += dist(rng);
    return noisy;
}

static float l1_error(const std::vector<float>& a, const std::vector<float>& b) {
    float err = 0.0f;
    for (int i = 0; i < (int)a.size(); ++i) err += std::abs(a[i] - b[i]);
    return err;
}

// offline tests

TEST_CASE("calibration converges to a known 1-tap system", "[calibration]") {
    // True secondary path: pure gain of 0.5
    std::vector<float> true_s = {0.5f};
    std::vector<float> x_exc(512, 0.0f);
    // Use a PRBS-like pattern
    for (int i = 0; i < 512; ++i) x_exc[i] = (i % 7 < 3) ? 1.0f : -1.0f;
    
    std::vector<float> y_mic = convolve(x_exc, true_s);
    
    Controller ctrl({0.5f}, 1, 0.01f); // dummy shat to construct
    auto shat = ctrl.calibration(x_exc, y_mic, 1, 0.01f, 10);
    
    REQUIRE(shat.size() == 1);
    REQUIRE(shat[0] == Approx(0.5f).margin(0.01f));
}   

TEST_CASE("calibration converges to a known 3-tap FIR", "[calibration]") {
    std::vector<float> true_s = {0.6f, -0.3f, 0.1f};
    std::vector<float> x_exc(1024, 0.0f);
    for (int i = 0; i < 1024; ++i) x_exc[i] = (i % 5 < 2) ? 1.0f : -1.0f;
    
    std::vector<float> y_mic = convolve(x_exc, true_s);
    Controller ctrl(true_s, 3, 0.01f);
    auto shat = ctrl.calibration(x_exc, y_mic, 3, 0.01f, 20);
    
    REQUIRE(shat[0] == Approx(0.6f).margin(0.02f));
    REQUIRE(shat[1] == Approx(-0.3f).margin(0.02f));
    REQUIRE(shat[2] == Approx(0.1f).margin(0.02f));
}

TEST_CASE("calibration throws on bad inputs", "[calibration]") {
    Controller ctrl({0.5f}, 1, 0.01f);
    std::vector<float> x = {1.0f, 2.0f};
    std::vector<float> y = {1.0f};
    
    REQUIRE_THROWS_AS(ctrl.calibration(x, y, 1, 0.01f),   std::invalid_argument); // size mismatch
    REQUIRE_THROWS_AS(ctrl.calibration(x, x, 0, 0.01f),   std::invalid_argument); // L <= 0
    REQUIRE_THROWS_AS(ctrl.calibration(x, x, 1, 0.0f),    std::invalid_argument); // mu <= 0
    REQUIRE_THROWS_AS(ctrl.calibration(x, x, 1, 0.01f, 1, 1.5f), std::invalid_argument); // bad leak >= 1
    REQUIRE_THROWS_AS(ctrl.calibration(x, x, 1, 0.01f, 1, -0.1f), std::invalid_argument); // bad leak < 0
}


TEST_CASE("calibration: zero excitation produces zero shat", "[calibration]") {
    // If the excitation is all zeros then LMS has nothing to learn so shat stays zero
    Controller ctrl({0.5f}, 3, 0.01f);
    std::vector<float> x_exc(256, 0.0f);
    std::vector<float> y_mic(256, 0.0f);
    auto shat = ctrl.calibration(x_exc, y_mic, 3, 0.01f, 5);
    for (auto c : shat) REQUIRE(c == Approx(0.0f).margin(1e-6f));
}

TEST_CASE("calibration: more passes converges closer to true value", "[calibration]") {
    // More LMS passes over the same data should reduce the identification error
    std::vector<float> true_s = {0.7f, -0.2f};
    std::vector<float> x_exc(512, 0.0f);
    for (int i = 0; i < 512; ++i) x_exc[i] = (i % 7 < 3) ? 1.0f : -1.0f;
    std::vector<float> y_mic = convolve(x_exc, true_s);

    Controller ctrl(true_s, 2, 0.01f);
    auto shat_1pass  = ctrl.calibration(x_exc, y_mic, 2, 0.01f, 1);
    auto shat_20pass = ctrl.calibration(x_exc, y_mic, 2, 0.01f, 20);

    float err_1  = std::abs(shat_1pass[0]  - true_s[0]) + std::abs(shat_1pass[1]  - true_s[1]);
    float err_20 = std::abs(shat_20pass[0] - true_s[0]) + std::abs(shat_20pass[1] - true_s[1]);
    REQUIRE(err_20 < err_1);
}

TEST_CASE("calibration: leakage biases taps toward zero", "[calibration]") {
    // Leakage shrinks taps each sample — converged taps with leakage should have
    // smaller total magnitude than without leakage
    std::vector<float> true_s = {0.5f, 0.1f};
    std::vector<float> x_exc(512, 0.0f);
    for (int i = 0; i < 512; ++i) x_exc[i] = (i % 7 < 3) ? 1.0f : -1.0f;
    std::vector<float> y_mic = convolve(x_exc, true_s);

    Controller ctrl(true_s, 2, 0.01f);
    auto shat_no_leak   = ctrl.calibration(x_exc, y_mic, 2, 0.01f, 10, 0.0f);
    auto shat_with_leak = ctrl.calibration(x_exc, y_mic, 2, 0.01f, 10, 0.05f);

    float mag_no_leak   = std::abs(shat_no_leak[0])   + std::abs(shat_no_leak[1]);
    float mag_with_leak = std::abs(shat_with_leak[0]) + std::abs(shat_with_leak[1]);
    REQUIRE(mag_with_leak < mag_no_leak);
}

// realtime tests 

TEST_CASE("realtime: chunked processing matches batch result", "[realtime]") {
    std::vector<float> true_s = {0.6f, -0.3f, 0.1f};
    const int N = 512;
    const int CHUNK = 64;

    std::vector<float> x_exc(N), y_mic(N);
    for (int i = 0; i < N; ++i) x_exc[i] = (i % 7 < 3) ? 1.0f : -1.0f;
    y_mic = convolve(x_exc, true_s);

    // batch reference
    Controller ctrl_batch(true_s, 3, 0.01f);
    auto shat_batch = ctrl_batch.calibration(x_exc, y_mic, 3, 0.01f, 1);

    // chunks feed 64 samples at a time across 8 calls
    // the controller must preserve xhist and shat state across calls rather than reset each time
    Controller ctrl_chunk(true_s, 3, 0.01f);
    std::vector<float> shat_chunk;
    for (int start = 0; start < N; start += CHUNK) {
        std::vector<float> x_chunk(x_exc.begin() + start, x_exc.begin() + start + CHUNK);
        std::vector<float> y_chunk(y_mic.begin()  + start, y_mic.begin()  + start + CHUNK);
        shat_chunk = ctrl_chunk.calibration(x_chunk, y_chunk, 3, 0.01f, 1);
    }

    REQUIRE(l1_error(shat_chunk, shat_batch) < 0.05f);
}

TEST_CASE("realtime: tap state persists and improves across successive calls", "[realtime]") {
    std::vector<float> true_s = {0.5f, -0.2f};
    const int N = 512;
    std::vector<float> x_exc(N), y_mic(N);
    for (int i = 0; i < N; ++i) x_exc[i] = (i % 7 < 3) ? 1.0f : -1.0f;
    y_mic = convolve(x_exc, true_s);

    Controller ctrl(true_s, 2, 0.01f);

    // First call where taps start from zero
    auto shat_after_1 = ctrl.calibration(x_exc, y_mic, 2, 0.01f, 1);
    float err_1 = l1_error(shat_after_1, true_s);

    // Second call where the first call left off
    auto shat_after_2 = ctrl.calibration(x_exc, y_mic, 2, 0.01f, 1);
    float err_2 = l1_error(shat_after_2, true_s);

    // Error must decrease to prove state carried over between calls
    REQUIRE(err_2 < err_1);
}

TEST_CASE("realtime: taps remain bounded over a long run", "[realtime]") {
    std::vector<float> true_s = {0.6f, -0.3f, 0.1f};
    const int N = 50000;

    std::vector<float> x_exc(N), y_mic(N);
    for (int i = 0; i < N; ++i) x_exc[i] = (i % 7 < 3) ? 1.0f : -1.0f;
    y_mic = convolve(x_exc, true_s);

    Controller ctrl(true_s, 3, 0.01f);
    auto shat = ctrl.calibration(x_exc, y_mic, 3, 0.01f, 1);

    for (auto c : shat) {
        REQUIRE(std::isfinite(c));      // no bullshit
        REQUIRE(std::abs(c) < 10.0f);  // not exploding
        REQUIRE(std::abs(c) > 1e-6f);  // not zero
    }
}
TEST_CASE("realtime: converges reasonably under noisy mic signal", "[realtime]") {
    std::vector<float> true_s = {0.6f, -0.3f, 0.1f};
    const int N = 2048;

    std::vector<float> x_exc(N), y_clean(N);
    for (int i = 0; i < N; ++i) x_exc[i] = (i % 7 < 3) ? 1.0f : -1.0f;
    y_clean = convolve(x_exc, true_s);

    auto y_noisy = add_noise(y_clean, 0.05f); // noise std dev = 0.05

    // use smaller mu and more passes to compensate for noise
    Controller ctrl(true_s, 3, 0.01f);
    auto shat = ctrl.calibration(x_exc, y_noisy, 3, 0.005f, 10);

    // Loose margin where noise prevents tight convergence but result should be same same
    REQUIRE(shat[0] == Approx(0.6f).margin(0.1f));
    REQUIRE(shat[1] == Approx(-0.3f).margin(0.1f));
    REQUIRE(shat[2] == Approx(0.1f).margin(0.1f));
}

TEST_CASE("realtime: large step size causes divergence", "[realtime]") {
    std::vector<float> true_s = {0.5f, 0.1f};
    const int N = 512;
    std::vector<float> x_exc(N), y_mic(N);
    for (int i = 0; i < N; ++i) x_exc[i] = (i % 7 < 3) ? 1.0f : -1.0f;
    y_mic = convolve(x_exc, true_s);

    Controller ctrl(true_s, 2, 10.0f); // mu = 10 L = 2
    auto shat = ctrl.calibration(x_exc, y_mic, 2, 10.0f, 1);

    // bullshit result check
    bool diverged = !std::isfinite(shat[0]) || !std::isfinite(shat[1]) || l1_error(shat, true_s) > 1.0f;
    REQUIRE(diverged);
}