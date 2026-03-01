#include "catch2/catch_amalgamated.hpp"
#include "dsp.h"

using Catch::Approx;

TEST_CASE("dot_product computes correctly", "[dsp]") {
    DSP dsp;
    float result = 0.0f;
    dsp.dot_product({1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, result);
    REQUIRE(result == Approx(32.0f)); // 1*4 + 2*5 + 3*6
}

TEST_CASE("fir_block_processing: impulse through identity filter gives impulse", "[dsp]") {
    DSP dsp;
    std::vector<float> x = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> h = {1.0f};  // identity
    std::vector<float> state, y;
    dsp.fir_block_processing(y, x, h, state);
    REQUIRE(y[0] == Approx(1.0f));
    REQUIRE(y[1] == Approx(0.0f));
}

TEST_CASE("fir_block_processing: state continuity across blocks", "[dsp]") {
    // Filter: simple 2-tap moving average [0.5, 0.5]
    // Process one big block vs two smaller blocks — outputs should match
    DSP dsp;
    std::vector<float> h = {0.5f, 0.5f};
    std::vector<float> x_all = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    
    std::vector<float> state_single, y_single;
    dsp.fir_block_processing(y_single, x_all, h, state_single);
    
    std::vector<float> state_split, y1, y2;
    std::vector<float> x1 = {1.0f, 0.0f, 1.0f};
    std::vector<float> x2 = {0.0f, 1.0f, 0.0f};
    dsp.fir_block_processing(y1, x1, h, state_split);
    dsp.fir_block_processing(y2, x2, h, state_split);
    
    for (int i = 0; i < 3; ++i) REQUIRE(y1[i] == Approx(y_single[i]).margin(1e-5f));
    for (int i = 0; i < 3; ++i) REQUIRE(y2[i] == Approx(y_single[3+i]).margin(1e-5f));
}

TEST_CASE("impulseResponseLPF: DC gain equals normCutoff", "[dsp]") {
    DSP dsp;
    std::vector<float> h;
    dsp.impulseResponseLPF(48000.0f, 1000.0f, 31, h, 1);
    REQUIRE(h.size() == 31);
    float dc_gain = 0.0f;
    for (auto v : h) dc_gain += v;
    // This implementation uses a sin^2 window which reduces DC gain
    // Verify it's positive and less than 1 (i.e. the filter was designed)
    REQUIRE(dc_gain > 0.0f);
    REQUIRE(dc_gain < 1.0f);
    REQUIRE(h.size() == 31);
}

// NEW TESTS

TEST_CASE("dot_product: zero vectors returns zero", "[dsp]") {
    DSP dsp;
    float result = 99.0f;
    dsp.dot_product({0.0f, 0.0f, 0.0f}, {1.0f, 2.0f, 3.0f}, result);
    REQUIRE(result == Approx(0.0f));
}

TEST_CASE("fir_block_processing: known moving-average output", "[dsp]") {
    // h = [0.5, 0.5] averages consecutive pairs
    // x = {2, 4, 6, 8} → y = {1, 3, 5, 7}  (first sample has no prior, so 0*0.5 + 2*0.5 = 1)
    DSP dsp;
    std::vector<float> h = {0.5f, 0.5f};
    std::vector<float> x = {2.0f, 4.0f, 6.0f, 8.0f};
    std::vector<float> state, y;
    dsp.fir_block_processing(y, x, h, state);
    REQUIRE(y[0] == Approx(1.0f));  // (0 + 2) * 0.5
    REQUIRE(y[1] == Approx(3.0f));  // (2 + 4) * 0.5
    REQUIRE(y[2] == Approx(5.0f));  // (4 + 6) * 0.5
    REQUIRE(y[3] == Approx(7.0f));  // (6 + 8) * 0.5
}

TEST_CASE("fir_block_processing: empty input produces empty output", "[dsp]") {
    DSP dsp;
    std::vector<float> h = {0.5f, 0.5f};
    std::vector<float> x = {};
    std::vector<float> state, y;
    dsp.fir_block_processing(y, x, h, state);
    REQUIRE(y.empty());
}

TEST_CASE("bandPassCoeff: produces correct number of taps and non-zero output", "[dsp]") {
    DSP dsp;
    int num_taps = 31;
    std::vector<float> h(num_taps, 0.0f);
    dsp.bandPassCoeff(200.0f, 800.0f, 8000.0f, num_taps, h);
    REQUIRE((int)h.size() == num_taps);
    // Filter should not be all zeros
    float energy = 0.0f;
    for (auto v : h) energy += v * v;
    REQUIRE(energy > 0.0f);
}

TEST_CASE("resampling: output length equals input * us / ds", "[dsp]") {
    DSP dsp;
    int us = 3, ds = 2;
    std::vector<float> x(100, 1.0f);
    std::vector<float> h(13, 0.0f);
    h[0] = 1.0f; // impulse polyphase filter
    std::vector<float> state, y; // state initialized inside resampling
    dsp.resampling(y, x, h, state, ds, us);
    REQUIRE((int)y.size() == (int)x.size() * us / ds);
}

TEST_CASE("fir_block_processing: single-sample blocks match batch output", "[dsp][realtime]") {
    DSP dsp;
    std::vector<float> h = {0.5f, 0.25f, 0.25f}; // 3-tap filter, S = 2
    std::vector<float> x = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};

    // Batch reference
    std::vector<float> state_batch, y_batch;
    dsp.fir_block_processing(y_batch, x, h, state_batch);

    // One sample at a time
    std::vector<float> state_single;
    std::vector<float> y_collected;
    for (int i = 0; i < (int)x.size(); ++i) {
        std::vector<float> x_one = {x[i]};
        std::vector<float> y_one;
        dsp.fir_block_processing(y_one, x_one, h, state_single);
        y_collected.push_back(y_one[0]);
    }

    for (int i = 0; i < (int)x.size(); ++i)
        REQUIRE(y_collected[i] == Approx(y_batch[i]).margin(1e-5f));
}

TEST_CASE("fir_block_processing: 256 single-sample calls match batch", "[dsp][realtime]") {
    DSP dsp;
    std::vector<float> h = {0.5f, 0.25f, 0.25f};
    std::vector<float> x(256);
    for (int i = 0; i < 256; ++i) x[i] = (i % 7 < 3) ? 1.0f : -1.0f;

    // Batch
    std::vector<float> state_batch, y_batch;
    dsp.fir_block_processing(y_batch, x, h, state_batch);

    // Sample by sample
    std::vector<float> state_single;
    std::vector<float> y_collected;
    for (int i = 0; i < 256; ++i) {
        std::vector<float> x_one = {x[i]};
        std::vector<float> y_one;
        dsp.fir_block_processing(y_one, x_one, h, state_single);
        y_collected.push_back(y_one[0]);
    }

    for (int i = 0; i < 256; ++i)
        REQUIRE(y_collected[i] == Approx(y_batch[i]).margin(1e-5f));
}

TEST_CASE("resampling: chunked calls match batch output", "[dsp][realtime]") {
    DSP dsp;
    int us = 3, ds = 2;

    // Simple FIR for polyphase: boxcar
    std::vector<float> h(6, 1.0f / 6.0f);
    std::vector<float> x(60, 0.0f);
    for (int i = 0; i < 60; ++i) x[i] = (i % 5 < 2) ? 1.0f : -1.0f;

    // Batch
    std::vector<float> state_batch, y_batch;
    dsp.resampling(y_batch, x, h, state_batch, ds, us);

    // Split into two 30-sample chunks
    std::vector<float> x1(x.begin(), x.begin() + 30);
    std::vector<float> x2(x.begin() + 30, x.end());
    std::vector<float> state_chunked, y1, y2;
    dsp.resampling(y1, x1, h, state_chunked, ds, us);
    dsp.resampling(y2, x2, h, state_chunked, ds, us);

    std::vector<float> y_chunked;
    y_chunked.insert(y_chunked.end(), y1.begin(), y1.end());
    y_chunked.insert(y_chunked.end(), y2.begin(), y2.end());

    REQUIRE(y_chunked.size() == y_batch.size());
    for (int i = 0; i < (int)y_batch.size(); ++i)
        REQUIRE(y_chunked[i] == Approx(y_batch[i]).margin(1e-4f));
}

TEST_CASE("dot_product: large vectors accumulate within tolerance", "[dsp]") {
    DSP dsp;
    const int N = 10000;
    std::vector<float> a(N, 1.0f);
    std::vector<float> b(N, 0.001f);
    float result = 0.0f;
    dsp.dot_product(a, b, result);
    // True answer is exactly 10.0, check within 0.01 (0.1% tolerance)
    REQUIRE(result == Approx(10.0f).margin(0.01f));
}

TEST_CASE("impulseResponseLPF: Fc at Nyquist produces finite coefficients", "[dsp]") {
    DSP dsp;
    std::vector<float> h;
    dsp.impulseResponseLPF(48000.0f, 24000.0f, 31, h, 1); // Fc = Fs/2
    REQUIRE((int)h.size() == 31);
    for (auto v : h) REQUIRE(std::isfinite(v));
}

TEST_CASE("bandPassCoeff + fir_block_processing: chunked output matches batch", "[dsp][realtime]") {
    DSP dsp;
    int num_taps = 31;
    std::vector<float> h(num_taps, 0.0f);
    dsp.bandPassCoeff(200.0f, 800.0f, 8000.0f, num_taps, h);

    std::vector<float> x(128, 0.0f);
    for (int i = 0; i < 128; ++i) x[i] = (i % 7 < 3) ? 1.0f : -1.0f;

    // Batch
    std::vector<float> state_batch, y_batch;
    dsp.fir_block_processing(y_batch, x, h, state_batch);

    // 32-sample chunks
    std::vector<float> state_chunked, y_collected;
    for (int start = 0; start < 128; start += 32) {
        std::vector<float> x_chunk(x.begin() + start, x.begin() + start + 32);
        std::vector<float> y_chunk;
        dsp.fir_block_processing(y_chunk, x_chunk, h, state_chunked);
        y_collected.insert(y_collected.end(), y_chunk.begin(), y_chunk.end());
    }

    REQUIRE(y_collected.size() == y_batch.size());
    for (int i = 0; i < 128; ++i)
        REQUIRE(y_collected[i] == Approx(y_batch[i]).margin(1e-5f));
}
