#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>

namespace pn {

// ---------- helpers ----------
// --- existing helpers (example) ---
inline float fir_dot(const float* h, const float* x_hist, std::size_t L){
  float acc = 0.f;
  for (std::size_t k = 0; k < L; ++k)
    acc += h[k] * x_hist[L-1-k];
  return acc;
}

// 🔹 add THIS:
inline float dot(const float* a, const float* b, std::size_t n){
  float acc = 0.f;
  for (std::size_t i = 0; i < n; ++i)
    acc += a[i] * b[i];
  return acc;
}
inline float sinc_pi(float x){
  return (std::fabs(x) < 1e-8f) ? 1.f : std::sin(3.14159265358979323846f * x) / (3.14159265358979323846f * x);
}

inline std::vector<float> hann(std::size_t N){
  std::vector<float> w(N);
  if (N <= 1) return w;
  for (std::size_t n=0; n<N; ++n){
    w[n] = 0.5f - 0.5f * std::cos(2.f * 3.14159265358979323846f * float(n) / float(N-1));
  }
  return w;
}

// ---------- design: FIR band-pass (windowed-sinc) ----------
inline std::vector<float> fir_bandpass(unsigned fs, float f1, float f2, std::size_t taps){
  const std::size_t M = std::max<std::size_t>(taps, 3);
  const float n0 = 0.5f * float(M - 1);
  const float fc1 = f1 / float(fs);
  const float fc2 = f2 / float(fs);

  std::vector<float> h(M);
  auto w = hann(M);

  for (std::size_t n=0; n<M; ++n){
    float k = float(n) - n0;
    float lp2 = 2.f * fc2 * sinc_pi(2.f * fc2 * k);
    float lp1 = 2.f * fc1 * sinc_pi(2.f * fc1 * k);
    h[n] = (lp2 - lp1) * w[n];
  }
  // normalize sum
  float s=0.f; for (float v: h) s += v;
  if (std::fabs(s) > 1e-12f) for (auto& v: h) v /= s;
  return h;
}


inline std::vector<float> design_bandpass(unsigned fs, float f_lo, float f_hi, std::size_t taps){
  return fir_bandpass(fs, f_lo, f_hi, taps);
}

// ---------- apply: convolution helpers ----------
inline std::vector<float> convolve_full(const std::vector<float>& x, const std::vector<float>& h){
  const std::size_t N = x.size(), M = h.size();
  std::vector<float> y(N + M - 1, 0.f);
  for (std::size_t n=0; n<N; ++n){
    const float xn = x[n];
    for (std::size_t k=0; k<M; ++k){
      y[n + k] += xn * h[k];
    }
  }
  return y;
}

// Linear-phase FIR application with group-delay trim (like your Python fallback)
inline std::vector<float> bandpass_signal(const std::vector<float>& x, const std::vector<float>& bp){
  auto y_full = convolve_full(x, bp);
  const std::size_t gd = (bp.size() - 1) / 2; // group delay
  std::vector<float> y(x.size(), 0.f);
  if (y_full.size() >= gd + x.size()){
    std::copy(y_full.begin() + gd, y_full.begin() + gd + x.size(), y.begin());
  }
  return y;
}

// ---------- optional: FIR filtfilt (zero-phase-ish) ----------
inline std::vector<float> pad_reflect(const std::vector<float>& x, std::size_t P){
  if (P == 0 || x.empty()) return x;
  std::vector<float> y; y.reserve(P + x.size() + P);
  // left reflect
  for (std::size_t i=0; i<P; ++i){
    std::size_t idx = std::min(P - i, x.size()-1); // reflect within range
    y.push_back(x[idx]);
  }
  // original
  y.insert(y.end(), x.begin(), x.end());
  // right reflect
  for (std::size_t i=0; i<P; ++i){
    std::size_t idx = x.size() - 2 - std::min(i, x.size()-2);
    y.push_back(x[idx]);
  }
  return y;
}

// forward-backward FIR: y = reverse( conv( reverse( conv(x,b) ), b) ), with padding
inline std::vector<float> filtfilt_fir(const std::vector<float>& b, const std::vector<float>& x){
  const std::size_t M = b.size();
  const std::size_t P = (M > 1) ? (M - 1) : 0; // simple pad length
  auto xpad = pad_reflect(x, P);

  // forward
  auto y1_full = convolve_full(xpad, b);

  // trim to original+pad length
  std::vector<float> y1(xpad.size(), 0.f);
  const std::size_t start1 = (M - 1);
  if (y1_full.size() >= start1 + xpad.size()){
    std::copy(y1_full.begin() + start1, y1_full.begin() + start1 + xpad.size(), y1.begin());
  } else {
    // fallback if short
    std::copy(y1_full.begin(), y1_full.begin() + std::min(y1_full.size(), y1.size()), y1.begin());
  }

  // reverse, filter, reverse back
  std::reverse(y1.begin(), y1.end());
  auto y2_full = convolve_full(y1, b);

  // center-trim to xpad length
  std::vector<float> y2(y1.size(), 0.f);
  const std::size_t start2 = (M - 1);
  if (y2_full.size() >= start2 + y1.size()){
    std::copy(y2_full.begin() + start2, y2_full.begin() + start2 + y1.size(), y2.begin());
  } else {
    std::copy(y2_full.begin(), y2_full.begin() + std::min(y2_full.size(), y2.size()), y2.begin());
  }
  std::reverse(y2.begin(), y2.end());

  // remove padding
  std::vector<float> out(x.size(), 0.f);
  if (y2.size() >= 2*P + x.size()){
    std::copy(y2.begin() + P, y2.begin() + P + x.size(), out.begin());
  } else {
    // fallback
    std::copy(y2.begin(), y2.begin() + std::min(y2.size(), out.size()), out.begin());
  }
  return out;
}

} // namespace pn
