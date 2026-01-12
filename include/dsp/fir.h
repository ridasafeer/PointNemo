#pragma once
#include <cstddef>
namespace pn {
inline float fir_dot(const float* h, const float* x_hist, std::size_t L){
  float acc=0.f; for(std::size_t k=0;k<L;k++) acc += h[k]*x_hist[L-1-k]; return acc;
}
}
