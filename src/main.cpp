#include <iostream>
#include "params.hpp"
#include "dsp/fir.hpp"
#include "algorithm/fxlms.hpp"

int main(){
  // header-only math sanity
  float h[3] = {0.2f, 0.3f, 0.5f};
  float x_hist_newest_end[3] = {3.f, 2.f, 1.f}; 
  float y = pn::fir_dot(h, x_hist_newest_end, 3); // = 2.3
  std::cout << "fir_dot test: " << y << "\n";

  // type visibility sanity
  FxParams fp; pn::FxLMS fx(fp);
  std::cout << "includes OK, build OK.\n";
  return (y > 2.29f && y < 2.31f) ? 0 : 1;
}
