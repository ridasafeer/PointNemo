//Class definition for DSP functions: Utility functions for digital signal processing 
// - Block processing functions for DSP operations inside FxLMS
// - DSP utility functions for computations: convolution, simple filtering

#include <vector>
#include <cmath>
#include <iostream>

#include "dsp.h"

// Add DSP functions here
// y-> ouput x-> input h->impulse
void fir_block_processing(std::vector<float>& y, const std::vector<float>& x, const std::vector<float>& h, std::vector<float>& state)
{
    y.clear();
    y.resize(x.size(), 0.0);

    //convolution
    for (int i = 0; i < x.size(); ++i) {
        for (int j = 0; j < h.size(); ++j) {
            if (i-j >= 0){
                y[i] = h[j] * x[i-j];
            }
            else {
                y[i] = h[j] * state[(i-j) + (h.size() - 1)];
            }
        }
    }

    // update state: keep last (hsize-1) samples of x
    for (int i = 0; i < h.size() - 1; ++i)
        state[i] = x[x.size() - (h.size() - 1) + i];
}


