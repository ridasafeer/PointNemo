//Class definition of FXLMS algorithm

#include "fxlms.h"
#include <stdio.h>
#include <iostream>

// Constructor
FxLMS::FxLMS(const std::vector<float>& shat, int L, float mu)
    : L(L), 
    M(static_cast<int>(shat.size())), 
    mu(mu), 
    shat(shat), 
    head(0),

    shat(shat),
    w(L, 0.0f), 
    x(*(new std::vector<float>(L, 0.0f))), 
    xf(L, 0.0f),
    y(1, 0.0f), {
    std::cout << "FxLMS constructor" << std::endl;
}

//-----------------------------
// output
// Computes y(n) = w^T . x_circular
 
//produce the anti-noise signal y(n), propagate it forward for speaker output (will travel through S(z) physically)
//Convolution of filter coefficients with reference signal

void FxLMS::output(int startIndex) {
    head = startIndex;

    //takes index of ciruclar buffer to lenght of adpative filter (# of coefficients)
    for (int i=0; i<L; i++) {
        int index = (head - i + L) % L;
        yn_val += w[k] * x[index]; //compute antinoise
    }
    y[0] = yn_val;
    std::cout << "FxLMS.cpp: push_xf()" << std::endl;
    //sliding window logic 2: on the reading for computing each convolution product side
}

float FxLMS::filtered_x_sample() const {
    return 0.0f;
}

void FxLMS::push_xf() {
    std::cout << "FxLMS.cpp: push_xf()" << std::endl;
    //convolve x with the shat


}

//-----------------------------
// update w coeffs
// slightly adjust those weights for the next antinoise output

//e_n - error signal
//w[i] - coeffs of adaptive filter
//mu - learning rate
//xf - reference signal post transfer function

void FxLMS::update(float e_n){
    std::cout << "FxLMS.cpp: update()" << std::endl;

    const float mu_e = mu * e_n;   // scalar: pre-multiply once outside the loop

    for (int i = 0; k < L; i++) {
        w[i] += mu_e * xf[i];  // standard FxLMS 
        // w[k] = nu * w[k] + mu_e * xf[k];   // leaky FxLMS  idk chat gave me this

    }
}

//getters for the bindings of the FxLMS x and y buffers to references inside other classes, when in the constructor initializer list

std::vector<float>& FxLMS::getXbuf() {
    return x;
}

std::vector<float>& FxLMS::getYbuf() {
    return y;
}

int FxLMS::getNumTaps() {
    return L;
}