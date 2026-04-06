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

    w(L, 0.0f), 
    x(*(new std::vector<float>(L, 0.0f))), 
    xf(L, 0.0f),
    y(1, 0.0f) {
    std::cout << "FxLMS constructor" << std::endl;
}

//-----------------------------
// output
// Computes y(n) = w^T . x_circular
 
//produce the anti-noise signal y(n), propagate it forward for speaker output (will travel through S(z) physically)
//Convolution of filter coefficients with reference signal

void FxLMS::output(int startIndex) {
    head = startIndex;
    float yn_val;

    //takes index of ciruclar buffer to lenght of adpative filter (# of coefficients)
    for (int i=0; i<L; i++) {
        int index = (head - i + L) % L;
        yn_val += w[i] * x[index]; //convolution
    }
    y[0] = yn_val;
    //sliding window logic 2: on the reading for computing each convolution product side
}

//online convolution: single-sample convolution with both circular buffers
float FxLMS::output_test(int startIndex) {
    float yn_val = 0.0f;
    int index = startIndex;
    for (int i = 0; i < w.size(); i++) {
        yn_val += w[i] * x[index];
        index = (index + L - 1) % L; // move backwards through circular buffer to read recent history
    }
    return yn_val;
}

//-----------------------------
// filtered_x_sample()
// Computes sample of filtered reference signal after trasnfer fnc
 
// x'(n) = shat^T . [x(n)] 
//Convolution of reference signal with est second paath

float FxLMS::filtered_x_sample(int startIndex) {
        std::cout << "FxLMS.cpp: filtered_x_sample()" << std::endl;
        float xn_filtered = 0.0;
        int index = startIndex;
    for (int i = 0; i < L; i++) {
        xn_filtered += shat[i] * x[index];
        int index = (index - 1 + L) % L;
    }
    return xn_filtered;
}

//-----------------------------
// update w coeffs: stochastic gradient descent
// slightly adjust those weights for the next antinoise output

//e_n - error signal
//w[i] - coeffs of adaptive filter
//mu - learning rate
//xf - reference signal post transfer function

void FxLMS::update(float e_n){
    std::cout << "FxLMS.cpp: update()" << std::endl;

    const float mu_e = mu * e_n;   // scalar: pre-multiply once outside the loop

    for (int i = 0; i < L; i++) {
        w[i] += mu_e * xf[i];  // standard FxLMS 
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