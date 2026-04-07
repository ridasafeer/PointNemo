//Class definition of FXLMS algorithm

#include "fxlms.h"
#include <stdio.h>
#include <iostream>
#include <complex>

// Constructor
FxLMS::FxLMS(const std::vector<float>& shat, int L, float mu)
    : L(L), 
    M(static_cast<int>(shat.size())), 
    mu(mu), 
    shat(shat), 
    x_tail(-1),
    xf_tail(-1),
    y_tail(-1),

    w(L, 0.0f), 
    x(L, 0.0f), 
    xf(L, 0.0f),
    y(1, 0.0f) {
    std::cout << "FxLMS constructor" << std::endl;
}

const std::vector<float>& FxLMS::getWeights() const {
    return w;
}

//-----------------------------
// output
// Computes y(n) = w^T . x_circular
 
//produce the anti-noise signal y(n), propagate it forward for speaker output (will travel through S(z) physically)
//Convolution of filter coefficients with reference signal

void FxLMS::push_reference_sample(float curr_sample) {
    x_tail = (x_tail+1) % L; //move tail to sample's new slot
    x[x_tail] = curr_sample;
    printf("New x[n] sample: %.4f\t", curr_sample);
}

//online convolution: single-sample convolution with both circular buffers
float FxLMS::output() {
    
    y_tail = (y_tail+1) % y.size(); //move to next spot for current value to be placed in
    y[y_tail] = 0.0f; //reset that value to 0

    int index = x_tail;
    for (int i = 0; i < w.size(); i++) {
        y[y_tail] += w[i] * x[index];
        index = (index + L - 1) % L; // move backwards through circular buffer to read recent history
    }
    return y[y_tail];
}

//-----------------------------
// filtered_x_sample()
// Computes sample of filtered reference signal after trasnfer fnc
 
// x'(n) = shat^T . [x(n)] 
//Convolution of reference signal with est second paath

float FxLMS::filtered_x_sample() {

    xf_tail = (xf_tail + 1) % L;
    xf[xf_tail] = 0.0f;

    int index = x_tail;
    for (int i = 0; i < M; i++) {
        xf[xf_tail] += shat[i] * x[index];
        index = (index - 1 + L) % L;
    }

    printf("Corresponding xf sample computed : %.4f\t", xf[xf_tail]);
    return xf[xf_tail];
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

std::vector<float> FxLMS::getFilteredReferenceHistory() const {
    std::vector<float> hist(L, 0.0f);

    for (int k = 0; k < L; k++) {
        int idx = (xf_tail - k + L) % L;
        hist[k] = xf[idx];
    }

    return hist;
}

void FxLMS::update_aligned_sample(float e_n, const std::vector<float>& x_filt_hist) {
    const float mu_e = mu * e_n;

    for (int k = 0; k < L; k++) {
        w[k] += mu_e * x_filt_hist[k];
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