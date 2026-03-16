//Class definition of FXLMS algorithm

#include "fxlms.h"
#include <stdio.h>

// Constructor
FxLMS::FxLMS(const std::vector<float>& shat, int L, float mu)
    : L(L), M(static_cast<int>(shat.size())), mu(mu), shat(shat), w(L, 0.0f), x(*(new std::vector<float>(L, 0.0f))), xf(L, 0.0f) {


}

void FxLMS::output() const {
    //produce the anti-noise signal y(n), propagate it forward for speaker output (will travel through S(z) physically)
    //Convolution of filter coefficients with reference signal
    std::cout << "FxLMS.cpp: push_xf()" << std::endl;

}

float FxLMS::filtered_x_sample() const {
    return 0.0f;
}

void FxLMS::push_xf() {
    std::cout << "FxLMS.cpp: push_xf()" << std::endl;
    //convolve x with the shat


}


void FxLMS::update(float e){
    std::cout << "FxLMS.cpp: update()" << std::endl;

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