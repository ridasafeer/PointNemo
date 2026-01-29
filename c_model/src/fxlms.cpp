//Class definition of FXLMS algorithm


#include "fxlms.h"

// Constructor
FxLMS::FxLMS(int L, const std::vector<float>& shat, float mu)
    : L(L), M(static_cast<int>(shat.size())), mu(mu), shat(shat), w(L, 0.0f), xbuf(L, 0.0f), xfbuf(L, 0.0f) {
    //TODO: should also have a dsp reference
    }

void FxLMS::push_x(float x){

 }

float FxLMS::output() const {
}

float FxLMS::filtered_x_sample() const {

}

void FxLMS::push_xf(float xf) {

}


void FxLMS::update(float e){

}
