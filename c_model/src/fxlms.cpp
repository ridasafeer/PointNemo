//Class definition of FXLMS algorithm


#include "fxlms.h"

// Constructor
FxLMS::FxLMS(const std::vector<float>& shat, int L, float mu)
    : L(L), M(static_cast<int>(shat.size())), mu(mu), shat(shat), w(L, 0.0f), x(*(new std::vector<float>(L, 0.0f))), xf(L, 0.0f) {
    //TODO: should also have a dsp reference

    //initialize all buffers: buffers should be internal to the fxlms class, expcet s_hat
    //init w
    //receive x_ref - this should be sample bby sample

    }

//sample by sample
void FxLMS::push_x(const std::vector<float>& xbuf){

 }

float FxLMS::output() const {
    return 0.0f;
}

float FxLMS::filtered_x_sample() const {
    return 0.0f;
}

void FxLMS::push_xf(float xf) {

}


void FxLMS::update(float e){

}