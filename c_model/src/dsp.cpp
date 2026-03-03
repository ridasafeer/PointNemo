//Class definition for DSP functions: Utility functions for digital signal processing 
// - Block processing functions for DSP operations inside FxLMS
// - DSP utility functions for computations: convolution, simple filtering

#include <vector>
#include <cmath>
#include <iostream>

#include "dsp.h"

#define PI 3.14159265358979323846

// Forward declaration
void fir_convolution(std::vector<float>& y, const std::vector<float>& x, const std::vector<float>& h, const std::vector<float>& state);

DSP::DSP() {}

void DSP::fir_block_processing(std::vector<float>& y, const std::vector<float>& x, const std::vector<float>& h, std::vector<float>& state)
{
    y.resize(x.size(),0.0f);

    if(h.size() == 0 || x.size()==0){
        state.clear();
        return;
    }


    //N = block size
    //M = FIR length
    //S = state length

    const size_t M = h.size();
    const size_t S = M - 1;

    if (state.size() != S)
    {
        state.assign(S, 0.0f);
    } 
    if (x.empty()) return;

    fir_convolution(y, x, h, state);

    // Update state : keep last S samples of (state + x)
    if (x.size() >= S) {
        // last S samples are inside x
        std::copy(x.end() - S, x.end(), state.begin());
    } else {
        // need some old state + all of x
        const size_t N = x.size();
        std::move(state.begin() + N, state.end(), state.begin());
        std::copy(x.begin(), x.end(), state.end() - N);
    }
    
}



// y-> ouput x-> input h->impulse
void fir_convolution(std::vector<float>& y,
                     const std::vector<float>& x,
                     const std::vector<float>& h,
                     const std::vector<float>& state)
{
    const int M = (int)h.size();
    const int S = M - 1;

    y.resize(x.size(), 0.0f);  // y[i] starts at 0, so we can accumulate into it

    for (int i = 0; i < (int)x.size(); ++i) {
        for (int j = 0; j < M; ++j) {
            int idx = i - j;
            if (idx >= 0) y[i] += h[j] * x[idx];
            else          y[i] += h[j] * state[idx + S];
        }
    }
}


// function to compute the impulse response "h" based on the sinc function
void DSP::impulseResponseLPF(float Fs, float Fc, unsigned short int num_taps, std::vector<float> &h, int gain)
{
    h.clear(); h.resize(num_taps, 0.0);
    float normCutoff = Fc / (Fs/2);

    for (int i = 0; i < num_taps; i++) {

        if (i == ((num_taps -1) / 2)) {
            h[i] = normCutoff;
        } else {
            h[i] = normCutoff * (sin(PI * normCutoff * (i - (num_taps - 1) / 2))) / (PI * normCutoff*(i - (num_taps - 1)/2));
        }
        h[i] = gain * h[i] * pow((sin((i * PI) / num_taps)), 2);
    }
}

// convultion with down sampling
void DSP::convolution_w_ds(std::vector<float> &h, std::vector<float> &block, std::vector<float> &state, std::vector<float> &sub_res, int ds) {

    unsigned int k, n;
    static int debug_block = 0;

	sub_res.clear(); sub_res.resize(block.size() / ds, 0.0);

	for (n = 0; n < block.size(); n+=ds) {
		for (k = 0; k < h.size(); k++) {
			if (((int) n- (int)k) >= 0) {
					sub_res[n/ds] += h[k] * block[n-k];
          if ((debug_block == -1) && (n == 10)) {
              std::cerr << "normal(" << n/ds << "," << k << "," << n-k << ")" << std::endl;
          }
			} else {
					int index = state.size() + ((int)n - (int)k);
					sub_res[n/ds] += h[k] * state[index];
          if ((debug_block == -1) && (n == 10)) {
            std::cerr << "state(" << n/ds << "," << k << "," << index << ")" << std::endl;
          }
			}
		}
	}

  state.clear(); state.resize(h.size()-1, 0.0);

  for (int i = 0; i < (int)state.size(); i++) {
    state[i] = block[i+(block.size()-h.size() + 1)];
    if (debug_block == -1) {
      std::cerr << "save(" << i << "," << i+(block.size()-h.size() + 1) << ")" << std::endl;
    }
  }

  debug_block += 1;
}

// band pass filter 
void DSP::bandPassCoeff(float fb, float fc, float fs, int num_taps, std::vector<float> &h) {

    float normCenter = ((fc + fb) / 2) / (fs / 2);
    float normPass = (fc-fb) / (fs/2);

    for (int i = 0; i < num_taps; i++) {
	if (i == (num_taps - 1) / 2) {
	    h[i] = normPass;
	} else {
	    h[i]= normPass * (sin(PI*(normPass/2)*(i-(num_taps-1)/2))) / (PI * (normPass/2)*(i-(num_taps-1)/2));
	}
	h[i] = h[i] * cos(i * PI * normCenter);
	h[i] = h[i] * pow(sin((i*PI)/num_taps),2);
    }
}


// convolution with down and up sampling
void DSP::resampling(std::vector<float>& y, const std::vector<float>& x, const std::vector<float>& h, std::vector<float>&state, int ds, int us){

	y.clear(); y.resize(x.size()* us / ds, 0.0);
    int k, n;
    int h_size = (int) h.size();
    // initialize state to zeros if first call or wrong size
    if ((int)state.size() != h_size)
        state.assign(h_size, 0.0f);
    int state_size = (int) state.size();
    int y_size = (int) y.size();

    for (n = 0; n < y_size; n++) {

        for (k = (n*ds) % us; k < h_size; k+=us) {

			int i = (n*ds-k)/us;

             if (i >= 0) {
                     y[n] += h[k] * x[i];

            }
             else {
                int index = state_size + i;
                y [n] += h[k] * state[index];
            }
	        }
    }

    state.clear(); state.resize(h.size(), 0.0);

    state.assign(x.end() - state.size(), x.end());

}

void DSP::dot_product(const std::vector<float>& a, const std::vector<float>& b, float& result)
{
    result = 0.0f;
    for (int i = 0; i < (int)a.size(); i++) {
        result += a[i] * b[i];
    }
}

