//Controller parent class: Owns FXLMS and DSP objects, controls flow of FXLMS
//Starts learning loop
//Initializes FxLMS and DSP objects - this should be done when the prorgam begins
//Learning loop ends here as well

#include "controller.h"
#include <stdexcept>

Controller::Controller(std::vector<float> shat, int L, float mu) : dspObj(), shat(shat), fxlmsObj(shat, L, mu) {
    // Initialize parameters for the controller class below
    // Before constructing fxlms object, we need to call calibrate() to identify estimated secondary path s_hat

}

std::vector<float> Controller::calibration(
    const std::vector<float>& x_exc, // HVAC signal sent to ONE speaker
    const std::vector<float>& y_mic, // recorded mic sample, make sure its the same length as x_exec
    int L,                           // num taps
    float mu,                        // step size
    int passes,                      // factor of how many times to repeat the data set
    float leak)                      // optional leakage factor for 2ndary path
{

    // shit ton of error catching

    if (L <= 0) throw std::invalid_argument("L must be > 0");
    if (x_exc.size() != y_mic.size()) throw std::invalid_argument("x_exc and y_mic must be same length");
    if (x_exc.empty()) throw std::invalid_argument("signals are empty");
    if (mu <= 0.0f) throw std::invalid_argument("mu must be > 0");
    if (passes <= 0) passes = 1;
    if (leak < 0.0f || leak >= 1.0f) throw std::invalid_argument("leak must be in 0 to 1 range 0 <= leak < 1");


    // Initialize persistent state on first call or if L changed
    if ((int)cal_shat.size() != L) {
        cal_shat.assign(L, 0.0f);
        cal_xhist.assign(L, 0.0f);
    }

    for (int p = 0; p < passes; ++p) {                  // more passes helps system converge
        std::fill(cal_xhist.begin(), cal_xhist.end(), 0.0f);    // clear input history each pass, keep taps across passes and calls
        for (size_t n = 0; n < x_exc.size(); ++n) {
        
        // shift in new excitation sample. shift old samples “down” one position. Ex: xhist[1] becomes previous xhist[0]
        for (int i = L - 1; i > 0; --i) cal_xhist[i] = cal_xhist[i - 1];
        cal_xhist[0] = x_exc[n]; //insert new excitation sample at front

        float yhat = 0.0f;
        // model output computed from the current h (fir coeff) at time n (where n = 0 to n = size of x_exc).

        dspObj.dot_product(cal_shat, cal_xhist, yhat);     //Computes convolution for this time step, where dot product cal_shat * cal_xhist
        // yhat(n) = h[k] x(n-k)


        // error
        // if yhat is too small, e is positive and if too large, e is negative
        // LMS uses this to adjust h to reduce future error
        float e = y_mic[n] - yhat;

        // leakage factor (helps slow drift / keeps taps bounded)
        if (leak > 0.0f) {
            float keep = 1.0f - leak; // how much of the old coefficient value to keep
            for (int k = 0; k < L; ++k) cal_shat[k] *= keep; //Shrinks coefficients slightly every sample, helps if data is noisy
        }
        
        // update LMS
        float g = mu * e; // step scalar for this sample
        for (int k = 0; k < L; ++k) cal_shat[k] += g * cal_xhist[k]; 

        //If error e is positive and xhist[k] is positive, h[k] increases
        // If error e is negative, the update goes the opposite way

        //BIG POINT BIG FAT IDEA --> these updates push h toward a set of taps that make yhat match y_mic

        //ELI5 EXAMPLE FROM CHAT FOR DUMBAHH IF UR DUMBAHH
        /*
            h = the recipe (ingredients list).
            x = the ingredients you put in right now (current + past samples).
            yhat = the cake you predict you’ll get from that recipe right now.
            y (your mic recording) = the cake you actually got.
            The difference e = y − yhat tells you how to tweak the recipe h.
        */
    }
    }
    return cal_shat;
}

void Controller::startLearningLoop(float* referenceSignal, float* desiredSignal, int signalLength) {
    //
    
}

Controller::~Controller() {}
    

