//Controller parent class: Owns FXLMS and DSP objects, controls flow of FXLMS
//Starts learning loop
//Initializes FxLMS and DSP objects - this should be done when the prorgam begins
//Learning loop ends here as well

#include "controller.h"

Controller::Controller(std::vector<float> w, int L, float mu) : dspObj(nullptr), shat(calibration()) {
    // Initialize parameters for the controller class below
    // Before constructing fxlms object, we need to call calibrate() to identify estimated secondary path s_hat
    shat = calibration(); //output of calibration: shat
    fxlmsObj = new FxLMS(shat, L, mu);

}

std::vector<float> Controller::calibration() {
    // Placeholder for actual calibration logic
    return std::vector<float>();
}

void Controller::startLearningLoop(const std::vector<float>& x, float* desiredSignal, int signalLength) {
    //
    
}
    

