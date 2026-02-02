//Controller parent class: Owns FXLMS and DSP objects, controls flow of FXLMS
//Starts learning loop
//Initializes FxLMS and DSP objects - this should be done when the prorgam begins
//Learning loop ends here as well

#include "controller.h"

Controller::Controller(std::vector<float> w, int L, float mu) : dspObj(), shat(calibration()), fxlmsObj(calibration(), L, mu) {
    // Initialize parameters for the controller class below
    // Before constructing fxlms object, we need to call calibrate() to identify estimated secondary path s_hat

}

std::vector<float> Controller::calibration() {
    // Placeholder for actual calibration logic
    return std::vector<float>();
}

void Controller::startLearningLoop(float* referenceSignal, float* desiredSignal, int signalLength) {
    //
    
}
    

