
#include "fxlms.h"
#include "dsp.h"
#include "estimated_secondary_path.h"
#include "audio_processing.h"
#include <vector>

#pragma once

class Controller {
public:
    // Constructor: creates FxLMS & DSP objects
    Controller(std::vector<float> shat, int L, float mu);

    //test for estimated secondary path via prbs
    std::vector<float> calibration(const std::vector<float>& x_exc, const std::vector<float>& y_mic, int L, float mu, int passes = 1, float leak = 0.0f);

   //Starts learning loop
    void startLearningLoop(float* referenceSignal, float* desiredSignal, int signalLength);
    //Where FXLMS will be used

    //functions for interfacing withe audio_proc and the fxlms
    void pushReferenceSignal();

    int writeAntinoiseSignal();


private:

    DSP dspObj;
    FxLMS fxlmsObj;
    AudioIO audioProcObj;
    std::vector<float> &shat;
    std::vector<float> &x;
    std::vector<float> &y;
    float errorSignal;
    
    int head = 0; //oldest
    int tail = 0;
    //calibrate reference

};