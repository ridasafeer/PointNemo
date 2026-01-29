
#include "fxlms.h"
#include "dsp.h"
#include "estimated_secondary_path.h"

class Controller {
public:
    // Constructor: creates FxLMS & DSP objects
    Controller(std::vector<float> shat, int L, float mu);

    //test for estimated secondary path via prbs
    std::vector<float> calibration();

   //Starts learning loop
    void startLearningLoop(float* referenceSignal, float* desiredSignal, int signalLength);
    //Where FXLMS will be used
    
    //Steady state behaviour - after learning, regular functionining of the ANC
    // Continues processing signals without updating filter coefficients
    void steadyStateProcessing(float* referenceSignal, float* desiredSignal, int signalLength);

    // Destructor: cleans up FxLMS & DSP objects
    ~Controller();

private:

    DSP dspObj;
    FxLMS fxlmsObj;
    std::vector<float> shat;
    //calibrate reference
};