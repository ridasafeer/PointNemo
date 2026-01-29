
#include "fxlms.h"
#include "dsp.h"
#include "estimated_secondary_path.h"

class Controller {
public:
    // Constructor: creates FxLMS & DSP objects
    Controller();

    //test for estimated secondary path via prbs
    void calibration();

   //Starts learning loop
    void startLearningLoop(float* referenceSignal, float* desiredSignal, int signalLength);
    //Where FXLMS will be used
    
    //Steady state behaviour - after learning, regular functionining of the ANC
    // Continues processing signals without updating filter coefficients
    void steadyStateProcessing(float* referenceSignal, float* desiredSignal, int signalLength);

    // Destructor: cleans up FxLMS & DSP objects
    ~Controller();

private:
    int L;                  // Adaptive filter length
    int M;                  // Secondary path length
    float mu;               // Step size

    DSP* dspObj;
    FxLMS* fxlmsObj;
    //calibrate reference
};