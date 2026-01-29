
#include "fxlms.h"
#include "dsp.h"

class Controller {
public:
    // Constructor: creates FxLMS & DSP objects
    Controller();

private:
    int L;                  // Adaptive filter length
    int M;                  // Secondary path length
    float mu;               // Step size

    DSP* dspObj;
    FxLMS* fxlmsObj;
};