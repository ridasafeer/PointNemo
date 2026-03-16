//Controller parent class: Owns FXLMS and DSP objects, controls flow of FXLMS
//Starts learning loop
//Initializes FxLMS and DSP objects - this should be done when the prorgam begins
//Learning loop ends here as well

#include "controller.h"
#include <stdexcept>

Controller::Controller(std::vector<float> shat, int L, float mu) : dspObj(), shat(shat), fxlmsObj(shat, L, mu), audioProcObj(), x(fxlmsObj.getXbuf()), y(fxlmsObj.getYbuf()) {
    // Initialize parameters for the controller class below
    // Before constructing fxlms object, we need to call calibrate() to identify estimated secondary path s_hat
    //FxLMS, AudioIO, and DSP objects already instantiated in the initializer list constructor syntax

    //link the input x to the controller's x

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


    std::vector<float> shat(L, 0.0f); //fill vector with num of fir coefficients
    std::vector<float> xhist(L, 0.0f); //vec holding most recent L number of x input excitations, aka hvac noise samples with vector size L. x_hist[0] is newest

    for (int p = 0; p < passes; ++p) {                  // more passes helps system converge
        std::fill(xhist.begin(), xhist.end(), 0.0f);    // clear input history, and keep h vector across each passes. for each pass h gets better/more accurate
        for (size_t n = 0; n < x_exc.size(); ++n) {
        
        // shift in new excitation sample. shift old samples “down” one position. Ex: xhist[1] becomes previous xhist[0]
        for (int i = L - 1; i > 0; --i) xhist[i] = xhist[i - 1];
        xhist[0] = x_exc[n]; //insert new excitation sample at front

        float yhat = 0.0f;
        // model output computed from the current h (fir coeff) at time n (where n = 0 to n = size of x_exc).

        for (int k = 0; k < L; ++k) yhat += shat[k] * xhist[k];     //Computes convolution for this time step, where dot product shat · xhist
        // yhat(n) = Σ h[k] x(n-k)


        // error
        // if yhat is too small, e is positive and if too large, e is negative
        // LMS uses this to adjust h to reduce future error
        float e = y_mic[n] - yhat;

        // leakage factor (helps slow drift / keeps taps bounded)
        if (leak > 0.0f) {
            float keep = 1.0f - leak; // how much of the old coefficient value to keep
            for (int k = 0; k < L; ++k) shat[k] *= keep; //Shrinks coefficients slightly every sample, helps if data is noisy
        }
        
        // update LMS
        float g = mu * e; // step scalar for this sample
        for (int k = 0; k < L; ++k) shat[k] += g * xhist[k]; 

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
    return shat;
}

void Controller::pushReferenceSignal() { 
    //receive the refrence signal new values buffer from the audio_proc
    std::vector<float> inputBuffer = audioProcObj.readReferenceSignal();
    int num_taps = fxlmsObj.getNumTaps();
    //add to the reference signal's sliding window via x: sie of lliding window is equal to num_taps
    //therfore, x should be a circular buffer: the oldest value is overwritten
    //therfore, input the values into the x buffer of fxlms using circular 
    for (int i = 0; i < inputBuffer.size(); i++) {
        //shift each value into the circular buffer, 
        x[tail] = inputBuffer[i];
        tail = tail+1 % num_taps;
    }

}

int Controller::writeAntinoiseSignal() {
    return 0;
}

void Controller::startLearningLoop(float* referenceSignal, float* desiredSignal, int signalLength) {
    
    //Manages the entire control flow of the FxLMS algorithm, links input and output buffers, and identifies termination

    //Mnagement of the batch gradient learning
    while (1) {
        //update the reference signal
        audioProcObj.readReferenceSignal(); //controller is arleady bound to the specific dsp and fxlms instance
        //compute antinoise
        fxlmsObj.output();

        //PATH 1: send the output signal to the speakers, going through the real S(z) in the DSP/physical env as it travels to the error mic
        //Write to the main user anti-noise speaker
        writeAntinoiseSignal();

        //PATH 2: LMS update

        //compute the xf filtered signal before the update
        fxlmsObj.push_xf(); //xf is internal to fxlms obj

        //weight update using the xf
        fxlmsObj.update();

        //Measure the sound seen by the error mic (right beside the main user speaker)
        int test = audioProcObj.readErrorSignal();

    }
    
    
}
    

