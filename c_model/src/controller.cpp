//Controller parent class: Owns FXLMS and DSP objects, controls flow of FXLMS
//Starts learning loop
//Initializes FxLMS and DSP objects - this should be done when the prorgam begins
//Learning loop ends here as well

#include "controller.h"
#include <stdexcept>

Controller::Controller(std::vector<float> shat, int L, float mu) : dspObj(L), shat(shat), fxlmsObj(shat, L, mu), audioProcObj(), x(fxlmsObj.getXbuf()), y(fxlmsObj.getYbuf()) {
    std::cout << "inside controller constructor" << std::endl;
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

 std::vector<float> Controller::pushReferenceSignal() { 
    //receive the refrence signal new values buffer from the audio_proc
    std::vector<float> inputBuffer = audioProcObj.readReferenceSignal(); //256 samples
    std::cout << "\ninputBuffer size : " << inputBuffer.size() << std::endl;
    std::cout << "\nsize of fxlms x buf check : " << x.size() << std::endl;
    return inputBuffer;
}

void Controller::writeAntinoiseSample(float yn_val) {

    //use internal y buffer and pass to audioProc
    static std::vector<float> y_alsa_temp; //make static
    y_alsa_temp.push_back(yn_val);
    if ((int)y_alsa_temp.size() >= (int)audioProcObj.getPeriodSize()) { //unhardcoded
        audioProcObj.writeAntinoiseSignal(y_alsa_temp);
        y_alsa_temp.clear();
    }

}

void Controller::startLearningLoop() {
    
    //Manages the entire control flow of the FxLMS algorithm, links input and output buffers, and identifies termination

    //Mnagement of the batch gradient learning
    while (1) {
        //update the reference signal
        //pushReferenceSignal(); //256 chunk of samples

        std::vector<float> refSigChunk = pushReferenceSignal();
        std::vector<float> antinoiseSigChunk;
        int num_taps = fxlmsObj.getNumTaps();

        for (int i = 0; i < refSigChunk.size(); i++) {
            //UPDATE X(N) SLIDING WINDOW: shift each value into the circular buffer
                //sliding window logic 1: on the pushing into the buffer side
                //num_taps: size of the window, matching the size of the filter impulse response
                //audio buffer size: alll the new samples to place in window
            tail = (tail+1) % num_taps; //move tail to sample's new slot
            x[tail] = refSigChunk[i];
            printf("%x\t", refSigChunk[i]);

            // CONVOLUTION 1: 101 taps
            float yn_val = fxlmsObj.output_test(tail);

            //OUTPUT SIGNAL CIRCULAR BUFFER: place at current tail
            //y[ytail] = yn_val;

            // //PATH 1: send the output signal to the speakers, going through the real S(z) in the DSP/physical env as it travels to the error mic
            writeAntinoiseSample(yn_val);

            //ytail = (ytail+1) % y.size();

            //PATH 2: LMS update
            //compute the xf filtered signal before the update
        fxlmsObj.push_xf_learning();
        }

        // read error mic after chunk COMMENTED OUT CUZ ERROR MIC NOT CONNECTED
        // std::vector<float> errorChunk = audioProcObj.readErrorSignal();
        // for (int i = 0; i < (int)errorChunk.size(); i++) {
        //     fxlmsObj.update(errorChunk[i]);
        // }
    }
}
    

