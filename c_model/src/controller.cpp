//Controller parent class: Owns FXLMS and DSP objects, controls flow of FXLMS
//Starts learning loop
//Initializes FxLMS and DSP objects - this should be done when the prorgam begins
//Learning loop ends here as well

#include "controller.h"
//include "signal_testing.h"
#include <stdexcept>

Controller::Controller(std::vector<float> shat, int L, float mu) : dspObj(L), shat(shat), fxlmsObj(shat, L, mu), audioProcObj(), x(fxlmsObj.getXbuf()), y(fxlmsObj.getYbuf()), signalTester(48000.0f) {
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

 std::vector<float> Controller::readReferenceSignal() { 
    //receive the refrence signal new values buffer from the audio_proc
    std::vector<float> inputBuffer = audioProcObj.readReferenceSignal(); //256 samples
    std::cout << "\ninputBuffer size : " << inputBuffer.size() << std::endl;
    std::cout << "\nsize of fxlms x buf check : " << x.size() << std::endl;
    return inputBuffer;
}

void Controller::writeAntinoiseSample(std::vector<float> antinoiseBlock) {

    audioProcObj.writeAntinoiseSignal(antinoiseBlock); 

}

void Controller::pushReferenceSample(float refSigSample) {
    //push new sample into the x buffer, which is used for the convolution and update
    fxlmsObj.push_reference_sample(refSigSample);  
}

float Controller::computeAntinoiseSample(int i) {
    // CONVOLUTION 1: 101 taps
    float yn_val = fxlmsObj.output();
    printf("Current iteration %d: %.4f\t", i, yn_val);
    return yn_val;
}

void Controller::startLearningLoop() {

    int analysisCounter = 0;

    while (1) {

        // reading ref chunk
        std::vector<float> refSigChunk = readReferenceSignal();
        std::cout << "\ninputBuffer size : " << refSigChunk.size() << std::endl;

        // some useful buffers
        std::vector<float> antinoiseBlock;
        std::vector<std::vector<float>> xfHistBlock;

        antinoiseBlock.reserve(refSigChunk.size());
        xfHistBlock.reserve(refSigChunk.size());

        // sample by sample processing of input chunk
        for (int i = 0; i < (int)refSigChunk.size(); i++) {

            pushReferenceSample(refSigChunk[i]);

            float yn = computeAntinoiseSample(i);
            antinoiseBlock.push_back(yn);

            // compute filtered reference sample
            fxlmsObj.filtered_x_sample();

            // store FULL xf history for THIS sample
            xfHistBlock.push_back(fxlmsObj.getFilteredReferenceHistory());
        }

        if (analysisCounter % 10 == 0) {
            try {
                std::cout << "\n[SignalTesting] Reference block stats:"
                          << " mean=" << signalTester.computeMean(refSigChunk)
                          << " rms="  << signalTester.computeRMS(refSigChunk)
                          << " peak=" << signalTester.computePeak(refSigChunk)
                          << std::endl;

                std::cout << "[SignalTesting] Antinoise block stats:"
                          << " mean=" << signalTester.computeMean(antinoiseBlock)
                          << " rms="  << signalTester.computeRMS(antinoiseBlock)
                          << " peak=" << signalTester.computePeak(antinoiseBlock)
                          << std::endl;

                // frequency responses
                auto refSpec = signalTester.computeSpectrum(refSigChunk, true);
                signalTester.plotSpectrum(refSpec, "Reference Block Spectrum");

                auto antiSpec = signalTester.computeSpectrum(antinoiseBlock, true);
                signalTester.plotSpectrum(antiSpec, "Antinoise Block Spectrum");

                // current adaptive FIR response: gonna compare the reference and antinoise in the laplace dom to show mag and phase shift between ref to antinoise
                auto filtResp = signalTester.computeFIRFrequencyResponse(fxlmsObj.getWeights());
                signalTester.plotTransfer(filtResp, "Adaptive Filter Frequency Response");
            }
            catch (const std::exception& e) {
                std::cerr << "[SignalTesting] Pre-write analysis failed: " << e.what() << std::endl;
            }
        }

        // after all antinoise samples pushed, push full antinoise block
        audioProcObj.writeAntinoiseSignal(antinoiseBlock);

        // ===== 5. Store xf history block =====
        xf_hist_blocks.push_back(xfHistBlock);

        // ===== 6. Read error block ONCE =====
        std::vector<float> errSigChunk = audioProcObj.readErrorSignal();

        // ===== 6A. Error-side testing =====
        if (analysisCounter % 10 == 0) {
            try {
                std::cout << "[SignalTesting] Error block stats:"
                          << " mean=" << signalTester.computeMean(errSigChunk)
                          << " rms="  << signalTester.computeRMS(errSigChunk)
                          << " peak=" << signalTester.computePeak(errSigChunk)
                          << std::endl;

                // 4. Error spectrum
                auto errSpec = signalTester.computeSpectrum(errSigChunk, true);
                signalTester.plotSpectrum(errSpec, "Error Block Spectrum");

                // 5. Coherence between current reference block and current error block
                // Note: for true physical alignment this is approximate unless the delayBlocks relationship
                // is accounted for perfectly, but still useful for early debugging.
                if (refSigChunk.size() == errSigChunk.size()) {
                    auto coh = signalTester.estimateCoherence(refSigChunk, errSigChunk, true);
                    signalTester.plotCoherence(coh, "Reference-to-Error Coherence");
                }

                // 6. Measured transfer from antinoise output block -> error block
                if (antinoiseBlock.size() == errSigChunk.size()) {
                    auto outToErr = signalTester.estimateTransferFunction(
                        antinoiseBlock, errSigChunk, true);
                    signalTester.plotTransfer(outToErr, "Output-to-Error Transfer Function");
                }
            }
            catch (const std::exception& e) {
                std::cerr << "[SignalTesting] Error-side analysis failed: " << e.what() << std::endl;
            }
        }

        // delayed update: pop oldest xf block, and use that with the current error block to do the FxLMS update for each sample
        if ((int)xf_hist_blocks.size() > delayBlocks) {

            std::vector<std::vector<float>> alignedXfBlock = xf_hist_blocks.front();
            xf_hist_blocks.pop_front();

            int N = std::min((int)errSigChunk.size(), (int)alignedXfBlock.size());

            for (int i = 0; i < N; i++) {
                fxlmsObj.update_aligned_sample(errSigChunk[i], alignedXfBlock[i]);
            }
        }

        analysisCounter++;

        //break; // run one chunk for testing
    }
}