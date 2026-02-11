//
#include <vector>
#include <cmath>
#include "audio_processing.h"


int main ()
{
    // Initialize audio processing i/o 

    //Calibration

    //Initialize hardware for audio i/o: speakers and mic set-up, open pcm interfaces
    AudioIO audioIO = new AudioIO();
        //call constructor: therefore, completes ahrdware configuration
        //then, creates all the necessary pcm handles

    //open all interfaces
    audioIO.openInterfaces();

    //initalize learning loop, FxLMS algorithm, etc.


    // main loop: CURRENTLY BLOCKING, 1 THREAD

    while (1) {

        //receive reference mic input x(n) - blocking call until buffer full on audio i/o side
        audioIO.readReferenceSignal(); //reads 1 period of the buffer, which is number of frames wanted to read

        //learning loop iteration

        //output the anti-noise signal to the speaker interface - blocking call until buffer full on audio i/o side

    }

    return 0;
}



