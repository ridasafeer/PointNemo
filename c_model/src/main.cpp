//
#include <vector>
#include <cmath>


int main ()
{
    // Initialize audio processing i/o 

    //Calibration

    //Initialize hardware for audio i/o: speakers and mic set-up, open pcm interfaces

    //initalize learning loop, FxLMS algorithm, etc.


    // main loop: CURRENTLY BLOCKING, 1 THREAD

    while (1) {

        //receive reference mic input x(n) - blocking call until buffer full on audio i/o side

        //learning loop iteration

        //output the anti-noise signal to the speaker interface - blocking call until buffer full on audio i/o side

    }

    return 0;
}



