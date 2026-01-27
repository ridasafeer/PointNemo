// hardware audio processing interface
// add audio processing from mics 
// tiny alsa or other library to capture audio input and output

// speaker interface 

// 1) process audio input from mics both ref and error microphones
// 2) send processed audio to speaker output

// will go directly to dsp to be used there 


#include "audio_processing.h"


struct auidoio_interface {
    // audio input/output parameters
    int sample_rate;
    int buffer_size;
    // Add other necessary members for audio handling
};

// Initialize audio processing interface




