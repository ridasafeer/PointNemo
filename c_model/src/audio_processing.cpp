
#include "audio_processing.h"

// Initialize audio processing interface

AudioIO::AudioIO() {
    //constructor

    self.refMicHandle = *initHardware();
    self.refMicBuffer = new int[self.refMicHandle.periodSize]; //allocate user-side buffer for ref mic input, size of 1 period

}

pcmHandle_t* AudioIO::initHardware() {
    //
    
}

void AudioIO::receive_audio_input_ref() {

    //blocking read: reads until buffer of size periodSize is full, then returns number of frames read (should be periodSize unless error)
    rc = snd_pcm_readi(self.refMicHandle.handle, self.refMicBuffer, self.refMicHandle.periodSize);
    printf(); //number of frames read
    //


}


