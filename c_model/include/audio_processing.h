
#include <alsa/asoundlib.h> //alsa api
#include <iostream>
#include <vector>

#define HARDWARECONFIG {2, 1}
//all other relevant configuration details for each struct is in hardware.conf


typedef struct pcmHandle {
    snd_pcm_t* handle;
    snd_pcm_stream_t direction; // CAPTURE or PLAYBACK
    unsigned int channels;
    unsigned int rate;
    snd_pcm_format_t format;
    char device_name[64]
    snd_pcm_status_t status; //current status of this pcm interface/line

} pcmHandle_t;

class AudioIO {

    public:
        AudioIO();
        pcmHandle_t* initHardware();
        int openInterfaces();
        int openInterface(pcmHandle_t* handle);
        int receive_audio_input_ref(float* buffer, int size); //blocking: 
        int closeInterface(pcmHandle_t* handle);
        int closeInterfaces();

    private: //just building for ref mic right now
        pcmHandle_t refMicHandle; //reference mic input
        int* refMicBuffer; //user-side/application-side buffer for ref mic input

};