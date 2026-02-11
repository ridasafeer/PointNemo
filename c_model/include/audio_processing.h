
#include <alsa/asoundlib.h> //alsa api
#include <iostream>
#include <vector>

#define HARDWARECONFIG {2, 1}
#define HARDWARECONFIGPATH "c_model/src/anc.conf"
//all other relevant configuration details for each struct is in hardware.conf

typedef struct {

    snd_pcm_format_t format;
    unsigned int rate;
    int periods;
    int period_size;

} streamParams;

typedef struct {

    int numDevices;
    int numSpeakers;
    int numRefMics;
    int numErrorMics;

    char** devices; //array of strings (char*) for each hw device in the hardware config
    //error mics and speakers must be paired
    streamParams sParams;

} hardwareConfig_t;

typedef struct pcmHandle {
    snd_pcm_t* handle;
    snd_pcm_stream_t direction; // CAPTURE or PLAYBACK
    unsigned int channels;
    snd_pcm_format_t format;
    char device_name[64];
    snd_pcm_status_t status; //current status of this pcm interface/line
    //the application-side buffer designateed for this channel
    int* buffer;
    snd_pcm_hw_params_t params;
    streamParams sParams;

} pcmHandle_t;

class AudioIO {

    public:
        AudioIO();
        int openInterfaces();
        int openInterface(pcmHandle_t* handle);
        int readReferenceSignal(float* buffer, int size); //blocking: 
        int closeInterface(pcmHandle_t* handle);
        int closeInterfaces();

    private: //just building for ref mic right now
        std::vector<pcmHandle_t*> handles;
        hardwareConfig_t hardwareConfig;

        //hardware configuration should only be within the class, not accessible by the user/outside this interface internally
        //only called within constructor
        hardwareConfig_t parseHardwareConfig(char* cfgFilePath);
        pcmHandle_t* initHardware(hardwareConfig_t hardwareConfig, std::vector<pcmHandle_t*>& pcmHandleList);

};