
#include <alsa/asoundlib.h> //alsa api
#include <iostream>
#include <vector>

#pragma once

#define HARDWARECONFIG {2, 1}
#define HARDWARECONFIGPATH "anc.ini" //const char*
//all other relevant configuration details for each struct is in hardware.conf

typedef struct {

    snd_pcm_format_t format;
    unsigned int rate;
    int periods;
    snd_pcm_uframes_t period_size;

} streamParams;

typedef struct {

    int numDevices;
    int numSpeakers;
    int numRefMics;
    int numErrorMics;

    const char** devices; //array of strings (char*) for each hw device in the hardware config
    //error mics and speakers must be paired
    streamParams sParams; //the sParams obj that will be assumed for each handle

} hardwareConfig_t;

typedef struct pcmHandle {
    snd_pcm_t* handle;
    snd_pcm_stream_t direction; // CAPTURE or PLAYBACK
    unsigned int channels;
    snd_pcm_format_t format;
    const char* device_name;
    snd_pcm_status_t* status; //current status of this pcm interface/line
    //the application-side buffer designateed for this channel
    int* buffer;
    int dir;
    snd_pcm_hw_params_t* params; //the hardware struct actually used by alsa in initHardware
    streamParams sParams; //set inside the parser

    pcmHandle(const char* deviceName) : device_name(deviceName) {};

} pcmHandle_t;

class AudioIO {

    public:
        AudioIO();
        std::vector<float> readReferenceSignal(); //blocking: 
        //int readErrorSignal(float* buffer, int size);
        int writeAntinoiseSignal();
        int closeInterface(pcmHandle_t* handle);
        int readErrorSignal();

    private: //just building for ref mic right now
        std::vector<pcmHandle_t*> handles;
        hardwareConfig_t hardwareConfig;
        int currentIndexOfBuffer = 0;

        //hardware configuration should only be within the class, not accessible by the user/outside this interface internally
        //only called within constructor
        void parseHardwareConfig(const char* cfgFilePath);
        void initHardware();

        std::vector<float> x; //buffer to hold the reference signal read from the mic, which will be passed to the controller/fxlms class for processing



};