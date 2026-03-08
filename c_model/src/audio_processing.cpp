
#include "include\audio_processing.h"
#include "simpleini\SimpleIni.h"
#include <vector>

// Initialize audio processing interface
CSimpleIniA ini;
ini.SetUnicode();

AudioIO::AudioIO() : {
    //constructor

    //parse hardwareConfig via the iniParser
    handles = parseHardwareConfig(HARDWARECONFIGPATH);
}

//parsing the ini file, outputting a hardwareConfig struct with the configuration
pcmHandle_t* AudioIO::parseHardwareConfig(char* cfgFilePath) {

    //fill in the hardwareConfig str uct with the details from the ini file, using the SimpleIni library
    ini.LoadFile(cfgFilePath);
    int count;
    
    //assign reference mic
    char* refMicDevice = ini[reference_mics][0];
    count++;

    hardwareConfig.numRefMics = count;
    hardwareConfig.sParams = {
        .periods = (unsigned int)ini[audio][periods];
        .rate = (unsigned int)ini[audio][rate];
        .period_size = ini[audio][period_size];
        .format = ini[audio][format]; //casting uhh with an alsa type
    }
    hardwareConfig.devices[0] = refMicDevice;

    std::vector<pcmHandle_t*> pcmHandleList(count);

    initHardware(hardwareConfig, pcmHandleList);

    return pcmHandleList;

}

//passing the pcmHandleList by reference to modify the real one and have this function as void
void AudioIO::initHardware(hardwareConfig_t hardwareConfig, std::vector<pcmHandle_t*>& pcmHandleList) {
    //create the pcmHandle structs for each peripheral in the hardware config
    //returns array of pcmHandles, sorted
    //pcm hardware params created

    snd_pcm_open(&(pcmHandleList[0].handle), "hw:0,0", SND_PCM_STREAM_CAPTURE, 0); //KEY: hw01 is the mic adc on the vm audio input enabled linux machine

    //set hardware parameters using all the relevant methods

    snd_pcm_hw_params_alloca(&(pcmHandleList[0].params));

    nd_pcm_hw_params(&(pcmHandleList[0].handle), &(pcmHandleList[0].params));
    
    // fill with default values
    snd_pcm_hw_params_any(&(pcmHandleList[0].handle), &(pcmHandleList[0].params));

    //set period size
    snd_pcm_hw_params_set_period_size_near(&(pcmHandleList[0].handle), &(pcmHandleList[0].params), &periodSize, &dir);

    snd_pcm_hw_params(&(pcmHandleList[0].handle), &(pcmHandleList[0].params));

    snd_pcm_prepare(&(pcmHandleList[0].handle));

}


std::vector<float> AudioIO::readReferenceSignal() {

    //blocking read: reads until buffer of size periodSize is full, then returns number of frames read (should be periodSize unless error)
    rc = snd_pcm_readi(handles[0], handles[0]->buffer, handles[0]->sParams.period_size);
    //printf("%d\n", handles[0]); //first value in frame 
    //push the values read from the buffer into the reference signal buffer: rewrites
    for (int i = 0; i < handles[0]->sParams.period_size; i++) {
        x[i] = handles[0]->buffer[i];
        printf("%d\n", handles[0]);
    }

    //
}

void AudioIO::writeAntinoiseSignal() {

    

}

void AudioIO::closeInterface(pcmHandle_t* handle) {
    snd_pcm_close(handle->handle);
}


