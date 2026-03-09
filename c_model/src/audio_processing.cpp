
#include "audio_processing.h"
#include "SimpleIni.h"
#include <vector>

// Initialize audio processing interface
CSimpleIniA ini;
CSimpleIniA::TNamesDepend sections;
CSimpleIniA::TNamesDepend keys;

AudioIO::AudioIO() {
    //constructor

    //parse hardwareConfig via the iniParser: produces hardwareConfig struct for instance & (2) the handles array in the class
    parseHardwareConfig(HARDWARECONFIGPATH);
    initHardware();
    ini.SetUnicode();
}

//parsing the ini file, outputting a hardwareConfig struct with the configuration
void AudioIO::parseHardwareConfig(const char* cfgFilePath) {

    //fill in the hardwareConfig str uct with the details from the ini file, using the SimpleIni library
    ini.LoadFile(cfgFilePath);
    int count = 0;
    ini.GetAllSections(sections);

    //sParams obj for each handle
    hardwareConfig.sParams.periods =
        static_cast<unsigned int>(std::stoi(ini.GetValue("audio", "periods", "2")));

    hardwareConfig.sParams.rate =
        static_cast<unsigned int>(std::stoi(ini.GetValue("audio", "rate", "48000")));

    hardwareConfig.sParams.period_size =
        static_cast<snd_pcm_uframes_t>(std::stoul(ini.GetValue("audio", "period_size", "256")));
        
    //For 1 ref mic and 1 speaker
    for (auto& section : sections) {
        char* currentDevice = section.pItem;
        //within the current section, add all devices
        ini.GetAllKeys(currentDevice, keys);
        for (auto& key : keys) {
            char* device = key.pItem;
            hardwareConfig.devices[count] = ini.GetValue(currentDevice, device);
            //create a new pcmHandle_t struct object for it as well
            pcmHandle_t deviceHandle = new pcmHandle_t;
            handles.push_back(&deviceHandle);
            handles[count]->sParams = hardwareConfig.sParams;
            handles[count]->device_name = ini.GetValue(currentDevice, device);
            count++;
        }
    }

    hardwareConfig.numDevices = count;
}

//passing the pcmHandleList by reference to modify the real one and have this function as void
void AudioIO::initHardware() {

    for (int i = 0; i < hardwareConfig.numDevices; i++) {

        snd_pcm_open(handles[i]->handle, handles[i]->device_name, SND_PCM_STREAM_CAPTURE, 0); //KEY: hw01 is the mic adc on the vm audio input enabled linux machine

        //set hardware parameters using all the relevant methods

        snd_pcm_hw_params_alloca(handles[i]->params);

        nd_pcm_hw_params(handles[i]->.handle, handles[i]->.params);
        
        // fill with default values
        snd_pcm_hw_params_any(handles[i]->handle, handles[i]->params);

        //set period size
        snd_pcm_hw_params_set_period_size_near(handles[i]->handle, handles[i]->params, &periodSize, &dir);

        snd_pcm_hw_params(handles[i]->handle, handles[i]->params);

        snd_pcm_prepare(handles[i]->handle);

    }

}


//Designed for only 1 reference mic signal
//TODO: how to identify whcih one is refernce mic or which reference mic to read from
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

int AudioIO::writeAntinoiseSignal() {
    
    return 0;

}

int AudioIO::closeInterface(pcmHandle_t* handle) {
    snd_pcm_close(handle->handle);
    return 0;
}


