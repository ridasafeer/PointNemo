#include "audio_processing.h"
#include "simpleini/SimpleIni.h"
#include <vector>
#include <string>

#define SI_CONVERT_GENERIC

CSimpleIniA ini;

AudioIO::AudioIO() {
    //constructor
    //parse hardwareConfig via the iniParser: produces hardwareConfig struct for instance & (2) the handles array in the class
    parseHardwareConfig("anc.ini");
    initHardware();
    std::cout << "passed initHardware()" << std::endl; //FAILED: issue is in PARSER
    ini.SetUnicode();
}

//parsing the ini file, outputting a hardwareConfig struct with the configuration
void AudioIO::parseHardwareConfig(const char* cfgFilePath) {

    // Initialize audio processing interface
    CSimpleIniA::TNamesDepend sections;
    CSimpleIniA::TNamesDepend keys;
    //fill in the hardwareConfig str uct with the details from the ini file, using the SimpleIni library
    SI_Error rc = ini.LoadFile("anc.ini");
    // if (rc < 0) {
    //     std::cout << rc << std::endl;
    // }
    int count = 0;
    ini.GetAllSections(sections); //get all sections
    const char* test = ini.GetValue("audio", "periods", "Hello: getVal failed"); //error: returning nullptr
    std::cout << test << std::endl;

    //sParams obj for each handle
    hardwareConfig.sParams.periods = static_cast<unsigned int>(std::stoi(ini.GetValue("audio", "periods")));

    hardwareConfig.sParams.rate =
        static_cast<unsigned int>(std::stoi(ini.GetValue("audio", "rate")));

    hardwareConfig.sParams.period_size =
        static_cast<snd_pcm_uframes_t>(std::stoul(ini.GetValue("audio", "period_size")));

    std::cout << hardwareConfig.sParams.periods << std::endl;
    std::cout << hardwareConfig.sParams.period_size << std::endl;
    std::cout << hardwareConfig.sParams.rate << std::endl;
        
    //For any config
    for (auto& section : sections) {
        const char* currentDevice = section.pItem;
        printf("%s\n", currentDevice);
        //within the current section, add all devices
        if (std::string(currentDevice) == "audio") { //skip the first section
            continue;
        }
        bool rc = ini.GetAllKeys(currentDevice, keys);
        // if (!rc) {
        //     std::cout << "Error: Section not found" << std::endl;
        // }
        hardwareConfig.devices = new const char*[10]; //10 const char* ptrs, therefore 3 ptrs to char ptrs

        for (auto& key : keys) {
            printf("%s\n", key.pItem);
            const char* device = ini.GetValue(currentDevice, key.pItem);
            printf("%s\n", device);
            printf("%d\n", count);
            
            size_t len = std::strlen(device);

            //best method
            char* buf = new char[len + 1]; //ptr to buffer on the heap, temp variable to get the ptr to heap
            std::strcpy(buf, device); //str copy pretty much only way to init this string on the heap with a string that already exists
            hardwareConfig.devices[count] = buf;
            //create a new pcmHandle_t struct object for it as well
            //for this device found under this section, create a new pcmHandle
            pcmHandle_t* newDeviceHandle = new pcmHandle(hardwareConfig.devices[count]); //on the heap, returns ptr
            std::cout << "new pcmHandle_t ptr made and added to handles[]" << std::endl;
            
            handles.push_back(newDeviceHandle); //also creates an index here, avoids seg fault

            std::cout << "new pcmHandle_t newDeviceHandle appended to handles[]" << std::endl;

            //create the handles application-side buffer: to hold a max of 3 periods
            handles[count]->buffer = new int(); //returns int* pointer, can traverse as array on heap

            handles[count]->sParams = hardwareConfig.sParams;

            //identify which device type (ref mic, speaker, error mic) and config pcmHandle attrs accordingly
            if (currentDevice == "reference_mics" | currentDevice == "error_mics") {
                handles[count]->direction = SND_PCM_STREAM_CAPTURE;
            } else {
                handles[count]->direction = SND_PCM_STREAM_PLAYBACK;
            }

            handles[count]->channels = 2; //all are stereo
            handles[count]->format = SND_PCM_FORMAT_S16_LE; //all used signed 16 bit
            count++;
        }
    }

    std::cout << "all device pcm interfaces created, parsing complete" << std::endl;
    hardwareConfig.numDevices = count;
}

//passing the pcmHandleList by reference to modify the real one and have this function as void
void AudioIO::initHardware() {

    for (int i = 0; i < hardwareConfig.numDevices; i++) {

        std::cout << i << std::endl;
        snd_pcm_open(&handles[i]->handle, handles[i]->device_name, handles[i]->direction, 0); //KEY: hw01 is the mic adc on the vm audio input enabled linux machine

        std::cout << "alsa open()" << std::endl;

        streamParams currentHandleStreamParams = handles[i]->sParams;
        //allocate a default params struct on heap

        snd_pcm_hw_params_alloca(&handles[i]->params);
        std::cout << "alsa alloca()" << std::endl;

        //set the hardware parameters
        
        // fill with default values
        snd_pcm_hw_params_any(handles[i]->handle, handles[i]->params);
        std::cout << "alsa default params()" << std::endl;

        // set period size
        snd_pcm_hw_params_set_period_size_near(handles[i]->handle, handles[i]->params, &currentHandleStreamParams.period_size, &handles[i]->dir);

        snd_pcm_hw_params(handles[i]->handle, handles[i]->params);

        snd_pcm_prepare(handles[i]->handle);

        std::cout << "all device pcm interfaces init'd" << std::endl;

    }

}


//Designed for only 1 reference mic signal
//TODO: how to identify whcih one is refernce mic or which reference mic to read from
std::vector<float> AudioIO::readReferenceSignal() {

    //blocking read: reads until buffer of size periodSize is full, then returns number of frames read (should be periodSize unless error)
    std::cout << handles[0]->device_name << std::endl;
    int rc = snd_pcm_readi(handles[0]->handle, handles[0]->buffer, handles[0]->sParams.period_size);
    //printf("%d\n", handles[0]); //first value in frame 
    //push the values read from the buffer into the reference signal buffer: rewrites
    
    for (int i = 0; i < handles[0]->sParams.period_size; i++) {
        x[i] = handles[0]->buffer[i];
        printf("%d\n", handles[0]->buffer[i]);
    }

    return x;
}

int AudioIO::writeAntinoiseSignal() {
    return 0;

}

int AudioIO::closeInterface(pcmHandle_t* handle) {
    return 0;
}


int AudioIO::readErrorSignal() {
    return 0;
}
