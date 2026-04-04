//
#include <vector>
#include <cmath>
#include "controller.h"
#include <stdio.h>
#include <iostream>


int main ()
{

    //std::vector<float> shatTest(256, 0.0f);
    std::vector<float> shat(256, 0.0f);
    shat[0] = 1.0f;  // change later j a lil placeholder cuz its a identity impulse response
    Controller controllerObj(shat, 101, 0.01); //lower mu

    // main loop: CURRENTLY BLOCKING, 1 THREAD
    std::cout << "main: loop start" << std::endl;

    //std::cout << "main: loop" << std::endl;
    controllerObj.startLearningLoop();
    //std::cout << "Controller: pushReferenceSignal()" << std::endl;

    return 0;
}



