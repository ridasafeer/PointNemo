//
#include <vector>
#include <cmath>
#include "controller.h"
#include <stdio.h>
#include <iostream>


int main ()
{

    std::vector<float> shatTest(256, 21.0f); //TODO: at some point, use the real calib function and check if works
    Controller controllerObj(shatTest, 101, 0.2);

    // main loop: CURRENTLY BLOCKING, 1 THREAD
    std::cout << "main: loop start" << std::endl;
    controllerObj.startLearningLoop();
    std::cout << "awooga" << std::endl;

    return 0;
}



