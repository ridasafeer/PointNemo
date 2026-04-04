//
#include <vector>
#include <cmath>
#include "controller.h"
#include <stdio.h>
#include <iostream>


int main ()
{

    std::vector<float> shatTest(256, 0.0f);
    Controller controllerObj(shatTest, 101, 0.2);

    // main loop: CURRENTLY BLOCKING, 1 THREAD
    std::cout << "main: loop start" << std::endl;
    controllerObj.startLearningLoop();
    std::cout << "awooga" << std::endl;

    return 0;
}



