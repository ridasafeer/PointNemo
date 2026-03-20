//
#include <vector>
#include <cmath>
#include "controller.h"
#include <stdio.h>
#include <iostream>


int main ()
{

    std::vector<float> shatTest(100, 0.0f);
    Controller controllerObj(shatTest, 100, 0.2);

    // main loop: CURRENTLY BLOCKING, 1 THREAD

    while (1) {

        controllerObj.pushReferenceSignal();
        std::cout << "Controller: pushReferenceSignal()" << std::endl;

    }

    return 0;
}



