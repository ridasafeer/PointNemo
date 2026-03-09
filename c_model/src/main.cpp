//
#include <vector>
#include <cmath>
#include "controller.h"
#include <stdio.h>


int main ()
{

    std::vector<float> shatTest(100, 0.0f);
    Controller controllerObj = new Controller(shatTest, 100, 0.2);

    // main loop: CURRENTLY BLOCKING, 1 THREAD

    while (1) {

        controllerObj.pushReferenceSignal();
        std::cout << "Controller: pushReferenceSignal()" << std::endl;

    }

    return 0;
}



