//
#include <vector>
#include <cmath>
#include "controller.h"
#include <stdio.h>


int main ()
{

    Controller controllerObj = new Controller();

    // main loop: CURRENTLY BLOCKING, 1 THREAD

    while (1) {

        controllerObj.pushReferenceSignal();
        std::cout << "Controller: pushReferenceSignal()" << std::endl;

    }

    return 0;
}



