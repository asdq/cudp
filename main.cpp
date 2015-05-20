#include <iostream>
#include "cudadevice.h"

using namespace std;

int main()
{
    auto nDev = CudaDevice::deviceCount();
    
    cout << "Number of devices found: " << nDev << endl;
    return 0;
}

