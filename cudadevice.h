#ifndef CUDA_DEVICE_H
#define CUDA_DEVICE_H

#include <memory>

struct cudaDeviceProp;

class CudaDevice
{
    std::unique_ptr<cudaDeviceProp> m_prop;

public:

    static int deviceCount();

    explicit
    CudaDevice(int);

    ~CudaDevice();

    int majorVersion() const;
    int minorVersion() const;
    char* name() const;
    bool deviceMapHost() const;
    bool deviceOverlap() const;
    unsigned totalGlobalMem() const;
    unsigned totalConstMem() const;
    unsigned sharedMemPerBlock() const;
    int maxThreadsPerBlock() const;
    int maxThreadsPerMultiProcessor() const;
    int maxGridSizeX() const;
    int maxGridSizeY() const;
    int maxGridSizeZ() const;
    int maxThreadsDimX() const;
    int maxThreadsDimY() const;
    int maxThreadsDimZ() const;
    int warpSize() const;
    int asyncEngineCount() const;
};

#endif
