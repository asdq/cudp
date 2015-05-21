#ifndef CUDA_DEVICE_H
#define CUDA_DEVICE_H

#include <exception>
#include <string>

struct cudaDeviceProp;

class CudaDevice
{
    cudaDeviceProp *m_prop;

    // hide default copy constructor and assigment
    CudaDevice(const CudaDevice&);
    CudaDevice& operator = (const CudaDevice&);

public:

    class Exception : public std::exception
    {
        std::string m_msg;
    public:
        Exception(const std::string &msg) : m_msg(msg) {}
        const char* what() const { return m_msg.c_str(); }
    };

    static
    int deviceCount();

    explicit
    CudaDevice(int);

    ~CudaDevice();

    int majorVersion() const;
    int minorVersion() const;
    const char* name() const;
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
