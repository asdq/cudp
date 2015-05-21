#ifndef CUDA_DEVICE_H
#define CUDA_DEVICE_H

#include <exception>
#include <memory>
#include <string>

struct cudaDeviceProp;

class CudaDevice
{
    std::unique_ptr<cudaDeviceProp> m_prop;

public:

    class Exception : public std::exception
    {
        std::string m_msg;
    public:
        Exception(const std::string &msg) noexcept : m_msg(msg) {}
        Exception(std::string &&msg) noexcept : m_msg(std::move(msg)) {}
        const char* what() const noexcept { return m_msg.c_str(); }
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
