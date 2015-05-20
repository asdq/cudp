#include <cuda.h>
#include <cstdio>
#include "cudadevice.h"

#define cudaCheck(stmt)                                     \
    do {                                                    \
        cudaError_t err = stmt;                             \
        if (err != cudaSuccess) {                           \
            fprintf(stderr, "%s in %s at line %d\n",        \
            cudaGetErrorString(err), __FILE__, __LINE__);   \
            cudaDeviceReset();                              \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while(0)

int CudaDevice::deviceCount()
{
    int n;
    
    cudaCheck(cudaGetDeviceCount(&n));
    return n;
}

CudaDevice::CudaDevice(int dev)
{
    auto p = new cudaDeviceProp;
    cudaCheck(cudaGetDeviceProperties(p, dev));
    m_prop = std::unique_ptr<cudaDeviceProp>(p);
}

CudaDevice::~CudaDevice() {}

int CudaDevice::majorVersion() const
{
    return m_prop -> major;
}

int CudaDevice::minorVersion() const
{
    return m_prop -> minor;
}

char* CudaDevice::name() const
{
    return m_prop -> name;
}

unsigned CudaDevice::totalGlobalMem() const
{
    return m_prop -> totalGlobalMem;
}

unsigned CudaDevice::totalConstMem() const
{
    return m_prop -> totalConstMem;
}

unsigned CudaDevice::sharedMemPerBlock() const
{
    return m_prop -> sharedMemPerBlock;
}

int CudaDevice::maxThreadsPerBlock() const
{
    return m_prop -> maxThreadsPerBlock;
}

int CudaDevice::maxThreadsPerMultiProcessor() const
{
    return m_prop -> maxThreadsPerMultiProcessor;
}

int CudaDevice::maxGridSizeX() const
{
    return m_prop -> maxGridSize[0];
}

int CudaDevice::maxGridSizeY() const
{
    return m_prop -> maxGridSize[1];
}

int CudaDevice::maxGridSizeZ() const
{
    return m_prop -> maxGridSize[2];
}

int CudaDevice::maxThreadsDimX() const
{
    return m_prop -> maxThreadsDim[0];
}

int CudaDevice::maxThreadsDimY() const
{
    return m_prop -> maxThreadsDim[1];
}

int CudaDevice::maxThreadsDimZ() const
{
    return m_prop -> maxThreadsDim[2];
}

int CudaDevice::warpSize() const
{
    return m_prop -> warpSize;
}

bool CudaDevice::deviceOverlap() const
{
    return m_prop -> deviceOverlap > 0;
}

int CudaDevice::asyncEngineCount() const
{
    return m_prop -> asyncEngineCount;
}

bool CudaDevice::deviceMapHost() const
{
    return m_prop -> canMapHostMemory;
}
