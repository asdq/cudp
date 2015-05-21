#include "cudadevice.h"
#include <cassert>
#include <cuda.h>
#include <sstream>

int CudaDevice::deviceCount()
{
    int n;
    cudaError_t res;
    
    res = cudaGetDeviceCount(&n);
    if (res != cudaSuccess) { n = 0; }
    return n;
}

CudaDevice::CudaDevice(int dev)
{
    cudaError_t res;
    m_prop = new cudaDeviceProp;
    
    res = cudaGetDeviceProperties(m_prop, dev);
    if (res != cudaSuccess) {
        std::stringstream sstr;
        
        delete m_prop;
        sstr << "CudaDevice: invalid device " << dev << '.';
        throw Exception(sstr.str());
    }
}

CudaDevice::~CudaDevice()
{
    delete m_prop;
}

// hide default copy constructor
CudaDevice::CudaDevice(const CudaDevice&)
{
    assert(false);
}

// hide default assigment
CudaDevice& CudaDevice::operator = (const CudaDevice&)
{
    assert(false);
    return *this;
}

int CudaDevice::majorVersion() const
{
    return m_prop -> major;
}

int CudaDevice::minorVersion() const
{
    return m_prop -> minor;
}

const char* CudaDevice::name() const
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
    return m_prop -> canMapHostMemory > 0;
}
