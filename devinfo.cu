#include <stdio.h>

static void print_device_limit(const char* s, cudaLimit e)
{
  size_t v;
  cudaDeviceGetLimit(&v, e);
  printf("%s %d\n", s, v);
}

static void print_device_prop(struct cudaDeviceProp* p)
{
  printf("name \"%s\"\n", p->name);
  printf("total global memory %zu bytes\n", p->totalGlobalMem);
  printf("shared memory per block %zu bytes\n", p->sharedMemPerBlock);
  printf("registers per block %d\n", p->regsPerBlock);
  printf("warp size %d\n", p->warpSize);
  printf("memory pitch %zu\n", p->memPitch);
  printf("max threads per block %d\n", p->maxThreadsPerBlock);
  printf("max threads dim (%d,%d,%d)\n", p->maxThreadsDim[0], p->maxThreadsDim[1], p->maxThreadsDim[2]);
  printf("max grid size (%d,%d,%d)\n", p->maxGridSize[0], p->maxGridSize[1], p->maxGridSize[2]);
  printf("peak clock rate %dkHz\n", p->clockRate);
  printf("total const mem %zu\n", p->totalConstMem);
  static char const* const names[6] = {
    "???",
    "Tesla",
    "Fermi",
    "Kepler",
    "???",
    "Maxwell"
  };
  printf("compute capability %d.%d (%s)\n", p->major, p->minor, names[p->major]);
  printf("supports concurrent memcpy+compute %d\n", p->deviceOverlap);
  printf("SM count %d\n", p->multiProcessorCount);
  printf("timeout enabled %d\n", p->kernelExecTimeoutEnabled);
  printf("\"integrated\" with memory %d\n", p->integrated);
  printf("can map host memory %p\n", p->canMapHostMemory);
  printf("compute mode: ");
  if (p->computeMode == cudaComputeModeDefault)
    printf("device not restricted and multiple CPU threads can use it\n");
  else if (p->computeMode == cudaComputeModeExclusive)
    printf("only one CPU thread can use it\n");
  else if (p->computeMode == cudaComputeModeProhibited)
    printf("using this device is prohibited !\n");
  else if (p->computeMode == cudaComputeModeExclusiveProcess)
    printf("multiple CPU threads my use it, but all from one process\n");
  else
    printf("unknown compute mode %d\n", p->computeMode);
  printf("supports concurrent kernels %d\n", p->concurrentKernels);
  printf("ECC enabled %d\n", p->ECCEnabled);
  printf("PCI bus ID %d\n", p->pciBusID);
  printf("PCI device ID %d\n", p->pciDeviceID);
  printf("PCI domain ID %d\n", p->pciDomainID);
  printf("async engine count %d\n", p->asyncEngineCount);
  printf("supports unified addressing %d\n", p->unifiedAddressing);
  printf("peak memory clock rate %dkHz\n", p->memoryClockRate);
  printf("memory bus width %d bits\n", p->memoryBusWidth);
  printf("L2 cache size %d bytes\n", p->l2CacheSize);
  printf("max threads per SM %d\n", p->maxThreadsPerMultiProcessor);
  printf("supports stream priorities %d\n", p->streamPrioritiesSupported);
  printf("supports global L1 cache %d\n", p->globalL1CacheSupported);
  printf("supports local L1 cache %d\n", p->localL1CacheSupported);
  printf("shared mem per SM %zu bytes\n", p->sharedMemPerMultiprocessor);
  printf("registers per SM %d\n", p->regsPerMultiprocessor);
//printf("supports managed memory %d\n", p->managedMemSupported);
  printf("is multi GPU board %d\n", p->isMultiGpuBoard);
  printf("multi GPU board group ID %d\n", p->multiGpuBoardGroupID);
}

static void print_device_info(void)
{
  int count;
  cudaGetDeviceCount(&count);
  int dev;
  cudaGetDevice(&dev);
  printf("%d devices, using #%d\n", count, dev);
  struct cudaDeviceProp props;
  cudaGetDeviceProperties(&props, dev);
  print_device_prop(&props);
  print_device_limit("GPU thread stack size (bytes)", cudaLimitStackSize);
  print_device_limit("malloc heap size (bytes)", cudaLimitMallocHeapSize);
}

int main()
{
  print_device_info();
  return 0;
}
