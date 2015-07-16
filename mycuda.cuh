#ifndef MYCUDA_H
#define MYCUDA_H

#define BLOCK_SIZE 256

static inline int ceildiv(int a, int b)
{
  int c = a / b;
  if (a % b)
    ++c;
  return c;
}

#define CUDACALL(f) \
do { \
  cudaError_t err = f; \
  if (err != cudaSuccess) \
    fprintf(stderr, "call %s failed at %s:%d\n", \
                    #f, __FILE__, __LINE__); \
} while (0)

#define CUDALAUNCH(fname,n,args) \
do { \
  fname<<< ceildiv((n), BLOCK_SIZE), BLOCK_SIZE >>>args; \
  cudaDeviceSynchronize(); \
} while (0)

#define CUDAINDEX (blockIdx.x * blockDim.x + threadIdx.x)

#endif
