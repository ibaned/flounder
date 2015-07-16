#ifndef MYCUDA_H
#define MYCUDA_H

#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

static inline int ceildiv(int a, int b)
{
  int c = a / b;
  if (a % b)
    ++c;
  return c;
}

#define CUDACALL2(f) \
do { \
  cudaError_t err = f; \
  if (err != cudaSuccess) { \
    const char* errs = cudaGetErrorString(err); \
    fprintf(stderr, "call %s failed at %s:%d : %s\n", \
                    #f, __FILE__, __LINE__, errs); \
    abort(); \
  } \
} while (0)

#define CUDACALL(f) \
do { \
  CUDACALL2(f); \
} while (0)

#define CUDALAUNCH(fname,n,args) \
do { \
  fname<<< ceildiv((n), BLOCK_SIZE), BLOCK_SIZE >>>args; \
} while (0)

#define CUDAINDEX (blockIdx.x * blockDim.x + threadIdx.x)

#endif
