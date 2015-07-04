#include "ints.h"
#include "mycuda.h"
#include <stdlib.h>
#include <stdio.h>

struct ints ints_new(int n)
{
  struct ints is;
  is.n = n;
  cudaMalloc(&is.i, sizeof(int) * n);
  return is;
}

void ints_free(struct ints is)
{
  cudaFree(is.i);
}

struct ints ints_exscan(struct ints is)
{
  struct ints o = ints_new(is.n + 1);
  int a = 0;
  o.i[0] = 0;
  for (int i = 0; i < is.n; ++i) {
    a += is.i[i];
    o.i[i + 1] = a;
  }
  return o;
}

int ints_max(struct ints is)
{
  int max = is.i[0];
  for (int i = 1; i < is.n; ++i)
    if (is.i[i] > max)
      max = is.i[i];
  return max;
}

void ints_print(struct ints is)
{
  printf("ints (%d)\n", is.n);
  for (int i = 0; i < is.n; ++i)
    printf("%d: %d\n", i, is.i[i]);
}

__global__ void ints_zero_0(struct ints is)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < is.n)
    is.i[i] = 0;
}

void ints_zero(struct ints is)
{
  CUDACALL(ints_zero_0, is.n, is);
}

void ints_from_dat(struct ints is, int const dat[])
{
  for (int i = 0; i < is.n; ++i)
    is.i[i] = dat[i];
}
