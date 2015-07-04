#include "ints.h"
#include "mycuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

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

void ints_zero(struct ints is)
{
  cudaMemset(is.i, 0, sizeof(int) * is.n);
}

void ints_copy(struct ints into, struct ints from)
{
  assert(into.n >= from.n);
  cudaMemcpy(into.i, from.i, sizeof(int) * from.n, cudaMemcpyDeviceToDevice);
}

void ints_from_host(struct ints is, int const host_dat[])
{
  cudaMemcpy(is.i, host_dat, sizeof(int) * is.n, cudaMemcpyHostToDevice);
}
