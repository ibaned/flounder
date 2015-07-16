#include "ints.cuh"
#include "mycuda.cuh"
#include <assert.h>
#include <limits.h>
#if 0
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#endif

struct ints ints_new(int n)
{
  struct ints is;
  is.n = n;
  CUDACALL(cudaMalloc(&is.i, sizeof(int) * n));
  return is;
}

void ints_free(struct ints is)
{
  CUDACALL(cudaFree(is.i));
}

struct ints ints_exscan(struct ints is)
{
  struct ints o = ints_new(is.n + 1);
#if 0
  thrust::device_ptr<int> inp(is.i);
  thrust::device_ptr<int> outp(o.i);
  thrust::exclusive_scan(inp, inp + is.n, outp);
  /* fixup the last element quirk */
  int sum = thrust::reduce(inp, inp + is.n);
  CUDACALL(cudaMemcpy(o.i + is.n, &sum, sizeof(int), cudaMemcpyHostToDevice));
#else
  int* hi = (int*) malloc(sizeof(int) * (is.n + 1));
  int* ho = (int*) malloc(sizeof(int) * (is.n + 1));
  CUDACALL(cudaMemcpy(hi, is.i, sizeof(int) * is.n, cudaMemcpyDeviceToHost));
  int sum = 0;
  ho[0] = 0;
  for (int i = 0; i < is.n; ++i) {
    sum += hi[i];
    ho[i + 1] = sum;
  }
  CUDACALL(cudaMemcpy(o.i, ho, sizeof(int) * (is.n + 1), cudaMemcpyHostToDevice));
  free(hi);
  free(ho);
#endif
  return o;
}

int ints_max(struct ints is)
{
#if 0
  thrust::device_ptr<int> p(is.i);
  return thrust::reduce(p, p + is.n, INT_MIN, thrust::maximum<int>());
#else
  int* hi = (int*) malloc(sizeof(int) * is.n);
  CUDACALL(cudaMemcpy(hi, is.i, sizeof(int) * is.n, cudaMemcpyDeviceToHost));
  int max = hi[0];
  for (int i = 1; i < is.n; ++i)
    if (hi[i] > max)
      max = hi[i];
  free(hi);
  return max;
#endif
}

void ints_zero(struct ints is)
{
  CUDACALL(cudaMemset(is.i, 0, sizeof(int) * is.n));
}

void ints_copy(struct ints into, struct ints from, int n)
{
  CUDACALL(cudaMemcpy(into.i, from.i, sizeof(int) * n, cudaMemcpyDeviceToDevice));
}

void ints_from_host(struct ints is, int const host_dat[])
{
  CUDACALL(cudaMemcpy(is.i, host_dat, sizeof(int) * is.n, cudaMemcpyHostToDevice));
}
