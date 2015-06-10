#include "ints.h"
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

struct ints ints_new(int n)
{
  struct ints is;
  is.n = n;
  is.i = malloc(sizeof(int) * n);
  return is;
}

void ints_free(struct ints is)
{
  free(is.i);
}

struct ints ints_exscan(struct ints is)
{
  int nthreads = omp_get_max_threads();
  int* thread_sums = malloc(sizeof(int) * nthreads);
  thread_sums[0] = 0;
  #pragma omp parallel
  {
    int thread_sum = 0;
    #pragma omp for schedule(static)
    for (int i = 0; i < is.n; ++i)
      thread_sum += is.i[i];
    thread_sums[omp_get_thread_num()] = thread_sum;
  }
  {
    int accum = 0;
    for (int i = 0; i < nthreads; ++i) {
      int sum = thread_sums[i];
      thread_sums[i] = accum;
      accum += sum;
    }
  }
  struct ints o = ints_new(is.n + 1);
  o.i[0] = 0;
  #pragma omp parallel
  {
    int accum = thread_sums[omp_get_thread_num()];
    #pragma omp for schedule(static)
    for (int i = 0; i < is.n; ++i) {
      accum += is.i[i];
      o.i[i + 1] = accum;
    }
  }
  free(thread_sums);
  return o;
}

int ints_max(struct ints is)
{
  int max = is.i[0];
  #pragma omp parallel
  {
    int thread_max = is.i[0];
    #pragma omp for
    for (int i = 1; i < is.n; ++i)
      if (is.i[i] > thread_max)
        thread_max = is.i[i];
    #pragma omp critical
    {
      if (thread_max > max)
        max = thread_max;
    }
  }
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
  for (int i = 0; i < is.n; ++i)
    is.i[i] = 0;
}

void ints_from_dat(struct ints is, int const dat[])
{
  #pragma omp parallel for
  for (int i = 0; i < is.n; ++i)
    is.i[i] = dat[i];
}
