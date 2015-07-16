#include <math.h>
#include "refine.cuh"
#include "size.cuh"
#include "vtk.cuh"
#include "mycuda.cuh"
#include <stdio.h>
#include <time.h>

static double get_time(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  double s = ts.tv_sec;
  double ns = ts.tv_nsec;
  s += (ns / 1e9);
  return s;
}

static void trigger_cuda_init(void)
{
  int* p;
  CUDACALL(cudaMalloc(&p, sizeof(int)));
  CUDACALL(cudaFree(p));
}

int main()
{
  int const fvs_dat[6] = {
    0,1,2,
    2,3,0
  };
  struct x const x_dat[4] = {
    {0,0},
    {1,0},
    {1,1},
    {0,1}
  };
  double t0 = get_time();
  trigger_cuda_init();
  double t1 = get_time();
  printf("CUDA init took %f seconds\n", t1 - t0);
  double t2 = get_time();
  struct rgraph fvs = rgraph_new_from_host(2, 3, fvs_dat);
  struct xs xs = xs_new_from_host(4, x_dat);
  int done = 0;
  while (!done) {
    struct ss dss = gen_size_field(fvs, xs);
    struct rgraph fvs2;
    struct xs xs2;
    refine(fvs, xs, dss, &fvs2, &xs2);
    if (fvs.nverts == fvs2.nverts)
      done = 1;
    ss_free(dss);
    rgraph_free(fvs);
    xs_free(xs);
    fvs = fvs2;
    xs = xs2;
  }
  CUDACALL2(cudaDeviceSynchronize());
  double t3 = get_time();
  printf("num faces %d, BLOCK_SIZE %d\n", fvs.nverts, BLOCK_SIZE);
  printf("refinement took %f seconds\n", t3 - t2);
//write_vtk("out.vtk", &fvs, &xs);
  xs_free(xs);
  rgraph_free(fvs);
  return 0;
}
