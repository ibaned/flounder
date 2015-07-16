#include "space.cuh"
#include "mycuda.cuh"

struct xs xs_new(int n)
{
  struct xs xs;
  xs.n = n;
  cudaMalloc(&xs.x, sizeof(struct x) * n);
  cudaDeviceSynchronize();
  return xs;
}

void xs_free(struct xs xs)
{
  cudaFree(xs.x);
}

struct xs xs_new_from_host(int n, struct x const dat[])
{
  struct xs xs = xs_new(n);
  cudaMemcpy(xs.x, dat, sizeof(struct x) * n, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  return xs;
}

struct ss ss_new(int n)
{
  struct ss ss;
  ss.n = n;
  cudaMalloc(&ss.s, sizeof(double) * n);
  cudaDeviceSynchronize();
  return ss;
}

static __global__ void ss_new_const_0(struct ss ss, double v)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ss.n)
    ss.s[i] = v;
}

struct ss ss_new_const(int n, double v)
{
  struct ss ss = ss_new(n);
  CUDACALL(ss_new_const_0, n, (ss, v));
  return ss;
}

void ss_free(struct ss ss)
{
  cudaFree(ss.s);
}

static __global__ void ss_gt_0(struct ints gts, struct ss a, struct ss b)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < a.n)
    gts.i[i] = (a.s[i] > b.s[i]);
}

struct ints ss_gt(struct ss a, struct ss b)
{
  struct ints gts = ints_new(a.n);
  CUDACALL(ss_gt_0, a.n, (gts, a, b));
  return gts;
}

static __global__ void compute_areas_0(struct ss as, struct xs xs, struct rgraph fvs)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < fvs.nverts) {
    int fv[3];
    rgraph_get(fvs, i, fv);
    struct x fx[3];
    xs_get(xs, fv, 3, fx);
    as.s[i] = fx_area(fx);
  }
}

struct ss compute_areas(struct xs xs, struct rgraph fvs)
{
  struct ss as = ss_new(fvs.nverts);
  CUDACALL(compute_areas_0, fvs.nverts, (as, xs, fvs));
  return as;
}
