#include "space.cuh"
#include "mycuda.cuh"

struct xs xs_new(int n)
{
  struct xs xs;
  xs.n = n;
  CUDACALL(cudaMalloc(&xs.x, sizeof(struct x) * n));
  return xs;
}

void xs_free(struct xs xs)
{
  CUDACALL(cudaFree(xs.x));
}

struct xs xs_new_from_host(int n, struct x const dat[])
{
  struct xs xs = xs_new(n);
  CUDACALL(cudaMemcpy(xs.x, dat, sizeof(struct x) * n, cudaMemcpyHostToDevice));
  return xs;
}

struct ss ss_new(int n)
{
  struct ss ss;
  ss.n = n;
  CUDACALL(cudaMalloc(&ss.s, sizeof(double) * n));
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
  CUDALAUNCH(ss_new_const_0, n, (ss, v));
  return ss;
}

void ss_free(struct ss ss)
{
  CUDACALL(cudaFree(ss.s));
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
  CUDALAUNCH(ss_gt_0, a.n, (gts, a, b));
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
  CUDALAUNCH(compute_areas_0, fvs.nverts, (as, xs, fvs));
  return as;
}

void xs_to_host(struct xs* xs)
{
  struct x* tmp = (struct x*) malloc(sizeof(struct x) * xs->n);
  CUDACALL(cudaMemcpy(tmp, xs->x, sizeof(struct x) * xs->n, cudaMemcpyDeviceToHost));
  CUDACALL(cudaFree(xs->x));
  xs->x = tmp;
}

void xs_to_device(struct xs* xs)
{
  struct x* tmp;
  CUDACALL(cudaMalloc(&tmp, sizeof(struct x) * xs->n));
  CUDACALL(cudaMemcpy(tmp, xs->x, sizeof(struct x) * xs->n, cudaMemcpyHostToDevice));
  free(xs->x);
  xs->x = tmp;
}
