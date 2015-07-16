#include "size.cuh"
#include "mycuda.cuh"

__device__ double linear(struct x x)
{
  return 1e-5 + (1e-3) * x.x[0];
}

__device__ double ring(struct x x)
{
  struct x const c = {{.5,.5}};
  double r = x_dist(x, c);
  double d = 4 * fabs(r - .25);
  return 5e-7 + d * 5e-5;
}

__device__ double sinusoid(struct x x)
{
  double s = cos(x.x[0] * 8.0 * M_PI) / 4.0 + 1.0 / 2.0;
  double d = fabs(x.x[1] - s);
  return 1e-6 + d * 1e-4;
}

__device__ double gold_sinusoid(struct x x)
{
  double s = cos(x.x[0] * 8.0 * M_PI) / 4.0 + 1.0 / 2.0;
  double d = fabs(x.x[1] - s);
  return 1e-7 + d * 1e-5;
}

static __global__ void gen_size_field_0(struct ss dss,
    struct rgraph fvs, struct xs xs)
{
  int i = CUDAINDEX;
  if (i < dss.n) {
    int fv[3];
    rgraph_get(fvs, i, fv);
    struct x fx[3];
    xs_get(xs, fv, 3, fx);
    struct x c = fx_center(fx);
    dss.s[i] = linear(c);
  }
}

struct ss gen_size_field(struct rgraph fvs, struct xs xs)
{
  struct ss dss = ss_new(fvs.nverts);
  CUDALAUNCH(gen_size_field_0, dss.n, (dss, fvs, xs));
  return dss;
}
