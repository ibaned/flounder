#include "size.cuh"
#include "mycuda.cuh"

static __global__ void gen_size_field_0(struct ss dss,
    struct rgraph fvs, struct xs xs, sizefun fun)
{
  int i = CUDAINDEX;
  if (i < dss.n) {
    int fv[3];
    rgraph_get(fvs, i, fv);
    struct x fx[3];
    xs_get(xs, fv, 3, fx);
    struct x c = fx_center(fx);
    dss.s[i] = fun(c);
  }
}

struct ss gen_size_field(struct rgraph fvs, struct xs xs, sizefun fun)
{
  struct ss dss = ss_new(fvs.nverts);
  CUDACALL(gen_size_field_0, dss.n, (dss, fvs, xs, fun));
  return dss;
}
