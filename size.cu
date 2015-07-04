#include "size.cuh"

struct ss gen_size_field(struct rgraph fvs, struct xs xs,
    double (*fun)(struct x))
{
  struct ss dss = ss_new(fvs.nverts);
  for (int i = 0; i < dss.n; ++i) {
    int fv[3];
    rgraph_get(fvs, i, fv);
    struct x fx[3];
    xs_get(xs, fv, 3, fx);
    struct x c = fx_center(fx);
    dss.s[i] = fun(c);
  }
  return dss;
}
