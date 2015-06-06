#include "refine.h"
#include "size.h"
#include "vtk.h"
#include <stdio.h>

double size_fun(struct x x)
{
  (void)x;
  return 1e-4 + (5e-3) * x.x[0];
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
  struct rgraph fvs = rgraph_new_from_dat(2, 3, fvs_dat);
  struct xs xs = xs_new_from_dat(4, x_dat);
  int done = 0;
  while (!done) {
    struct ss dss = gen_size_field(fvs, xs, size_fun);
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
  write_vtk("out.vtk", fvs, xs);
  xs_free(xs);
  rgraph_free(fvs);
  return 0;
}
