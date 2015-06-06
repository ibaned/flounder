#include "refine.h"
#include "vtk.h"
#include <stdio.h>

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
  for (int i = 0; i < 2; ++i) {
    printf("round %d\n", i );
    struct ss dss = ss_new_const(fvs.nverts, 0.125);
    struct rgraph fvs2;
    struct xs xs2;
    refine(fvs, xs, dss, &fvs2, &xs2);
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
