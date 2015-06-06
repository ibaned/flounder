#include "refine.h"
#include "size.h"
#include "vtk.h"
#include <stdio.h>

double size_fun(struct x x)
{
  (void)x;
  return 0.05 + (.3) * x.x[0];
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
  char buf[64];
  for (int i = 0; i < 3; ++i) {
    printf("round %d\n", i );
    struct ss dss = gen_size_field(fvs, xs, size_fun);
    struct rgraph fvs2;
    struct xs xs2;
    refine(fvs, xs, dss, &fvs2, &xs2);
    ss_free(dss);
    rgraph_free(fvs);
    xs_free(xs);
    fvs = fvs2;
    xs = xs2;
    sprintf(buf, "out_%d.vtk", i);
    write_vtk(buf, fvs, xs);
  }
  xs_free(xs);
  rgraph_free(fvs);
  return 0;
}
