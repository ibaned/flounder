#include "refine.h"
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
  printf("\nfvs:\n");
  rgraph_print(fvs);
  struct xs xs = xs_new_from_dat(4, x_dat);
  printf("\nxs:\n");
  xs_print(xs);
  struct ss dss = ss_new_const(2, 0.25);
  printf("\ndss:\n");
  ss_print(dss);
  ss_free(dss);
  xs_free(xs);
  rgraph_free(fvs);
  return 0;
}
