#include "graph_ops.h"
#include <stdio.h>

int main()
{
  int const dat[6] = {
    0,1,2,
    2,3,0
  };
  struct rgraph fvs = rgraph_new_from_dat(2, 3, dat);
  printf("\nfvs:\n");
  rgraph_print(fvs);
  struct graph vfs = rgraph_invert(fvs);
  printf("\nvfs:\n");
  graph_print(vfs);
  graph_free(vfs);
  rgraph_free(fvs);
}
