#include "graph_ops.h"
#include "adj_ops.h"
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
  struct graph vvs = graph_rgraph_transit(vfs, fvs);
  printf("\nvvs:\n");
  graph_print(vvs);
  struct rgraph evs = graph_bridge(vvs);
  printf("\nevs:\n");
  rgraph_print(evs);
  struct graph ves = rgraph_invert(evs);
  printf("\nves:\n");
  graph_print(ves);
  struct rgraph fes = compute_fes(fvs, ves);
  printf("\nfes:\n");
  rgraph_print(fes);
  rgraph_free(fes);
  graph_free(ves);
  rgraph_free(evs);
  graph_free(vvs);
  graph_free(vfs);
  rgraph_free(fvs);
}
