#include "graph_ops.h"

int main()
{
  int const dat[6] = {
    0,1,2,
    2,3,0
  };
  struct rgraph fvs = rgraph_new_from_dat(2, 3, dat);
  rgraph_free(fvs);
}
