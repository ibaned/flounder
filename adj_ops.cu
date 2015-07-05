#include "adj_ops.cuh"
#include "mycuda.cuh"

static __device__ int const fevi[3][2] = {{0,1},{1,2},{2,0}};

static __global__ void compute_fes_0(struct rgraph fes,
    struct rgraph fvs, struct graph ves)
{
  int i = CUDAINDEX;
  if (i < fvs.nverts) {
    struct adj ve[2];
    ve[0] = adj_new_graph(ves);
    ve[1] = adj_new_graph(ves);
    int fv[3];
    rgraph_get(fvs, i, fv);
    int fe[3];
    for (int j = 0; j < 3; ++j) {
      int ev[2];
      for (int k = 0; k < 2; ++k)
        ev[k] = fv[fevi[j][k]];
      for (int k = 0; k < 2; ++k)
        graph_get(ves, ev[k], &ve[k]);
      adj_intersect(&ve[0], ve[1]);
      fe[j] = ve[0].e[0];
    }
    rgraph_set(fes, i, fe);
  }
}

struct rgraph compute_fes(struct rgraph fvs, struct graph ves)
{
  struct rgraph fes = rgraph_new(fvs.nverts, 3);
  CUDACALL(compute_fes_0, fvs.nverts, (fes, fvs, ves));
  return fes;
}
