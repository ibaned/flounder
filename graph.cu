#include "graph.cuh"

int graph_nedges(struct graph const g)
{
  int nedges;
  cudaMemcpy(&nedges, g.off.i + g.nverts, sizeof(int), cudaMemcpyDeviceToHost);
  return nedges;
}

struct graph_spec graph_spec_new(int nverts)
{
  struct graph_spec s;
  s.nverts = nverts;
  s.deg = ints_new(nverts);
  return s;
}

struct graph graph_new(struct graph_spec s)
{
  struct graph g;
  int nv = g.nverts = s.nverts;
  g.off = ints_exscan(s.deg);
  int ne = graph_nedges(g);
  g.adj = ints_new(ne);
  g.max_deg = ints_max(s.deg);
  ints_free(s.deg);
  return g;
}

void graph_free(struct graph g)
{
  ints_free(g.off);
  ints_free(g.adj);
}
