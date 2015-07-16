#include "graph.cuh"
#include "mycuda.cuh"
#include <stdio.h>

int graph_nedges(struct graph const g)
{
  int nedges;
  CUDACALL(cudaMemcpy(&nedges, g.off.i + g.nverts, sizeof(int), cudaMemcpyDeviceToHost));
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
  fprintf(stderr, "starting graph_new...\n");
  struct graph g;
  int nv = g.nverts = s.nverts;
  fprintf(stderr, "starting exscan...\n");
  g.off = ints_exscan(s.deg);
  fprintf(stderr, "done exscan...\n");
  int ne = graph_nedges(g);
  g.adj = ints_new(ne);
  fprintf(stderr, "starting ints_max...\n");
  g.max_deg = ints_max(s.deg);
  fprintf(stderr, "done ints_max...\n");
  ints_free(s.deg);
  fprintf(stderr, "done graph_new...\n");
  return g;
}

void graph_free(struct graph g)
{
  ints_free(g.off);
  ints_free(g.adj);
}
