#include "graph.h"
#include <stdio.h>

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

void graph_print(struct graph g)
{
  printf("graph %d verts\n", g.nverts);
  for (int i = 0; i < g.nverts; ++i) {
    printf("%d:", i);
    int o = g.off.i[i];
    for (int j = 0; j < graph_deg(g, i); ++j)
      printf(" %d", g.adj.i[o + j]);
    printf("\n");
  }
}

struct adj adj_new_graph(struct graph g)
{
  return adj_new(g.max_deg);
}
