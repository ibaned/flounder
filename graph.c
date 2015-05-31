#include "graph.h"
#include <stdlib.h>

struct graph_spec graph_spec_new(int nverts)
{
  struct graph_spec s;
  s.nverts = nverts;
  s.degrees = malloc(sizeof(int) * nverts);
  return s;
}

struct graph graph_new(struct graph_spec s)
{
  struct graph g;
  int nv = g.nverts = s.nverts;
  g.offsets = malloc(sizeof(int) * (nv + 1));
  g.offsets[0] = 0;
  for (int i = 0; i < nv; ++i)
    /* exclusive scan summation */
    g.offsets[i + 1] = g.offsets[i] + s.degrees[i];
  int ne = graph_nedges(g);
  g.adjacent = malloc(sizeof(int) * ne);
  free(s.degrees);
  return g;
}

void graph_free(struct graph g)
{
  free(g.offsets);
  free(g.adjacent);
}

int graph_max_deg(struct graph g)
{
  int max = graph_deg(g, 0);
  for (int i = 1; i < g.nverts; ++i) {
    int deg = graph_deg(g, i);
    if (deg > max)
      max = deg;
  }
  return max;
}
