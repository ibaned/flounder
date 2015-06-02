#include "graph.h"
#include <stdlib.h>
#include <stdio.h>

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
  g.max_deg = 0;
  for (int i = 0; i < nv; ++i)
    if (s.degrees[i] > g.max_deg)
      g.max_deg = s.degrees[i];
  free(s.degrees);
  return g;
}

void graph_free(struct graph g)
{
  free(g.offsets);
  free(g.adjacent);
}

void graph_print(struct graph g)
{
  printf("graph %d verts\n", g.nverts);
  for (int i = 0; i < g.nverts; ++i) {
    printf("%d:", i);
    int o = g.offsets[i];
    for (int j = 0; j < graph_deg(g, i); ++j)
      printf(" %d", g.adjacent[o + j]);
    printf("\n");
  }
}
