#include "rgraph.h"
#include <stdio.h>

struct rgraph rgraph_new(int nverts, int degree)
{
  struct rgraph g;
  g.nverts = nverts;
  g.degree = degree;
  g.adj = ints_new(nverts * degree);
  return g;
}

void rgraph_free(struct rgraph g)
{
  ints_free(g.adj);
}

int rgraph_max_adj(struct rgraph g)
{
  return ints_max(g.adj);
}

struct rgraph rgraph_new_from_dat(int nverts, int degree, int const dat[])
{
  struct rgraph g = rgraph_new(nverts, degree);
  ints_from_dat(g.adj, dat);
  return g;
}

void rgraph_print(struct rgraph g)
{
  printf("rgraph %d verts degree %d\n", g.nverts, g.degree);
  for (int i = 0; i < g.nverts; ++i) {
    printf("%d:", i);
    for (int j = 0; j < g.degree; ++j)
      printf(" %d", g.adj.i[i * g.degree + j]);
    printf("\n");
  }
}

struct adj adj_new_rgraph(struct rgraph g)
{
  struct adj a = adj_new(g.degree);
  a.n = g.degree;
  return a;
}
