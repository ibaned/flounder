#include "rgraph.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct rgraph rgraph_new(int nverts, int degree)
{
  struct rgraph g;
  g.nverts = nverts;
  g.degree = degree;
  g.adjacent = malloc(sizeof(int) * nverts * degree);
  return g;
}

void rgraph_free(struct rgraph g)
{
  free(g.adjacent);
}

int rgraph_max_adj(struct rgraph g)
{
  int max = g.adjacent[0];
  for (int i = 1; i < g.nverts * g.degree; ++i)
    if (g.adjacent[i] > max)
      max = g.adjacent[i];
  return max;
}

struct rgraph rgraph_new_from_dat(int nverts, int degree, int const dat[])
{
  struct rgraph g = rgraph_new(nverts, degree);
  memcpy(g.adjacent, dat, sizeof(int) * nverts * degree);
  return g;
}

void rgraph_print(struct rgraph g)
{
  printf("rgraph %d verts degree %d\n", g.nverts, g.degree);
  for (int i = 0; i < g.nverts; ++i) {
    printf("%d:", i);
    for (int j = 0; j < g.degree; ++j)
      printf(" %d", g.adjacent[i * g.degree + j]);
    printf("\n");
  }
}
