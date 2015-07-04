#include "rgraph.h"

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
  ints_from_host(g.adj, dat);
  return g;
}
