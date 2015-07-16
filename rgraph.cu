#include "rgraph.cuh"

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

struct rgraph rgraph_new_from_host(int nverts, int degree, int const dat[])
{
  struct rgraph g = rgraph_new(nverts, degree);
  ints_from_host(g.adj, dat);
  return g;
}

void rgraph_to_host(struct rgraph* g)
{
  ints_to_host(&g->adj);
}

void rgraph_to_device(struct rgraph* g)
{
  ints_to_device(&g->adj);
}
