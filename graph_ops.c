#include "graph_ops.h"
#include <stdlib.h>

struct graph rgraph_invert(struct rgraph rg)
{
  int nverts = rgraph_max_adj(rg) + 1;
  struct graph_spec s = graph_spec_new(nverts);
  for (int i = 0; i < nverts; ++i)
    s.degrees[i] = 0;
  int* a = malloc(sizeof(int) * rg.degree);
  for (int i = 0; i < rg.nverts; ++i) {
    rgraph_get(rg, i, a);
    for (int j = 0; j < rg.degree; ++j)
      s.degrees[a[j]]++;
  }
  struct graph g = graph_new(s);
  int* at = malloc(sizeof(int) * nverts);
  for (int i = 0; i < nverts; ++i)
    at[i] = g.offsets[i];
  for (int i = 0; i < rg.nverts; ++i) {
    rgraph_get(rg, i, a);
    for (int j = 0; j < rg.degree; ++j) {
      int av = a[j];
      g.adjacent[at[av]++] = i;
    }
  }
  free(a);
  free(at);
  return g;
}

struct graph graph_rgraph_transit(struct graph g, struct rgraph rg)
{
  struct adj ga = adj_new(graph_max_deg(g));
  struct adj ra = adj_new(rg.degree);
  ra.n = rg.degree;
  struct adj ta = adj_new(ga.c * ra.c);
  struct graph_spec s = graph_spec_new(g.nverts);
  for (int i = 0; i < s.nverts; ++i)
    s.degrees[i] = 0;
  for (int i = 0; i < g.nverts; ++i) {
    graph_get(g, i, &ga);
    rgraph_get(rg, ga.e[0], ta.e);
    ta.n = rg.degree;
    for (int j = 1; j < ga.n; ++j) {
      rgraph_get(rg, ga.e[j], ra.e);
      adj_unite(&ta, ra);
    }
    s.degrees[i] = ta.n - 1;
  }
  struct graph tg = graph_new(s);
  for (int i = 0; i < g.nverts; ++i) {
    graph_get(g, i, &ga);
    rgraph_get(rg, ga.e[0], ta.e);
    ta.n = rg.degree;
    for (int j = 1; j < ga.n; ++j) {
      rgraph_get(rg, ga.e[j], ra.e);
      adj_unite(&ta, ra);
    }
    adj_remove(&ta, i);
    graph_set(g, i, ta);
  }
  adj_free(ga);
  adj_free(ra);
  adj_free(ta);
  return tg;
}

struct rgraph graph_bridge(struct graph g)
{
  int nhe = graph_nedges(g);
  assert(nhe % 2 == 0);
  int ne = nhe / 2;
  struct rgraph rg = rgraph_new(ne, 2);
  struct adj a = adj_new(graph_max_deg(g));
  int ra[2];
  int k = 0;
  for (int i = 0; i < g.nverts; ++i) {
    ra[0] = i;
    graph_get(g, i, &a);
    for (int j = 0; j < a.n; ++j)
      if (i < a.e[j]) {
        ra[1] = a.e[j];
        rgraph_set(rg, k++, ra);
      }
  }
  adj_free(a);
  return rg;
}
