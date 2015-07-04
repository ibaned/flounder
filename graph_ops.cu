#include "graph_ops.h"
#include <stdlib.h>
#include <stdio.h>

struct graph rgraph_invert(struct rgraph rg)
{
  int nverts = rgraph_max_adj(rg) + 1;
  struct graph_spec s = graph_spec_new(nverts);
  ints_zero(s.deg);
  struct adj a = adj_new_rgraph(rg);
  a.n = rg.degree;
  for (int i = 0; i < rg.nverts; ++i) {
    rgraph_get(rg, i, a.e);
    for (int j = 0; j < a.n; ++j)
      s.deg.i[a.e[j]]++;
  }
  struct graph g = graph_new(s);
  struct ints at = ints_new(nverts);
  ints_copy(at, g.off);
  for (int i = 0; i < rg.nverts; ++i) {
    rgraph_get(rg, i, a.e);
    for (int j = 0; j < a.n; ++j) {
      int av = a.e[j];
      g.adj.i[at.i[av]] = i;
      at.i[av]++;
    }
  }
  ints_free(at);
  adj_free(a);
  return g;
}

struct graph graph_rgraph_transit(struct graph g, struct rgraph rg)
{
  struct graph_spec s = graph_spec_new(g.nverts);
  ints_zero(s.deg);
  struct adj ga = adj_new_graph(g);
  struct adj ra = adj_new_rgraph(rg);
  struct adj ta = adj_new(ga.c * ra.c);
  for (int i = 0; i < g.nverts; ++i) {
    graph_get(g, i, &ga);
    rgraph_get(rg, ga.e[0], ta.e);
    ta.n = rg.degree;
    for (int j = 1; j < ga.n; ++j) {
      rgraph_get(rg, ga.e[j], ra.e);
      adj_unite(&ta, ra);
    }
    s.deg.i[i] = ta.n - 1;
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
    graph_set(tg, i, ta);
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
  struct adj a = adj_new_graph(g);
  struct ints naes = ints_new(g.nverts);
  for (int i = 0; i < g.nverts; ++i) {
    graph_get(g, i, &a);
    int nae = 0;
    for (int j = 0; j < a.n; ++j)
      if (i < a.e[j])
        nae++;
    naes.i[i] = nae;
  }
  struct ints os = ints_exscan(naes);
  ints_free(naes);
  int ra[2];
  for (int i = 0; i < g.nverts; ++i) {
    ra[0] = i;
    graph_get(g, i, &a);
    int k = os.i[i];
    for (int j = 0; j < a.n; ++j)
      if (i < a.e[j]) {
        ra[1] = a.e[j];
        rgraph_set(rg, k++, ra);
      }
  }
  ints_free(os);
  adj_free(a);
  return rg;
}
