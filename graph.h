#ifndef GRAPH_H
#define GRAPH_H

#include "adj.h"
#include <assert.h>

struct graph_spec {
  int nverts;
  int padding_;
  int* degrees;
};

struct graph {
  int nverts;
  int padding_;
  int* offsets;
  int* adjacent;
};

struct graph_spec graph_spec_new(int nverts);
struct graph graph_new(struct graph_spec s);
void graph_free(struct graph g);

static inline int graph_nedges(struct graph const g)
{
  return g.offsets[g.nverts];
}

static inline int graph_deg(struct graph g, int i)
{
  return g.offsets[i + 1] - g.offsets[i];
}

static inline void graph_get(struct graph g, int i, struct adj* a)
{
  int* p = g.adjacent + g.offsets[i];
  int* e = g.adjacent + g.offsets[i + 1];
  a->n = e - p;
  int* q = a->e;
  while (p < e)
    *q++ = *p++;
}

static inline void graph_set(struct graph g, int i, struct adj a)
{
  int* p = g.adjacent + g.offsets[i];
  int* e = g.adjacent + g.offsets[i + 1];
  int* q = a.e;
  while (p < e)
    *p++ = *q++;
}

int graph_max_deg(struct graph g);
void graph_print(struct graph g);

#endif
