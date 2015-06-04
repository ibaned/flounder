#ifndef RGRAPH_H
#define RGRAPH_H

#include "ints.h"
#include "adj.h"

struct rgraph {
  int nverts;
  int degree;
  struct ints adj;
};

struct rgraph rgraph_new(int nverts, int degree);
void rgraph_free(struct rgraph g);

static inline void rgraph_set(struct rgraph g, int i, int const a[])
{
  int* p = g.adj.i + i * g.degree;
  int* e = p + g.degree;
  while (p < e)
    *p++ = *a++;
}

static inline void rgraph_get(struct rgraph g, int i, int a[])
{
  int* p = g.adj.i + i * g.degree;
  int* e = p + g.degree;
  while (p < e)
    *a++ = *p++;
}

int rgraph_max_adj(struct rgraph g);
struct rgraph rgraph_new_from_dat(int nverts, int degree, int const dat[]);
void rgraph_print(struct rgraph g);

struct adj adj_new_rgraph(struct rgraph g);

#endif
