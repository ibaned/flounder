#ifndef ADJ_H
#define ADJ_H

#include <assert.h>

struct adj {
  int n;
  int c;
  int* e;
};

struct adj adj_new(int c);
void adj_free(struct adj a);

static inline int adj_find(struct adj a, int e)
{
  for (int i = 0; i < a.n; ++i)
    if (a.e[i] == e)
      return i;
  return -1;
}

static inline int adj_has(struct adj a, int e)
{
  return adj_find(a, e) != -1;
}

static inline void adj_unite(struct adj* a, struct adj with)
{
  int j = a->n;
  for (int i = 0; i < with.n; ++i)
    if (!adj_has(*a, with.e[i]))
      a->e[j++] = with.e[i];
  a->n = j;
}

static inline void adj_intersect(struct adj* a, struct adj with)
{
  int j = 0;
  for (int i = 0; i < a->n; ++i)
    if (adj_has(with, a->e[i]))
      a->e[j++] = a->e[i];
  a->n = j;
}

static inline void adj_remove(struct adj* a, int e)
{
  int i = adj_find(*a, e);
  assert(i >= 0);
  a->n--;
  for (; i < a->n; ++i)
    a->e[i] = a->e[i + 1];
}

void debug_adj(struct adj a);

#endif
