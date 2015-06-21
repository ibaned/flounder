#include "adj.h"
#include <stdlib.h>
#include <stdio.h>

struct adj adj_new(int c)
{
  struct adj a;
  a.n = 0;
  a.c = c;
  a.e = (int*) malloc(sizeof(int) * c);
  return a;
}

void adj_free(struct adj a)
{
  free(a.e);
}

void debug_adj(struct adj a)
{
  assert(a.n <= a.c);
  for (int i = 0; i < a.n; ++i)
    printf(" %d", a.e[i]);
  printf("\n");
}
