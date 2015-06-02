#include "adj.h"
#include <stdlib.h>

struct adj adj_new(int c)
{
  struct adj a;
  a.n = 0;
  a.c = c;
  a.e = malloc(sizeof(int) * c);
  return a;
}

void adj_free(struct adj a)
{
  free(a.e);
}

