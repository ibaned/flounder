#include "adj.h"
#include <stdio.h>

void debug_adj(struct adj a)
{
  assert(a.n <= a.c);
  for (int i = 0; i < a.n; ++i)
    printf(" %d", a.e[i]);
  printf("\n");
}
