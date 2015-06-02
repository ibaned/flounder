#include "space.h"
#include <stdlib.h>

struct xs xs_new(int n)
{
  struct xs xs;
  xs.x = malloc(sizeof(struct x) * n);
  return xs;
}

void xs_free(struct xs xs)
{
  free(xs.x);
}

