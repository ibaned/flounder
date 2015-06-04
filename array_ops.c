#include "array_ops.h"

void ints_exscan(int const* a, int n, int* o)
{
  o[0] = 0;
  for (int i = 0; i < n; ++i)
    o[i + 1] = o[i] + a[i];
}
