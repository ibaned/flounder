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

struct ss ss_new(int n)
{
  struct ss ss;
  ss.n = n;
  ss.s = malloc(sizeof(double) * n);
  return ss;
}

void ss_free(struct ss ss)
{
  free(ss.s);
}

struct bits ss_gt(struct ss a, struct ss b)
{
  struct bits gts = bits_new(a.n);
  for (int i = 0; i < a.n; ++i) {
    if (a.s[i] > b.s[i])
      bits_set(gts, i);
    else
      bits_clear(gts, i);
  }
  return gts;
}

static inline double fx_area(struct fx fx)
{
  return (fx.x[0].x[0] * fx.x[1].x[1] -
          fx.x[0].x[1] * fx.x[1].x[0]) / 2;
}

struct ss compute_areas(struct xs xs, struct rgraph fvs)
{
  struct ss as = ss_new(fvs.nverts);
  for (int i = 0; i < fvs.nverts; ++i)
    as.s[i] = fx_area(fx_get(xs, fvs, i));
  return as;
}
