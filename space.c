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

struct ss ss_new_const(int n, double v)
{
  struct ss ss = ss_new(n);
  for (int i = 0; i < ss.n; ++i)
    ss.s[i] = v;
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

struct ss compute_areas(struct xs xs, struct rgraph fvs)
{
  struct ss as = ss_new(fvs.nverts);
  for (int i = 0; i < fvs.nverts; ++i) {
    int fv[3];
    rgraph_get(fvs, i, fv);
    struct x fx[3];
    xs_get(xs, fv, 3, fx);
    as.s[i] = fx_area(fx);
  }
  return as;
}
