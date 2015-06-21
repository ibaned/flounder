#include "space.h"
#include <stdlib.h>
#include <stdio.h>

struct xs xs_new(int n)
{
  struct xs xs;
  xs.n = n;
  xs.x = malloc(sizeof(struct x) * n);
  return xs;
}

void xs_free(struct xs xs)
{
  free(xs.x);
}

struct xs xs_new_from_dat(int n, struct x const dat[])
{
  struct xs xs = xs_new(n);
  xs.n = n;
  for (int i = 0; i < n; ++i)
    xs.x[i] = dat[i];
  return xs;
}

void xs_print(struct xs xs)
{
  printf("coords (%d)\n", xs.n);
  for (int i = 0; i < xs.n; ++i)
    printf("%d: %f %f\n", i,
        xs.x[i].x[0],
        xs.x[i].x[1]);
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

void ss_print(struct ss ss)
{
  printf("scalars (%d)\n", ss.n);
  for (int i = 0; i < ss.n; ++i)
    printf("%d: %f\n", i, ss.s[i]);
}

struct ints ss_gt(struct ss a, struct ss b)
{
  struct ints gts = ints_new(a.n);
  for (int i = 0; i < a.n; ++i)
    gts.i[i] = (a.s[i] > b.s[i]);
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
