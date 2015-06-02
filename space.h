#ifndef SPACE_H
#define SPACE_H

#include "bits.h"
#include "rgraph.h"

/* how many different things can we call "x"... */

struct x {
  double x[2];
};

struct fx {
  struct x x[3];
};

struct xs {
  struct x* x;
};

struct ss {
  int n;
  double* s;
};

struct xs xs_new(int n);
void xs_free(struct xs xs);

static inline struct fx fx_get(struct xs xs, struct rgraph fvs, int i)
{
  struct fx fx;
  int fv[3];
  rgraph_get(fvs, i, fv);
  for (int j = 0; j < 3; ++j)
    fx.x[j] = xs.x[fv[i]];
  return fx;
}

struct ss ss_new(int n);
void ss_free(struct ss ss);

struct bits ss_gt(struct ss a, struct ss b);

struct ss compute_areas(struct xs xs, struct rgraph fvs);

#endif
