#ifndef SPACE_H
#define SPACE_H

#include "graph.h"
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

double fx_area(struct fx fx)
{
  return (fx.x[0].x[0] * fx.x[1].x[1] -
          fx.x[0].x[1] * fx.x[1].x[0]) / 2;
}

#endif
