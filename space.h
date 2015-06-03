#ifndef SPACE_H
#define SPACE_H

#include "bits.h"
#include "rgraph.h"

/* how many different things can we call "x"... */

struct x {
  double x[2];
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

struct ss ss_new(int n);
void ss_free(struct ss ss);

struct bits ss_gt(struct ss a, struct ss b);

static inline double x_cross(struct x a, struct x b)
{
  return a.x[0] * b.x[1] - a.x[1] * b.x[0];
}

static inline struct x x_sub(struct x a, struct x b)
{
  struct x c;
  c.x[0] = a.x[0] - b.x[0];
  c.x[1] = a.x[1] - b.x[1];
  return c;
}

static inline struct x x_add(struct x a, struct x b)
{
  struct x c;
  c.x[0] = a.x[0] + b.x[0];
  c.x[1] = a.x[1] + b.x[1];
  return c;
}

static inline struct x x_div(struct x a, double b)
{
  struct x c;
  c.x[0] = a.x[0] / b;
  c.x[1] = a.x[1] / b;
  return c;
}

static inline struct x x_avg(struct x a, struct x b)
{
  return x_div(x_add(a, b), 2);
}

static inline double x_dot(struct x a, struct x b)
{
  return a.x[0] * b.x[0] + a.x[1] * b.x[1];
}

static inline double fx_area(struct x x[3])
{
  return x_cross(x_sub(x[1], x[0]), x_sub(x[2], x[0])) / 2;
}

static inline double fx_qual(struct x x[3])
{
  double s;
  struct x v;
  v = x_sub(x[1], x[0]);
  s  = x_dot(v, v);
  v = x_sub(x[2], x[1]);
  s += x_dot(v, v);
  v = x_sub(x[0], x[2]);
  s += x_dot(v, v);
  return fx_area(x) / s; /* todo: normalize */
}

static inline void xs_get(struct xs xs, int const v[], int nv, struct x x[])
{
  for (int i = 0; i < nv; ++i)
    x[i] = xs.x[v[i]];
}

struct ss compute_areas(struct xs xs, struct rgraph fvs);

#endif
