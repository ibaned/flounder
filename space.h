#ifndef SPACE_H
#define SPACE_H

#include "ints.h"
#include "rgraph.h"
#include <math.h>

/* how many different things can we call "x"... */

struct x {
  double x[2];
};

struct xs {
  int n;
  struct x* x;
};

struct ss {
  int n;
  double* s;
};

struct xs xs_new(int n);
void xs_free(struct xs xs);
struct xs xs_new_from_host(int n, struct x const dat[]);

struct ss ss_new(int n);
struct ss ss_new_const(int n, double v);
void ss_free(struct ss ss);

struct ints ss_gt(struct ss a, struct ss b);

static __device__ inline double x_cross(struct x a, struct x b)
{
  return a.x[0] * b.x[1] - a.x[1] * b.x[0];
}

static __device__ inline struct x x_sub(struct x a, struct x b)
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

static __device__ inline double x_dist(struct x a, struct x b)
{
  struct x ab = x_sub(b, a);
  return sqrt(x_dot(ab, ab));
}

static __device__ inline double fx_area(struct x const x[3])
{
  return x_cross(x_sub(x[1], x[0]), x_sub(x[2], x[0])) / 2;
}

static __device__ inline double fx_qual(struct x const x[3])
{
  double s;
  double lsq;
  struct x v;
  v = x_sub(x[1], x[0]);
  lsq = x_dot(v, v);
  s = lsq;
  v = x_sub(x[2], x[1]);
  lsq = x_dot(v, v);
  s += lsq;
  v = x_sub(x[0], x[2]);
  lsq = x_dot(v, v);
  s += lsq;
  double a = fx_area(x);
  return (3.0 * 4.0 / sqrt(3.0)) * (a / s);
}

static inline struct x fx_center(struct x const x[3])
{
  struct x c;
  c.x[0] = (x[0].x[0] + x[1].x[0] + x[2].x[0]) / 3;
  c.x[1] = (x[0].x[1] + x[1].x[1] + x[2].x[1]) / 3;
  return c;
}

static __device__ inline void xs_get(struct xs xs, int const v[], int nv, struct x x[])
{
  for (int i = 0; i < nv; ++i)
    x[i] = xs.x[v[i]];
}

struct ss compute_areas(struct xs xs, struct rgraph fvs);

#endif
