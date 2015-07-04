#ifndef SIZE_H
#define SIZE_H

#include "rgraph.cuh"
#include "space.cuh"

typedef double (*sizefun)(struct x);

struct ss gen_size_field(struct rgraph fvs, struct xs xs,
    double (*fun)(struct x));

#endif
