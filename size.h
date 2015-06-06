#ifndef SIZE_H
#define SIZE_H

#include "rgraph.h"
#include "space.h"

struct ss gen_size_field(struct rgraph fvs, struct xs xs,
    double (*fun)(struct x));

#endif
