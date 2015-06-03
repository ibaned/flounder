#ifndef REFINE_H
#define REFINE_H

#include "space.h"

void refine(struct rgraph fvs, struct xs xs, struct ss dss,
    struct rgraph* fvs2, struct xs* xs2);

#endif
