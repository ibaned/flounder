#ifndef VTK_H
#define VTK_H

#include "rgraph.h"
#include "space.h"

void write_vtk(const char* filename, struct rgraph fvs, struct xs xs);

#endif
