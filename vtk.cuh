#ifndef VTK_H
#define VTK_H

#include "rgraph.cuh"
#include "space.cuh"

void write_vtk(const char* filename, struct rgraph* fvs, struct xs* xs);

#endif
