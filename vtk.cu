#include "vtk.cuh"
#include <stdio.h>

void write_vtk(const char* filename, struct rgraph* fvs, struct xs* xs)
{
  rgraph_to_host(fvs);
  xs_to_host(xs);
  FILE* f = fopen(filename, "w");
  fprintf(f, "# vtk DataFile Version 3.0\n");
  fprintf(f, "Refined Mesh\n");
  fprintf(f, "ASCII\n");
  fprintf(f, "DATASET UNSTRUCTURED_GRID\n");
  fprintf(f, "POINTS %d double\n", xs->n);
  for (int i = 0; i < xs->n; ++i)
    fprintf(f, "%f %f 0\n",
        xs->x[i].x[0],
        xs->x[i].x[1]);
  fprintf(f,"CELLS %d %d\n", fvs->nverts, fvs->nverts * 4);
  for (int i = 0; i < fvs->nverts; ++i) {
    int fv[3];
    rgraph_get_host(*fvs, i, fv);
    fprintf(f, "3 %d %d %d\n", fv[0], fv[1], fv[2]);
  }
  fprintf(f,"CELL_TYPES %d\n", fvs->nverts);
  for (int i = 0; i < fvs->nverts; ++i)
    fprintf(f, "5\n");
  fclose(f);
  xs_to_device(xs);
  rgraph_to_device(fvs);
}

