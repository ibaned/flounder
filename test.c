#include "refine.h"
#include "size.h"
#include "vtk.h"
#include <stdio.h>
#include <omp.h>

double linear(struct x x)
{
  return 1e-5 + (1e-3) * x.x[0];
}

double ring(struct x x)
{
  static struct x const c = {{.5,.5}};
  double r = x_dist(x, c);
  double d = 4 * fabs(r - .25);
  return 5e-7 + d * 5e-5;
}

int main()
{
  int const fvs_dat[6] = {
    0,1,2,
    2,3,0
  };
  struct x const x_dat[4] = {
    {0,0},
    {1,0},
    {1,1},
    {0,1}
  };
  omp_set_dynamic(0);
  printf("%d OpenMP threads\n", omp_get_num_threads());
  struct rgraph fvs = rgraph_new_from_dat(2, 3, fvs_dat);
  struct xs xs = xs_new_from_dat(4, x_dat);
  int done = 0;
  double t0 = omp_get_wtime();
  while (!done) {
    struct ss dss = gen_size_field(fvs, xs, ring);
    struct rgraph fvs2;
    struct xs xs2;
    refine(fvs, xs, dss, &fvs2, &xs2);
    if (fvs.nverts == fvs2.nverts)
      done = 1;
    ss_free(dss);
    rgraph_free(fvs);
    xs_free(xs);
    fvs = fvs2;
    xs = xs2;
  }
  double t1 = omp_get_wtime();
  printf("total runtime %f seconds\n", t1 - t0);
  write_vtk("out.vtk", fvs, xs);
  xs_free(xs);
  rgraph_free(fvs);
  return 0;
}
