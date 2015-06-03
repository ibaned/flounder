#include "refine.h"
#include <float.h>

static struct bits mark_fes(struct rgraph fes, struct bits bfs, int ne)
{
  struct bits ecss = bits_new(ne);
  int fe[3];
  for (int i = 0; i < ne; ++i)
    bits_clear(ecss, i);
  for (int i = 0; i < fes.nverts; ++i) {
    if (!bits_get(bfs, i))
      continue;
    rgraph_get(fes, i, fe);
    for (int j = 0; j < 3; ++j)
      bits_set(ecss, fe[j]);
  }
  return ecss;
}

static struct ss compute_split_quals(struct bits ecss, struct rgraph evs,
    struct graph efs, struct rgraph fvs, struct xs xs)
{
  struct ss eqs = ss_new(efs.nverts);
  struct adj ef = adj_new(2);
  for (int i = 0; i < ecss.n; ++i) {
    if (!bits_get(ecss, i))
      continue;
    int ev[2];
    rgraph_get(evs, i, ev);
    struct x ex[2];
    xs_get(xs, ev, 2, ex);
    struct x mid = x_avg(ex[0], ex[1]);
    graph_get(efs, i, &ef);
    double wq = DBL_MAX;
    for (int j = 0; j < ef.n; ++j) {
      int f = ef.e[j];
      int fv[3];
      rgraph_get(fvs, f, fv);
      struct x fx[3];
      xs_get(xs, fv, 3, fx);
      for (int k = 0; k < 2; ++k) {
        for (int l = 0; l < 3; ++l)
          if (fv[l] == ev[k])
            fx[l] = mid;
        double q = fx_qual(fx);
        if (q < wq)
          wq = q;
      }
    }
    eqs.s[i] = wq;
  }
  adj_free(ef);
  return eqs;
}

static struct bits compute_best_indset(struct bits ecss, struct graph ees,
    struct ss eqs)
{
  struct bits ewss = bits_new(ecss.n);
  for (int i = 0; i < ewss.n; ++i)
    bits_clear(ewss, i);
  struct bits enss = bits_new(ecss.n);
  for (int i = 0; i < enss.n; ++i)
    bits_clear(enss, i);
  struct adj ee = adj_new(ees.max_deg);
  int done = 0;
  for (int iters = 0; !done; ++iters) {
    for (int i = 0; i < ecss.n; ++i) {
      if (!bits_get(ecss, i))
        continue;
      if (bits_get(ewss, i))
        continue;
      if (bits_get(enss, i))
        continue;
      double q = eqs.s[i];
      graph_get(ees, i, ee);
      for (int j = 0; j < ee.n; ++j)
        if (bits_get(ewss, ee.e[j])) {
          bits_set(enss, i);
          continue;
        }
      int local_max = 1;
      for (int j = 0; j < ee.n; ++j) {
        double oq = eqs.s[ee.e[j]];
        if (oq >= q) /* todo: tiebreaker ? */
          local_max = 0;
      }
      if (local_max)
        bits_set(ewss, i);
    }
  }
  bits_free(enss);
  return ewss;
}

void refine(struct rgraph fvs, struct xs xs, struct ss dss,
    struct rgraph* fvs2, struct xs* xs2)
{
  struct graph vfs = rgraph_invert(fvs);
  struct graph vvs = graph_rgraph_transit(vfs, fvs);
  struct rgraph evs = graph_bridge(vvs);
  struct graph ves = rgraph_invert(evs);
  struct rgraph fes = compute_fes(fvs, ves);
  struct graph efs = rgraph_invert(fes);
  struct ss as = compute_areas(xs, fvs);
  struct bits bfs = ss_gt(as, dss);
  struct bits ecss = mark_fes(fes, bfs, efs.nverts);
  struct ss eqs = compute_split_quals(ecss, evs, efs, fvs, xs);
  struct graph ees = graph_rgraph_transit(efs, fes);
  struct bits ewss = compute_best_indset(ecss, ees, eqs);
  bits_free(ewss);
  ss_free(eqs);
  bits_free(ecss);
  bits_free(bfs);
  ss_free(as);
  graph_free(efs);
  rgraph_free(fes);
  graph_free(ves);
  rgraph_free(evs);
  graph_free(vvs);
  graph_free(vfs);
}

