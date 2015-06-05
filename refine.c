#include "refine.h"
#include "graph_ops.h"
#include "adj_ops.h"
#include <float.h>
#include <stdio.h>

static struct ints mark_fes(struct graph efs, struct ints bfs)
{
  struct ints ecss = ints_new(efs.nverts);
  ints_zero(ecss);
  int fe[3];
  struct adj ef = adj_new(efs.max_deg);
  for (int i = 0; i < efs.nverts; ++i) {
    graph_get(efs, i, &ef);
    for (int j = 0; j < ef.n; ++j)
      if (bfs.i[ef.e[j]]) {
        ecss.i[i] = 1;
        break;
      }
  }
  adj_free(ef);
  return ecss;
}

static struct ss compute_split_quals(struct ints ecss, struct rgraph evs,
    struct graph efs, struct rgraph fvs, struct xs xs)
{
  struct ss eqs = ss_new(efs.nverts);
  struct adj ef = adj_new(2);
  for (int i = 0; i < ecss.n; ++i) {
    if (!ecss.i[i])
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
      for (int k = 0; k < 2; ++k) {
        xs_get(xs, fv, 3, fx);
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

static struct ints compute_best_indset(struct ints ecss, struct graph ees,
    struct ss eqs)
{
  struct ints ewss = ints_new(ecss.n);
  ints_zero(ewss);
  struct ints enss = ints_new(ecss.n);
  ints_zero(enss);
  struct adj ee = adj_new(ees.max_deg);
  int done = 0;
  for (int iter = 0; !done; ++iter) {
    done = 1;
    for (int i = 0; i < ecss.n; ++i) {
      if (!ecss.i[i])
        continue;
      if (ewss.i[i])
        continue;
      if (enss.i[i])
        continue;
      double q = eqs.s[i];
      graph_get(ees, i, &ee);
      for (int j = 0; j < ee.n; ++j)
        if (ewss.i[ee.e[j]]) {
          enss.i[i] = 1;
          goto next_edge;
        }
      int local_max = 1;
      for (int j = 0; j < ee.n; ++j) {
        if (enss.i[ee.e[j]])
          continue;
        double oq = eqs.s[ee.e[j]];
        if (oq >= q) /* todo: tiebreaker ? */
          local_max = 0;
      }
      if (local_max)
        ewss.i[i] = 1;
      else
        done = 0;
next_edge:
      continue;
    }
  }
  adj_free(ee);
  ints_free(enss);
  return ewss;
}

static void split_edges(struct rgraph fvs, struct xs xs,
    struct ints bfs, struct ints ewss, struct rgraph evs,
    struct graph efs,
    struct rgraph* pfvs2, struct xs* pxs2)
{
  struct ints fos = ints_exscan(bfs);
  struct ints eos = ints_exscan(ewss);
  int nsf = fos.i[bfs.n];
  int nse = eos.i[ewss.n];
  struct xs xs2 = xs_new(xs.n + nse);
  for (int i = 0; i < xs.n; ++i)
    xs2.x[i] = xs.x[i];
  for (int i = 0; i < ewss.n; ++i)
    if (ewss.i[i]) {
      int ev[2];
      rgraph_get(evs, i, ev);
      struct x ex[2];
      xs_get(xs, ev, 2, ex);
      struct x mid = x_avg(ex[0], ex[1]);
      xs2.x[xs.n + eos.i[i]] = mid;
    }
  *pxs2 = xs2;
  struct rgraph fvs2 = rgraph_new(fvs.nverts + nsf, 3);
  for (int i = 0; i < bfs.n; ++i)
    if (!bfs.i[i]) {
      int fv[3];
      rgraph_get(fvs, i, fv);
      rgraph_set(fvs2, i, fv);
    }
  struct adj ef = adj_new_graph(efs);
  for (int i = 0; i < ewss.n; ++i)
    if (ewss.i[i]) {
      int ev[2];
      rgraph_get(evs, i, ev);
      graph_get(efs, i, &ef);
      int sv = ewss.n + eos.i[i];
      for (int j = 0; j < ef.n; ++j) {
        int f = ef.e[j];
        int sf[2];
        sf[0] = f;
        sf[1] = fvs.nverts + fos.i[f];
        int fv[3];
        for (int k = 0; k < 2; ++k) {
          rgraph_get(fvs, f, fv);
          for (int l = 0; l < 3; ++l)
            if (fv[l] == ev[k])
              fv[l] = sv;
          rgraph_set(fvs2, sf[k], fv);
        }
      }
    }
  adj_free(ef);
  *pfvs2 = fvs2;
}

void refine(struct rgraph fvs, struct xs xs, struct ss dss,
    struct rgraph* fvs2, struct xs* xs2)
{
  struct graph vfs = rgraph_invert(fvs);
  printf("\nvfs:\n");
  graph_print(vfs);
  struct graph vvs = graph_rgraph_transit(vfs, fvs);
  printf("\nvvs:\n");
  graph_print(vvs);
  struct rgraph evs = graph_bridge(vvs);
  printf("\nevs:\n");
  rgraph_print(evs);
  struct graph ves = rgraph_invert(evs);
  printf("\nves:\n");
  graph_print(ves);
  struct rgraph fes = compute_fes(fvs, ves);
  printf("\nfes:\n");
  rgraph_print(fes);
  struct graph efs = rgraph_invert(fes);
  printf("\nefs:\n");
  graph_print(efs);
  struct ss as = compute_areas(xs, fvs);
  printf("\nas:\n");
  ss_print(as);
  struct ints bfs = ss_gt(as, dss);
  printf("\nbfs:\n");
  ints_print(bfs);
  struct ints ecss = mark_fes(efs, bfs);
  printf("\necss:\n");
  ints_print(ecss);
  struct ss eqs = compute_split_quals(ecss, evs, efs, fvs, xs);
  printf("\neqs:\n");
  ss_print(eqs);
  struct graph ees = graph_rgraph_transit(efs, fes);
  printf("\nees:\n");
  graph_print(ees);
  struct ints ewss = compute_best_indset(ecss, ees, eqs);
  printf("\newss:\n");
  ints_print(ewss);
  ints_free(ewss);
  graph_free(ees);
  ss_free(eqs);
  ints_free(ecss);
  ints_free(bfs);
  ss_free(as);
  graph_free(efs);
  rgraph_free(fes);
  graph_free(ves);
  rgraph_free(evs);
  graph_free(vvs);
  graph_free(vfs);
}

