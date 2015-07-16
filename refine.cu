#include "refine.cuh"
#include "graph_ops.cuh"
#include "adj_ops.cuh"
#include "mycuda.cuh"
#if 0
#include <thrust/device_ptr.h>
#include <thrust/logical.h>
#endif
#include <float.h>
#include <stdio.h>

static struct ints mark_fes(struct graph efs, struct ints bfs) __attribute__((noinline));
static struct ss compute_split_quals(struct ints ecss, struct rgraph evs,
    struct graph efs, struct rgraph fvs, struct xs xs) __attribute__((noinline));
static struct ints compute_best_indset(struct ints ecss, struct graph ees,
    struct ss eqs) __attribute__((noinline));
static struct ints mark_split_faces(struct ints ewss, struct rgraph fes) __attribute__((noinline));
static struct xs split_edges(struct xs xs,
    struct ints ewss, struct rgraph evs) __attribute__((noinline));
static struct rgraph split_faces(struct rgraph fvs,
    struct ints fwss, struct ints ewss, struct rgraph evs,
    struct graph efs, int nv) __attribute__((noinline));

static __global__ void mark_fes_0(struct ints ecss, struct graph efs, struct ints bfs)
{
  int i = CUDAINDEX;
  if (i < efs.nverts) {
    struct adj ef = adj_new(efs.max_deg);
    graph_get(efs, i, &ef);
    for (int j = 0; j < ef.n; ++j)
      if (bfs.i[ef.e[j]]) {
        ecss.i[i] = 1;
        break;
      }
  }
}

static struct ints mark_fes(struct graph efs, struct ints bfs)
{
  struct ints ecss = ints_new(efs.nverts);
  ints_zero(ecss);
  CUDALAUNCH(mark_fes_0, efs.nverts, (ecss, efs, bfs));
  return ecss;
}

static __global__ void compute_split_quals_0(struct ss eqs, struct ints ecss,
    struct rgraph evs, struct graph efs, struct rgraph fvs, struct xs xs)
{
  int i = CUDAINDEX;
  if (i < ecss.n) {
    if (!ecss.i[i])
      return;
    struct adj ef = adj_new(2);
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
}

static struct ss compute_split_quals(struct ints ecss, struct rgraph evs,
    struct graph efs, struct rgraph fvs, struct xs xs)
{
  struct ss eqs = ss_new(efs.nverts);
  CUDALAUNCH(compute_split_quals_0, ecss.n, (eqs, ecss, evs, efs, fvs, xs));
  return eqs;
}

enum { WONT_SPLIT = 0, WILL_SPLIT = 1, COULD_SPLIT = 2 };

static __global__ void compute_best_indset_0(struct ints ewss, struct ints ecss)
{
  int i = CUDAINDEX;
  if (i < ecss.n) {
    if (ecss.i[i])
      ewss.i[i] = COULD_SPLIT;
    else
      ewss.i[i] = WONT_SPLIT;
  }
}

static __global__ void compute_best_indset_1(struct ints ewss_old, struct ints ewss,
    struct ints ecss, struct graph ees, struct ss eqs)
{
  int i = CUDAINDEX;
  if (i < ecss.n) {
    struct adj ee = adj_new_graph(ees);
    if (ewss_old.i[i] != COULD_SPLIT)
      return;
    double q = eqs.s[i];
    graph_get(ees, i, &ee);
    for (int j = 0; j < ee.n; ++j)
      if (ewss_old.i[ee.e[j]] == WILL_SPLIT)
        ewss.i[i] = WONT_SPLIT;
    if (ewss.i[i] != COULD_SPLIT)
      return;
    int local_max = 1;
    for (int j = 0; j < ee.n; ++j) {
      if (ewss_old.i[ee.e[j]] == WONT_SPLIT)
        continue;
      assert(ee.e[j] != i);
      double oq = eqs.s[ee.e[j]];
      if (oq == q) {
        if (ee.e[j] < i)
          local_max = 0;
      } else {
        if (q < oq)
          local_max = 0;
      }
    }
    if (local_max)
      ewss.i[i] = WILL_SPLIT;
  }
}

#if 0
struct is_determined {
  __device__ bool operator()(int i) const
  {
    return i == WILL_SPLIT || i == WONT_SPLIT;
  }
};
#endif

static struct ints compute_best_indset(struct ints ecss, struct graph ees,
    struct ss eqs)
{
  struct ints ewss = ints_new(ecss.n);
  struct ints ewss_old = ints_new(ecss.n);
  CUDALAUNCH(compute_best_indset_0, ecss.n, (ewss, ecss));
  int done = 0;
  int iter;
  for (iter = 0; !done; ++iter) {
    ints_copy(ewss_old, ewss, ewss.n);
    CUDALAUNCH(compute_best_indset_1, ecss.n, (ewss_old, ewss, ecss, ees, eqs));
#if 0
    thrust::device_ptr<int> p(ewss.i);
    done = thrust::all_of(p, p + ewss.n, is_determined());
#else
    int* hi = (int*) malloc(sizeof(int) * ewss.n);
    CUDACALL(cudaMemcpy(hi, ewss.i, sizeof(int) * ewss.n, cudaMemcpyDeviceToHost));
    done = 1;
    for (int i = 0; i < ewss.n; ++i)
      if (hi[i] == COULD_SPLIT)
        done = 0;
    free(hi);
#endif
    fprintf(stderr, "indset iter %d\n", iter);
    if (iter == 20)
      abort();
  }
  ints_free(ewss_old);
  return ewss;
}

static __global__ void mark_split_faces_0(struct ints fwss,
    struct ints ewss, struct rgraph fes)
{
  int i = CUDAINDEX;
  if (i < fes.nverts) {
    struct adj fe = adj_new_rgraph(fes);
    fwss.i[i] = 0;
    rgraph_get(fes, i, fe.e);
    for (int j = 0; j < fe.n; ++j)
      if (ewss.i[fe.e[j]])
        fwss.i[i] = 1;
  }
}

static struct ints mark_split_faces(struct ints ewss, struct rgraph fes)
{
  struct ints fwss = ints_new(fes.nverts);
  CUDALAUNCH(mark_split_faces_0, fes.nverts, (fwss, ewss, fes));
  return fwss;
}

static __global__ void split_edges_0(struct xs xs2, struct ints eos,
    struct xs xs, struct ints ewss, struct rgraph evs)
{
  int i = CUDAINDEX;
  if (i < ewss.n && ewss.i[i]) {
    int ev[2];
    rgraph_get(evs, i, ev);
    struct x ex[2];
    xs_get(xs, ev, 2, ex);
    struct x mid = x_avg(ex[0], ex[1]);
    xs2.x[xs.n + eos.i[i]] = mid;
  }
}

static struct xs split_edges(struct xs xs,
    struct ints ewss, struct rgraph evs)
{
  struct ints eos = ints_exscan(ewss);
  int nse;
  CUDACALL(cudaMemcpy(&nse, eos.i + ewss.n, sizeof(int), cudaMemcpyDeviceToHost));
  struct xs xs2 = xs_new(xs.n + nse);
  CUDACALL(cudaMemcpy(xs2.x, xs.x, sizeof(struct x) * xs.n, cudaMemcpyDeviceToDevice));
  CUDALAUNCH(split_edges_0, ewss.n, (xs2, eos, xs, ewss, evs));
  ints_free(eos);
  return xs2;
}

static __global__ void split_faces_0(struct rgraph fvs2, struct rgraph fvs,
    struct ints fwss)
{
  int i = CUDAINDEX;
  if (i < fwss.n && !fwss.i[i]) {
    int fv[3];
    rgraph_get(fvs, i, fv);
    rgraph_set(fvs2, i, fv);
  }
}

/* arguments, anyone ? */
static __global__ void split_faces_1(
    struct rgraph fvs2,
    struct ints fos, struct ints eos,
    struct rgraph fvs,
    struct ints fwss, struct ints ewss,
    struct rgraph evs, struct graph efs, int nv)
{
  int i = CUDAINDEX;
  if (i < ewss.n && ewss.i[i]) {
    struct adj ef = adj_new_graph(efs);
    int ev[2];
    rgraph_get(evs, i, ev);
    graph_get(efs, i, &ef);
    int sv = nv + eos.i[i];
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
}

static struct rgraph split_faces(struct rgraph fvs,
    struct ints fwss, struct ints ewss, struct rgraph evs,
    struct graph efs, int nv)
{
  struct ints fos = ints_exscan(fwss);
  struct ints eos = ints_exscan(ewss);
  int nsf;
  CUDACALL(cudaMemcpy(&nsf, fos.i + fwss.n, sizeof(int), cudaMemcpyDeviceToHost));
  struct rgraph fvs2 = rgraph_new(fvs.nverts + nsf, 3);
  CUDALAUNCH(split_faces_0, fwss.n, (fvs2, fvs, fwss));
  CUDALAUNCH(split_faces_1, ewss.n,
      (fvs2, fos, eos, fvs, fwss, ewss, evs, efs, nv));
  ints_free(eos);
  ints_free(fos);
  return fvs2;
}

void refine(struct rgraph fvs, struct xs xs, struct ss dss,
    struct rgraph* pfvs2, struct xs* pxs2)
{
  fprintf(stderr, "start refine\n");
  struct graph vfs = rgraph_invert(fvs);
  struct graph vvs = graph_rgraph_transit(vfs, fvs);
  graph_free(vfs);
  struct rgraph evs = graph_bridge(vvs);
  graph_free(vvs);
  struct graph ves = rgraph_invert(evs);
  struct rgraph fes = compute_fes(fvs, ves);
  graph_free(ves);
  struct graph efs = rgraph_invert(fes);
  struct ss as = compute_areas(xs, fvs);
  struct ints bfs = ss_gt(as, dss);
  ss_free(as);
  struct ints ecss = mark_fes(efs, bfs);
  ints_free(bfs);
  struct ss eqs = compute_split_quals(ecss, evs, efs, fvs, xs);
  struct graph ees = graph_rgraph_transit(efs, fes);
  struct ints ewss = compute_best_indset(ecss, ees, eqs);
  graph_free(ees);
  ss_free(eqs);
  ints_free(ecss);
  struct ints fwss = mark_split_faces(ewss, fes);
  rgraph_free(fes);
  struct xs xs2 = split_edges(xs, ewss, evs);
  *pxs2 = xs2;
  struct rgraph fvs2 = split_faces(fvs, fwss, ewss, evs, efs, xs.n);
  *pfvs2 = fvs2;
  ints_free(fwss);
  ints_free(ewss);
  rgraph_free(evs);
  graph_free(efs);
  fprintf(stderr, "stop refine\n");
}

