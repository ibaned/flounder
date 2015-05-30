#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define MAX_UP 15

struct fv {
  int v[3];
};

struct fe {
  int e[3];
};

struct ev {
  int v[2];
};

struct x {
  double x[2];
};

struct fx {
  struct x x[3];
};

struct bits {
  unsigned char* bytes;
};

struct up {
  int e[MAX_UP];
  int n;
};

struct ups {
  int* a;
  int* n;
  int* o;
};

int const tab_fev[3][2] = {{0,1},{1,2},{2,0}};

struct bits bits_new(int n)
{
  int nbytes = n / 8;
  if (n % 8)
    ++nbytes;
  struct bits bs;
  bs.bytes = malloc(nbytes);
  return bs;
}

void bit_set(struct bits bs, int i)
{
  int byte = i / 8;
  int bit = i % 8;
  bs.bytes[byte] |= (1<<bit);
}

void bit_clear(struct bits bs, int i)
{
  int byte = i / 8;
  int bit = i % 8;
  bs.bytes[byte] &= ~(1<<bit);
}

int bit_high(struct bits bs, int i)
{
  int byte = i / 8;
  int bit = i % 8;
  return (bs.bytes[byte] & (1<<bit)) != 0;
}

void bits_free(struct bits bs)
{
  free(bs.bytes);
}

/* start off a 2-triangle mesh
 */
void init_mesh(struct fv** pfvs, struct x** px, int* pnf, int* pnv)
{
  struct fv fv_dat[2] = {
    {{0,1,2}},
    {{2,3,0}}
  };
  struct fv* fvs = malloc(sizeof(struct fv) * 2);
  memcpy(fvs, fv_dat, sizeof(struct fv) * 2);
  struct x x_dat[4] = {
    {{0,0}},
    {{1,0}},
    {{1,1}},
    {{0,1}}
  };
  struct x* x = malloc(sizeof(struct x) * 4);
  memcpy(x, x_dat, sizeof(struct x) * 4);
  *pfvs = fvs;
  *px = x;
  *pnf = 2;
  *pnv = 4;
}

struct fx get_fx(struct fv const* fvs, int i, struct x const* x)
{
  struct fx fx;
  for (int j = 0; j < 3; ++j)
    fx.x[j] = x[fvs[i].v[j]];
  return fx;
}

struct x sub_x(struct x a, struct x b)
{
  struct x c;
  c.x[0] = a.x[0] - b.x[0];
  c.x[1] = a.x[1] - b.x[1];
  return c;
}

double cross_x(struct x u, struct x v)
{
  return u.x[0] * v.x[1] - u.x[1] * v.x[0];
}

double fx_size(struct fx fx)
{
  struct x u = sub_x(fx.x[1], fx.x[0]);
  struct x v = sub_x(fx.x[2], fx.x[0]);
  return cross_x(u, v) / 2;
}

double f_size(struct fv const* fvs, int i, struct x const* x)
{
  return fx_size(get_fx(fvs, i, x));
}

void compute_sizes(struct fv const* fvs, int nf, struct x const* x,
    double** ps)
{
  double* s = malloc(sizeof(double) * nf);
  for (int i = 0; i < nf; ++i)
    s[i] = f_size(fvs, i, x);
  *ps = s;
}

void compute_big_fs(struct fv const* fvs, int nf, struct x const* x,
    double const* ds, struct bits* pbs)
{
  double* s;
  compute_sizes(fvs, nf, x, &s);
  struct bits bs = bits_new(nf);
  for (int i = 0; i < nf; ++i)
    if (s[i] > ds[i])
      bit_set(bs, i);
    else
      bit_clear(bs, i);
  free(s);
  *pbs = bs;
}

void compute_scan(int const* a, int n, int** pb)
{
  int* b = malloc(sizeof(int) * (n + 1));
  b[0] = 0;
  for (int i = 1; i <= n; ++i)
    b[i] = b[i - 1] + a[i];
  *pb = b;
}

void compute_ups(int const* down, int nup, int degree,
    int ndown, struct ups* pups)
{
  int* n = malloc(sizeof(int) * ndown);
  for (int i = 0; i < ndown; ++i)
    n[i] = 0;
  for (int i = 0; i < nup; ++i)
    for (int j = 0; j < degree; ++j)
      ++(n[down[i * degree + j]]);
  for (int i = 0; i < ndown; ++i)
    assert(n[i] <= MAX_UP);
  int* o;
  compute_scan(n, ndown, &o);
  int* a = malloc(sizeof(int) * nup * degree);
  for (int i = 1; i < ndown; ++i)
    n[i] = 0;
  for (int i = 0; i < nup; ++i)
    for (int j = 0; j < degree; ++j) {
      int di = down[i * degree + j];
      a[o[di] + n[di]] = i;
      ++(n[di]);
    }
  pups->a = a;
  pups->n = n;
  pups->o = o;
}

void compute_vfs(struct fv const* fvs, int nf, int nv,
    struct ups* pvfs)
{
  return compute_ups((int*) fvs, nf, 3, nv, pvfs);
}

int up_has(struct up const* pup, int i)
{
  for (int j = 0; j < pup->n; ++j)
    if (pup->e[j] == i)
      return 1;
  return 0;
}

void get_up(struct ups const ups, int i, struct up* pup)
{
  pup->n = ups.n[i];
  for (int j = 0; j < ups.n[i]; ++j)
    pup->e[j] = ups.a[ups.o[i] + j];
}

void compute_vv(struct fv const* fvs, int vi, struct up const* vf, struct up* pvv)
{
  pvv->n = 0;
  for (int i = 0; i < vf->n; ++i)
    for (int j = 0; j < 3; ++j) {
      int ovi = fvs[vf->e[i]].v[j];
      if (ovi == vi)
        continue;
      if (up_has(pvv, ovi))
        continue;
      pvv->e[pvv->n++] = ovi;
    }
}

void compute_vvs(struct fv const* fvs, int nf, int nv,
    struct ups const vfs, struct ups* pvvs)
{
  struct up vf;
  struct up vv;
  int* n = malloc(sizeof(int) * nv);
  for (int i = 0; i < nv; ++i) {
    get_up(vfs, i, &vf);
    compute_vv(fvs, i, &vf, &vv);
    n[i] = vv.n;
  }
  int* o;
  compute_scan(n, nv, &o);
  int nvv = o[nv];
  int* a = malloc(sizeof(int) * nvv);
  for (int i = 0; i < nv; ++i) {
    get_up(vfs, i, &vf);
    compute_vv(fvs, i, &vf, &vv);
    for (int j = 0; j < vv.n; ++j)
      a[o[i] + j] = vv.e[j];
  }
  pvvs->a = a;
  pvvs->n = n;
  pvvs->o = o;
}

void compute_evs(struct ups const vvs, int nv,
    int* pne, struct ev** pevs)
{
  int nvvs = vvs.o[nv];
  assert(nvvs % 2 == 0);
  int ne = nvvs / 2;
  int* nh = malloc(sizeof(int) * nv);
  for (int i = 0; i < nv; ++i) {
    nh[i] = 0;
    for (int j = 0; j < vvs.n[i]; ++j)
      if (i < vvs.a[vvs.o[i] + j])
        ++(nh[i]);
  }
  int* oh;
  compute_scan(nh, nv, &oh);
  free(nh);
  struct ev* evs = malloc(sizeof(struct ev) * ne);
  for (int i = 0; i < nv; ++i) {
    int k = 0;
    for (int j = 0; j < vvs.n[i]; ++j) {
      int oi = vvs.a[vvs.o[i] + j];
      if (i < oi) {
        struct ev ev = {i, oi};
        evs[oh[i] + (k++)] = ev;
      }
    }
  }
  free(oh);
  *pne = ne;
  *pevs = evs;
}

void compute_ves(struct ev const* evs, int ne, int nv,
    struct ups* pves)
{
  compute_ups((int*) evs, ne, 2, nv, pves);
}

void intersect(struct up* a, struct up const* b)
{
  int j;
  for (int i = 0; i < a->n; ++i)
    if (up_has(b, a->e[i]))
      a->e[j++] = a->e[i];
  a->n = j;
}

void compute_common_up(int const* down, int ndown,
    struct ups ups, struct up* pcup)
{
  struct up tup;
  get_up(ups, down[0], pcup);
  for (int i = 1; i < ndown; ++i) {
    get_up(ups, down[i], &tup);
    intersect(pcup, &tup);
  }
}

void compute_fes(struct fv const* fvs, int nf,
    struct ups ves, struct fe** pfes)
{
  struct fe* fes = malloc(sizeof(struct fe) * nf);
  for (int i = 0; i < nf; ++i) {
    struct fv fv = fvs[i];
    struct fe fe;
    for (int j = 0; j < 3; ++j) {
      int ev[2];
      for (int k = 0; k < 2; ++k)
        ev[k] = fv.v[tab_fev[j][k]];
      struct up cup;
      compute_common_up(fv.v, 3, ves, &cup);
      assert(cup.n == 1);
      fe.e[j] = cup.e[0];
    }
    fes[i] = fe;
  }
  *pfes = fes;
}
