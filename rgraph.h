#ifndef RGRAPH_H
#define RGRAPH_H

struct rgraph {
  int nverts;
  int degree;
  int* adjacent;
};

struct rgraph rgraph_new(int nverts, int degree);
void rgraph_free(struct rgraph g);

static inline void rgraph_set(struct rgraph g, int i, int const a[])
{
  int* p = g.adjacent + i * g.degree;
  int* e = p + g.degree;
  while (p < e)
    *p++ = *a++;
}

static inline void rgraph_get(struct rgraph g, int i, int a[])
{
  int* p = g.adjacent + i * g.degree;
  int* e = p + g.degree;
  while (p < e)
    *a++ = *p++;
}

int rgraph_max_adj(struct rgraph g);

#endif
