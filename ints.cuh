#ifndef INTS_H
#define INTS_H

struct ints {
  int n;
  int* i;
};

struct ints ints_new(int n);
void ints_free(struct ints is);

struct ints ints_exscan(struct ints is);
int ints_max(struct ints is);

void ints_zero(struct ints is);
void ints_copy(struct ints into, struct ints from, int n);
void ints_from_host(struct ints is, int const host_dat[]);

#endif
