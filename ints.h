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

void ints_print(struct ints is);

void ints_zero(struct ints is);
void ints_from_dat(struct ints is, int const dat[]);

#endif
