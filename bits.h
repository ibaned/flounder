#ifndef BITS_H
#define BITS_H

struct bits {
  unsigned char* bytes;
};

struct bits bits_new(int nbits);
int bits_get(struct bits b, int i);
void bits_set(struct bits b, int i);
void bits_clear(struct bits b, int i);
void bits_free(struct bits b);

#endif
