#ifndef BITS_H
#define BITS_H

struct bits {
  int n;
  unsigned char* bytes;
};

struct bits bits_new(int nbits);

static inline int bits_get(struct bits b, int i)
{
  int byte = i / 8;
  int bit = i % 8;
  return (b.bytes[byte] & (1 << bit)) != 0;
}

static inline void bits_set(struct bits b, int i)
{
  int byte = i / 8;
  int bit = i % 8;
  b.bytes[byte] |= (1 << bit);
}

static inline void bits_clear(struct bits b, int i)
{
  int byte = i / 8;
  int bit = i % 8;
  b.bytes[byte] &= ~(1 << bit);
}

void bits_free(struct bits b);
void bits_print(struct bits b);

#endif
