#include "bits.h"
#include <stdlib.h>

struct bits bits_new(int nbits)
{
  struct bits b;
  int nbytes = nbits / 8;
  if (nbits % 8)
    ++nbytes;
  b.bytes = malloc(nbytes);
  return b;
}

int bits_get(struct bits b, int i)
{
  int byte = i / 8;
  int bit = i % 8;
  return (b.bytes[byte] & (1 << bit)) != 0;
}

void bits_set(struct bits b, int i)
{
  int byte = i / 8;
  int bit = i % 8;
  b.bytes[byte] |= (1 << bit);
}

void bits_clear(struct bits b, int i)
{
  int byte = i / 8;
  int bit = i % 8;
  b.bytes[byte] &= ~(1 << bit);
}

void bits_free(struct bits b)
{
  free(b.bytes);
}
