#include "bits.h"
#include <stdlib.h>

struct bits bits_new(int nbits)
{
  struct bits b;
  b.n = nbits;
  int nbytes = nbits / 8;
  if (nbits % 8)
    ++nbytes;
  b.bytes = malloc(nbytes);
  return b;
}

void bits_free(struct bits b)
{
  free(b.bytes);
}

void bits_print(struct bits b)
{
  printf("bits (%d)\n", b.n);
  for (int i = 0; i < n; ++i)
    printf("%d: %d\n", i, bits_get(b, i));
}
