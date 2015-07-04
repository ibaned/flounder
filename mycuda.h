#ifndef MYCUDA_H
#define MYCUDA_H

#define BLOCK_SIZE 128

static inline int ceildiv(int a, int b)
{
  int c = a / b;
  if (a % b)
    ++c;
  return c;
}

#define CUDACALL(fname,n,args) \
  fname<<< ceildiv((n), BLOCK_SIZE), BLOCK_SIZE >>>(args)

#endif
