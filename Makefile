CC=gcc
CFLAGS=-std=c99 -O2
LDFLAGS=-lm

flounder: flounder.o refine.o adj_ops.o graph_ops.o graph.o rgraph.o adj.o ints.o space.o vtk.o size.o

%.s: %.c
	$(CC) $(CFLAGS) -S -o $@ $^

%.dis: %.o
	objdump --source $^ > $@

clean:
	rm -f *.o
