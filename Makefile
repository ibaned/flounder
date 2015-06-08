CFLAGS=-fopenmp -std=c99 -g -O1 -fno-omit-frame-pointer
LDFLAGS=-lm

test: test.o refine.o adj_ops.o graph_ops.o graph.o rgraph.o adj.o ints.o space.o vtk.o size.o

%.s: %.c
	$(CC) $(CFLAGS) -S -o $@ $^

%.dis: %.o
	objdump --source $^ > $@

clean:
	rm -f *.o
