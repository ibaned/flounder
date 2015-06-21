CC=nvcc

flounder: flounder.o refine.o adj_ops.o graph_ops.o graph.o rgraph.o adj.o ints.o space.o vtk.o size.o
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

%.o: %.cu
	$(CC) $(CFLAGS) -c -o $@ $^

%.dis: %.o
	objdump --source $^ > $@

clean:
	rm -f *.o
