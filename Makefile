CC=nvcc

all: flounder devinfo

flounder: flounder.o refine.o adj_ops.o graph_ops.o graph.o rgraph.o ints.o space.o vtk.o size.o
devinfo: devinfo.o

%: %.o
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

%.o: %.cu
	$(CC) $(CFLAGS) -c -o $@ $^

%.dis: %.o
	objdump --source $^ > $@

clean:
	rm -f *.o
