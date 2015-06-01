#ifndef GRAPH_OPS_H
#define GRAPH_OPS_H

#include "graph.h"
#include "rgraph.h"

struct graph rgraph_invert(struct rgraph rg);
struct graph graph_rgraph_transit(struct graph g, struct rgraph rg);
struct rgraph graph_bridge(struct graph g);

#endif
