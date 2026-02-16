# ####   #####  #   #  #####  #
# #   #    #     # #   #      #
# ####     #      #    #####  #
# #        #     # #   #      #
# #      #####  #   #  #####  #####

from __future__ import annotations

import numpy as np

import pixelprism.math as pm
import pixelprism.math.functional as F
import pixelprism.math.functional.graph as G


def main() -> None:
    # Graph primitive
    g = pm.digraph(n_nodes=4, edges=[(0, 1), (1, 2), (0, 3)], allow_cycles=False)
    print("is cyclic:", g.is_cyclic())
    print("topological sort:", g.topological_sort())

    # Graph operators on adjacency matrix
    adj_expr = pm.const("adj", data=g.adjacency(), dtype=pm.DType.Z)
    print("out degree:", G.out_degree(adj_expr).eval().value)
    print("laplacian:\n", G.laplacian(adj_expr, directed=True).eval().value)

    # Expression to graph
    x = pm.var("graph_x", dtype=pm.DType.R, shape=())
    y = pm.var("graph_y", dtype=pm.DType.R, shape=())
    expr = F.mul(F.add(x, y), y)
    view = pm.expression_to_graph(expr)
    print("expr graph nodes/edges:", view.graph.n_nodes, view.graph.n_edges)
    print("expr graph adjacency:\n", pm.expression_to_adjacency(expr).eval().value)
    print("dot preview:\n", pm.graph_to_dot(view.graph))

    # Build from adjacency
    undirected = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int32)
    g2 = pm.graph_from_adjacency(undirected, directed=False)
    print("g2 degree:", g2.degree())
# end def main


if __name__ == "__main__":
    main()
