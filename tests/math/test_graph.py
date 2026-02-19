import numpy as np

import pixelprism.math as pm
from pixelprism.math.functional import add, mul


def test_directed_graph_cycle_and_toposort():
    dag = pm.digraph(n_nodes=4, edges=[(0, 1), (1, 2), (0, 3)], allow_cycles=False)
    assert dag.directed is True
    assert dag.is_cyclic() is False
    order = dag.topological_sort()
    assert set(order) == {0, 1, 2, 3}
    assert order.index(0) < order.index(1)
    assert order.index(1) < order.index(2)

    cyc = pm.digraph(n_nodes=3, edges=[(0, 1), (1, 2), (2, 0)], allow_cycles=True)
    assert cyc.is_cyclic() is True
# end test_directed_graph_cycle_and_toposort


def test_undirected_graph_from_adjacency_and_degrees():
    adj = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.int32,
    )
    g = pm.graph_from_adjacency(adj, directed=False, allow_cycles=False)
    assert g.directed is False
    np.testing.assert_array_equal(g.adjacency(), adj)
    np.testing.assert_array_equal(g.degree(), np.array([1, 2, 1]))
    assert g.is_cyclic() is False
# end test_undirected_graph_from_adjacency_and_degrees


def test_expression_to_graph_and_adjacency_expr():
    x = pm.var("gx", dtype=pm.DType.R, shape=())
    y = pm.var("gy", dtype=pm.DType.R, shape=())
    expr = mul(add(x, y), y)

    view = pm.expression_to_graph(expr)
    g = view.graph
    assert g.directed is True
    assert g.n_nodes >= 4
    assert g.n_edges >= 4
    assert 0 <= view.root_index < g.n_nodes

    adj_expr = pm.expression_to_adjacency(expr)
    adj = adj_expr.eval().value
    assert adj.shape == (g.n_nodes, g.n_nodes)
    assert int(np.sum(adj)) == g.n_edges
# end test_expression_to_graph_and_adjacency_expr
