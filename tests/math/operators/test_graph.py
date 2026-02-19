import numpy as np
import pytest

import pixelprism.math as pm
import pixelprism.math.functional.graph as FG


def test_degree_operators_directed_and_undirected():
    directed_adj = np.array(
        [
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )
    undirected_adj = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.int32,
    )

    d_expr = pm.const("graph_deg_directed", data=directed_adj, dtype=pm.DType.Z)
    u_expr = pm.const("graph_deg_undirected", data=undirected_adj, dtype=pm.DType.Z)

    in_deg = FG.in_degree(d_expr).eval().value
    out_deg = FG.out_degree(d_expr).eval().value
    total_deg = FG.degree(d_expr, directed=True, mode="total").eval().value
    und_deg = FG.degree(u_expr, directed=False).eval().value

    np.testing.assert_array_equal(in_deg, np.array([0, 1, 2]))
    np.testing.assert_array_equal(out_deg, np.array([2, 1, 0]))
    np.testing.assert_array_equal(total_deg, np.array([2, 2, 2]))
    np.testing.assert_array_equal(und_deg, np.array([1, 2, 1]))
# end test_degree_operators_directed_and_undirected


def test_laplacian_operator_matches_expected():
    undirected_adj = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.int32,
    )
    expr = pm.const("graph_laplacian_adj", data=undirected_adj, dtype=pm.DType.Z)
    lap = FG.laplacian(expr, directed=False, normalized=False).eval().value
    expected = np.array(
        [
            [1.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(lap, expected, rtol=1e-6, atol=1e-6)
# end test_laplacian_operator_matches_expected


def test_cycle_detection_and_topological_sort():
    dag_adj = np.array(
        [
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    cyc_adj = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=np.int32,
    )
    dag_expr = pm.const("graph_dag_adj", data=dag_adj, dtype=pm.DType.Z)
    cyc_expr = pm.const("graph_cycle_adj", data=cyc_adj, dtype=pm.DType.Z)

    assert bool(FG.is_cyclic(dag_expr, directed=True).eval().value) is False
    assert bool(FG.is_cyclic(cyc_expr, directed=True).eval().value) is True

    order = FG.topological_sort(dag_expr).eval().value
    assert set(order.tolist()) == {0, 1, 2, 3}
    assert int(np.where(order == 0)[0][0]) < int(np.where(order == 1)[0][0])
    assert int(np.where(order == 1)[0][0]) < int(np.where(order == 2)[0][0])
    assert int(np.where(order == 2)[0][0]) < int(np.where(order == 3)[0][0])

    with pytest.raises(ValueError):
        FG.topological_sort(cyc_expr).eval()
    # end with
# end test_cycle_detection_and_topological_sort
