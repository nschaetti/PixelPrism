import pixelprism.math as pm
import pixelprism.math.functional as F


def test_builder_capture_sets_name_and_return_expr():
    p_var = pm.P.v("x", as_="vx")
    assert p_var.name == "vx"
    assert p_var.return_expr is True

    p_any = pm.P.a()
    assert p_any.name is None
    assert p_any.return_expr is False

    p_node = pm.P.n("add", pm.P.v("x"), pm.P.c(2.0), as_="root")
    assert p_node.name == "root"
    assert p_node.return_expr is True
    assert p_node.operands is not None
    assert len(p_node.operands) == 2
# end def test_builder_capture_sets_name_and_return_expr


def test_builder_pattern_matches_like_manual_pattern():
    x = pm.var("x", dtype=pm.DType.R, shape=())
    c2 = pm.const("c2", data=2.0, dtype=pm.DType.R)
    expr = x + c2

    manual = pm.NodePattern(
        op="add",
        commutative=True,
        operands=[
            pm.VariablePattern(var_name="x", name="lhs", return_expr=True),
            pm.ConstantPattern(value=2.0, name="rhs", return_expr=True),
        ],
        name="root",
        return_expr=True,
    )
    compact = pm.P.n(
        "add",
        pm.P.v("x", as_="lhs"),
        pm.P.c(2.0, as_="rhs"),
        comm=True,
        as_="root",
    )

    manual_result = expr.match(manual)
    compact_result = expr.match(compact)

    assert manual_result.matched is True
    assert compact_result.matched is True
    assert set(manual_result.bindings.keys()) == set(compact_result.bindings.keys())
# end def test_builder_pattern_matches_like_manual_pattern


def test_classic_patterns_are_accessible_from_pm_patterns():
    x = pm.var("x", dtype=pm.DType.R, shape=())
    n = pm.var("n", dtype=pm.DType.R, shape=())
    a = pm.const("a", data=5.0, dtype=pm.DType.R)

    expr_axn = F.mul(a, F.pow(x, n))
    result_axn = expr_axn.match(pm.patterns.axn())
    assert result_axn.matched is True
    assert "a_const" in result_axn.bindings

    m = pm.var("m", dtype=pm.DType.R, shape=())
    expr_ratio = F.mul(a, F.div(F.log(n), F.log(m)))
    result_ratio = expr_ratio.match(pm.patterns.alog_ratio())
    assert result_ratio.matched is True
    assert "n_var" in result_ratio.bindings
    assert "m_var" in result_ratio.bindings
# end def test_classic_patterns_are_accessible_from_pm_patterns
