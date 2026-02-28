"""Expression pattern matching demo from simple to complex cases."""

from __future__ import annotations

from typing import cast

import pixelprism.math as pm
import pixelprism.math.functional as F


def print_match_case(title: str, expr, pattern) -> None:
    """Run one match case and print a readable result."""
    result = expr.match(pattern)
    print(f"\n=== {title} ===")
    print(f"expr    : {expr}")
    print(f"pattern : {pattern}")
    print(f"matched : {result.matched}")
    if result.bindings:
        print("bindings:")
        for name in sorted(result.bindings):
            print(f"  - {name}: {result.bindings[name]}")
    else:
        print("bindings: {}")


def print_build_and_match_case(title: str, expr_builder, pattern) -> None:
    """Build an expression, then run one match case with error reporting."""
    print(f"\n=== {title} ===")
    expr = expr_builder()

    result = expr.match(pattern)
    print(f"expr    : {expr}")
    print(f"pattern : {pattern}")
    print(f"matched : {result.matched}")
    if result.bindings:
        print("bindings:")
        for name in sorted(result.bindings):
            print(f"  - {name}: {result.bindings[name]}")
    else:
        print("bindings: {}")


def assert_match_case(
        title: str,
        expr,
        pattern,
        *,
        expected_matched: bool,
        expected_binding_keys: set[str] | None = None,
        expected_rest_len: int | None = None,
) -> None:
    """Run one match case with sanity assertions and readable output."""
    result = expr.match(pattern)
    print(f"\n=== {title} ===")
    print(f"expr    : {expr}")
    print(f"pattern : {pattern}")
    print(f"matched : {result.matched}")
    print(f"bindings: {result.bindings}")

    assert result.matched is expected_matched, (
        f"{title}: expected matched={expected_matched}, got {result.matched}"
    )

    if expected_binding_keys is not None:
        got_keys = set(result.bindings.keys())
        assert got_keys == expected_binding_keys, (
            f"{title}: expected binding keys={expected_binding_keys}, got={got_keys}"
        )

    if expected_rest_len is not None:
        rest = result.bindings.get("rest")
        assert isinstance(rest, list), f"{title}: expected 'rest' to be a list"
        assert len(rest) == expected_rest_len, (
            f"{title}: expected len(rest)={expected_rest_len}, got={len(rest)}"
        )


def assert_runtime_error_case(title: str, expr, pattern) -> None:
    """Ensure a pattern raises the expected RuntimeError."""
    print(f"\n=== {title} ===")
    print(f"expr    : {expr}")
    print(f"pattern : {pattern}")
    try:
        _ = expr.match(pattern)
    except RuntimeError as err:
        print(f"raised  : RuntimeError ({err})")
        return
    raise AssertionError(f"{title}: expected RuntimeError, but match succeeded")


def main() -> None:
    P = pm.P
    k = pm.constants

    x = pm.var("x", dtype=pm.DType.R, shape=())
    y = pm.var("y", dtype=pm.DType.R, shape=())
    z = pm.var("z", dtype=pm.DType.R, shape=())
    w = pm.var("w", dtype=pm.DType.R, shape=())
    n = pm.var("n", dtype=pm.DType.R, shape=())
    m = pm.var("m", dtype=pm.DType.R, shape=())

    a = pm.const("a", data=5.0, dtype=pm.DType.R)
    b = pm.const("b", data=10.0, dtype=pm.DType.R)

    c2 = pm.const("c2", data=2.0, dtype=pm.DType.R)
    c3 = pm.const("c3", data=3.0, dtype=pm.DType.R)

    expr_add = x + c2
    expr_sub = x - y
    expr_add_comm = x + y
    expr_complex = F.mul(x + c2, y - c3)
    expr_symbolic = x + k.PI
    expr2 = F.add(x, y)
    expr4 = F.add(x, y, z, w)

    print("Pattern matching demo (simple -> complex)")

    print_match_case(
        "VariablePattern: match by variable name",
        x,
        P.v("x", as_="vx"),
    )
    print_match_case(
        "VariablePattern: mismatch by variable name",
        x,
        P.v("z", as_="vz"),
    )

    print_match_case(
        "ConstantPattern: match by constant name + value",
        c2,
        P.c(2.0, const_name="c2", as_="const_two"),
    )
    print_match_case(
        "ConstantPattern: mismatch by value",
        c2,
        P.c(5.0, as_="const_five"),
    )

    print_match_case(
        "AnyPattern: capture any expression",
        expr_add,
        P.a(as_="any_expr"),
    )

    print_match_case(
        "NodePattern: add(x, c2) with operand capture",
        expr_add,
        P.n(
            "add",
            P.v("x", as_="left_var"),
            P.c(2.0, as_="right_const"),
            comm=True,
            arity=2,
            as_="add_node",
        ),
    )

    print_match_case(
        "NodePattern non-commutative: sub(x, y)",
        expr_sub,
        P.n(
            "sub",
            P.v("x", as_="lhs"),
            P.v("y", as_="rhs"),
            as_="sub_node",
        ),
    )
    print_match_case(
        "NodePattern non-commutative mismatch: sub(y, x)",
        expr_sub,
        P.n("sub", P.v("y"), P.v("x")),
    )

    print_match_case(
        "NodePattern commutative: add(y, x) over x + y",
        expr_add_comm,
        P.n("add", P.v("y", as_="first"), P.v("x", as_="second"), comm=True, as_="comm_add"),
    )

    print_match_case(
        "Complex nested NodePattern: strict shape (expected mismatch)",
        expr_complex,
        P.n(
            "mul",
            P.n(
                "add",
                P.v("x", as_="x_leaf"),
                P.c(2.0, as_="two_leaf"),
                comm=True,
                as_="left_branch",
            ),
            P.n(
                "sub",
                P.v("y", as_="y_leaf"),
                P.c(3.0, as_="three_leaf"),
                as_="right_branch",
            ),
            comm=True,
            as_="root_mul",
        ),
    )

    print_match_case(
        "Complex nested NodePattern: actual tree (2 + x) * (3 - y)",
        expr_complex,
        P.n(
            "mul",
            P.n(
                "add",
                P.v("x", as_="x_leaf"),
                P.c(2.0, as_="two_leaf"),
                comm=True,
                as_="left_branch",
            ),
            P.n(
                "sub",
                P.c(3.0, as_="three_leaf"),
                P.v("y", as_="y_leaf"),
                as_="right_branch",
            ),
            comm=True,
            as_="root_mul",
        ),
    )

    print_match_case(
        "AnyPattern on complex expression",
        expr_complex + z,
        P.a(as_="complex_any"),
    )

    print_match_case(
        "SymbolicConstant: x + pi",
        expr_symbolic,
        P.n(
            "add",
            P.v("x", as_="x_var"),
            P.c(value=float(k.PI.eval().value.item()), as_="pi_const"),
            comm=True,
            as_="add_x_pi",
        ),
    )

    print_build_and_match_case(
        "Power/log: a * x^n",
        lambda: F.mul(a, F.pow(x, n)),
        pm.patterns.axn(),
    )

    print_build_and_match_case(
        "Power/log: a * x^2",
        lambda: F.mul(a, F.pow(x, 2)),
        pm.patterns.ax2(),
    )

    print_build_and_match_case(
        "Power/log: a * log(b)",
        lambda: F.mul(a, F.log(b)),
        pm.patterns.alogb(),
    )

    print_build_and_match_case(
        "Power/log: a * log(n)",
        lambda: F.mul(a, F.log(n)),
        P.n(
            "mul",
            P.c(as_="a_const"),
            P.n("log", P.v("n", as_="n_var"), as_="log_n"),
            comm=True,
            as_="mul_alogn",
        ),
    )

    print_build_and_match_case(
        "Power/log: a * (log(n) / log(m))",
        lambda: F.mul(a, F.div(F.log(n), F.log(m))),
        pm.patterns.alog_ratio(),
    )

    print("\nEllipsis / EllipsisPattern sanity checks")

    # Non-commutative: suffix ellipsis and empty remainder
    assert_match_case(
        "Non-comm suffix Ellipsis literal",
        expr4,
        P.n("add", P.v("x", as_="x"), cast(pm.ExprPattern, ...)),
        expected_matched=True,
        expected_binding_keys={"x"},
    )
    assert_match_case(
        "Non-comm suffix EllipsisPattern",
        expr4,
        P.n("add", P.v("x", as_="x"), pm.p_ellipsis(as_="rest")),
        expected_matched=True,
        expected_binding_keys={"x", "rest"},
        expected_rest_len=3,
    )
    assert_match_case(
        "Non-comm suffix EllipsisPattern with empty remainder",
        expr2,
        P.n("add", P.v("x", as_="x"), P.v("y", as_="y"), pm.p_ellipsis(as_="rest")),
        expected_matched=True,
        expected_binding_keys={"x", "y", "rest"},
        expected_rest_len=0,
    )

    # Non-commutative: prefix ellipsis and empty remainder
    assert_match_case(
        "Non-comm prefix Ellipsis literal",
        expr4,
        P.n("add", cast(pm.ExprPattern, ...), P.v("w", as_="w")),
        expected_matched=True,
        expected_binding_keys={"w"},
    )
    assert_match_case(
        "Non-comm prefix EllipsisPattern",
        expr4,
        P.n("add", pm.p_ellipsis(as_="rest"), P.v("w", as_="w")),
        expected_matched=True,
        expected_binding_keys={"w", "rest"},
        expected_rest_len=3,
    )
    assert_match_case(
        "Non-comm prefix EllipsisPattern with empty remainder",
        expr2,
        P.n("add", pm.p_ellipsis(as_="rest"), P.v("x", as_="x"), P.v("y", as_="y")),
        expected_matched=True,
        expected_binding_keys={"x", "y", "rest"},
        expected_rest_len=0,
    )

    # Commutative: ellipsis can absorb non-matched operands
    assert_match_case(
        "Comm suffix EllipsisPattern",
        expr4,
        P.n("add", P.v("w", as_="w"), pm.p_ellipsis(as_="rest"), comm=True),
        expected_matched=True,
        expected_binding_keys={"w", "rest"},
        expected_rest_len=3,
    )
    assert_match_case(
        "Comm prefix EllipsisPattern",
        expr4,
        P.n("add", pm.p_ellipsis(as_="rest"), P.v("w", as_="w"), comm=True),
        expected_matched=True,
        expected_binding_keys={"w", "rest"},
        expected_rest_len=3,
    )

    # Mismatch without ellipsis
    assert_match_case(
        "Non-comm mismatch without ellipsis",
        expr4,
        P.n("add", P.v("w", as_="w"), P.v("x", as_="x"), P.v("y", as_="y"), P.v("z", as_="z")),
        expected_matched=False,
        expected_binding_keys=set(),
    )

    # Invalid shape: ellipsis in the middle is forbidden
    assert_runtime_error_case(
        "Invalid ellipsis position (middle)",
        expr4,
        P.n("add", P.v("x"), cast(pm.ExprPattern, ...), P.v("w")),
    )

    print("\nAll ellipsis sanity checks passed.")


if __name__ == "__main__":
    main()
