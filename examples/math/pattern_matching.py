"""Expression pattern matching demo from simple to complex cases."""

from __future__ import annotations

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


def main() -> None:
    P = pm.P
    k = pm.constants

    x = pm.var("x", dtype=pm.DType.R, shape=())
    y = pm.var("y", dtype=pm.DType.R, shape=())
    z = pm.var("z", dtype=pm.DType.R, shape=())
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


if __name__ == "__main__":
    main()
