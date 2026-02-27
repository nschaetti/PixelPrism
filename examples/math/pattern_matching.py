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

    print("Pattern matching demo (simple -> complex)")

    print_match_case(
        "VariablePattern: match by variable name",
        x,
        pm.VariablePattern(var_name="x", name="vx", return_expr=True),
    )
    print_match_case(
        "VariablePattern: mismatch by variable name",
        x,
        pm.VariablePattern(var_name="z", name="vz", return_expr=True),
    )

    print_match_case(
        "ConstantPattern: match by constant name + value",
        c2,
        pm.ConstantPattern(
            const_name="c2", value=2.0, name="const_two", return_expr=True
        ),
    )
    print_match_case(
        "ConstantPattern: mismatch by value",
        c2,
        pm.ConstantPattern(value=5.0, name="const_five", return_expr=True),
    )

    print_match_case(
        "AnyPattern: capture any expression",
        expr_add,
        pm.AnyPattern(name="any_expr", return_expr=True),
    )

    print_match_case(
        "NodePattern: add(x, c2) with operand capture",
        expr_add,
        pm.NodePattern(
            op="add",
            commutative=True,
            operands=[
                pm.VariablePattern(var_name="x", name="left_var", return_expr=True),
                pm.ConstantPattern(value=2.0, name="right_const", return_expr=True),
            ],
            arity=2,
            name="add_node",
            return_expr=True,
        ),
    )

    print_match_case(
        "NodePattern non-commutative: sub(x, y)",
        expr_sub,
        pm.NodePattern(
            op="sub",
            commutative=False,
            operands=[
                pm.VariablePattern(var_name="x", name="lhs", return_expr=True),
                pm.VariablePattern(var_name="y", name="rhs", return_expr=True),
            ],
            name="sub_node",
            return_expr=True,
        ),
    )
    print_match_case(
        "NodePattern non-commutative mismatch: sub(y, x)",
        expr_sub,
        pm.NodePattern(
            op="sub",
            commutative=False,
            operands=[
                pm.VariablePattern(var_name="y"),
                pm.VariablePattern(var_name="x"),
            ],
        ),
    )

    print_match_case(
        "NodePattern commutative: add(y, x) over x + y",
        expr_add_comm,
        pm.NodePattern(
            op="add",
            commutative=True,
            operands=[
                pm.VariablePattern(var_name="y", name="first", return_expr=True),
                pm.VariablePattern(var_name="x", name="second", return_expr=True),
            ],
            name="comm_add",
            return_expr=True,
        ),
    )

    print_match_case(
        "Complex nested NodePattern: strict shape (expected mismatch)",
        expr_complex,
        pm.NodePattern(
            op="mul",
            commutative=True,
            operands=[
                pm.NodePattern(
                    op="add",
                    commutative=True,
                    operands=[
                        pm.VariablePattern(var_name="x", name="x_leaf", return_expr=True),
                        pm.ConstantPattern(value=2.0, name="two_leaf", return_expr=True),
                    ],
                    name="left_branch",
                    return_expr=True,
                ),
                pm.NodePattern(
                    op="sub",
                    operands=[
                        pm.VariablePattern(var_name="y", name="y_leaf", return_expr=True),
                        pm.ConstantPattern(value=3.0, name="three_leaf", return_expr=True),
                    ],
                    name="right_branch",
                    return_expr=True,
                ),
            ],
            name="root_mul",
            return_expr=True,
        ),
    )

    print_match_case(
        "Complex nested NodePattern: actual tree (2 + x) * (3 - y)",
        expr_complex,
        pm.NodePattern(
            op="mul",
            commutative=True,
            operands=[
                pm.NodePattern(
                    op="add",
                    commutative=True,
                    operands=[
                        pm.VariablePattern(var_name="x", name="x_leaf", return_expr=True),
                        pm.ConstantPattern(value=2.0, name="two_leaf", return_expr=True),
                    ],
                    name="left_branch",
                    return_expr=True,
                ),
                pm.NodePattern(
                    op="sub",
                    operands=[
                        pm.ConstantPattern(value=3.0, name="three_leaf", return_expr=True),
                        pm.VariablePattern(var_name="y", name="y_leaf", return_expr=True),
                    ],
                    name="right_branch",
                    return_expr=True,
                ),
            ],
            name="root_mul",
            return_expr=True,
        ),
    )

    print_match_case(
        "AnyPattern on complex expression",
        expr_complex + z,
        pm.AnyPattern(name="complex_any", return_expr=True),
    )

    print_build_and_match_case(
        "Power/log: a * x^n",
        lambda: F.mul(a, F.pow(x, n)),
        pm.NodePattern(
            op="mul",
            commutative=True,
            operands=[
                pm.ConstantPattern(name="a_const", return_expr=True),
                pm.NodePattern(
                    op="pow",
                    operands=[
                        pm.VariablePattern(var_name="x", name="x_base", return_expr=True),
                        pm.VariablePattern(var_name="n", name="n_exp", return_expr=True),
                    ],
                    name="pow_node",
                    return_expr=True,
                ),
            ],
            name="mul_axn",
            return_expr=True,
        ),
    )

    print_build_and_match_case(
        "Power/log: a * x^2",
        lambda: F.mul(a, F.pow(x, 2)),
        pm.NodePattern(
            op="mul",
            commutative=True,
            operands=[
                pm.ConstantPattern(name="a_const", return_expr=True),
                pm.NodePattern(
                    op="pow",
                    operands=[
                        pm.VariablePattern(var_name="x", name="x_base", return_expr=True),
                        pm.ConstantPattern(value=2.0, name="two_exp", return_expr=True),
                    ],
                    name="pow_node",
                    return_expr=True,
                ),
            ],
            name="mul_ax2",
            return_expr=True,
        ),
    )

    print_build_and_match_case(
        "Power/log: a * log(b)",
        lambda: F.mul(a, F.log(b)),
        pm.NodePattern(
            op="mul",
            commutative=True,
            operands=[
                pm.ConstantPattern(name="a_const", return_expr=True),
                pm.NodePattern(
                    op="log",
                    operands=[
                        pm.ConstantPattern(const_name="b", name="b_const", return_expr=True)
                    ],
                    name="log_b",
                    return_expr=True,
                ),
            ],
            name="mul_alogb",
            return_expr=True,
        ),
    )

    print_build_and_match_case(
        "Power/log: a * log(n)",
        lambda: F.mul(a, F.log(n)),
        pm.NodePattern(
            op="mul",
            commutative=True,
            operands=[
                pm.ConstantPattern(name="a_const", return_expr=True),
                pm.NodePattern(
                    op="log",
                    operands=[
                        pm.VariablePattern(var_name="n", name="n_var", return_expr=True)
                    ],
                    name="log_n",
                    return_expr=True,
                ),
            ],
            name="mul_alogn",
            return_expr=True,
        ),
    )

    print_build_and_match_case(
        "Power/log: a * (log(n) / log(m))",
        lambda: F.mul(a, F.div(F.log(n), F.log(m))),
        pm.NodePattern(
            op="mul",
            commutative=True,
            operands=[
                pm.ConstantPattern(name="a_const", return_expr=True),
                pm.NodePattern(
                    op="div",
                    operands=[
                        pm.NodePattern(
                            op="log",
                            operands=[
                                pm.VariablePattern(
                                    var_name="n", name="n_var", return_expr=True
                                )
                            ],
                            name="log_n",
                            return_expr=True,
                        ),
                        pm.NodePattern(
                            op="log",
                            operands=[
                                pm.VariablePattern(
                                    var_name="m", name="m_var", return_expr=True
                                )
                            ],
                            name="log_m",
                            return_expr=True,
                        ),
                    ],
                    name="ratio",
                    return_expr=True,
                ),
            ],
            name="mul_ratio",
            return_expr=True,
        ),
    )


if __name__ == "__main__":
    main()
