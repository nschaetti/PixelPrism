import pixelprism.math as pm


def _print_expr_details(title: str, expr: pm.MathExpr, x: pm.MathExpr, y: pm.MathExpr, z: pm.MathExpr) -> None:
    print("\n" + "=" * 90)
    print(f"CASE: {title}")
    print("-" * 90)

    expr_str = str(expr)
    expr_repr = repr(expr)
    expr_op_info = None
    if hasattr(expr, "op"):
        expr_op_info = f"{expr.op} | {repr(expr.op)}"

    dx = expr.diff(wrt=x)
    dy = expr.diff(wrt=y)
    dz = expr.diff(wrt=z)

    dx_s = dx.simplify()
    dy_s = dy.simplify()
    dz_s = dz.simplify()

    with pm.new_context():
        pm.set_value("x", 2.0)
        pm.set_value("y", 3.0)
        pm.set_value("z", 4.0)
        expr_eval = expr.eval()

    simplified = expr.simplify()
    simplified_str = str(simplified)
    simplified_repr = repr(simplified)
    simplified_op_info = None
    if hasattr(simplified, "op"):
        simplified_op_info = f"{simplified.op} | {repr(simplified.op)}"

    print(f"expr          : {expr_str}")
    print(f"simplified    : {simplified_str}")

    print(f"expr repr     : {expr_repr}")
    print(f"simplified repr: {simplified_repr}")

    if expr_op_info is not None:
        print(f"expr op       : {expr_op_info}")
    if simplified_op_info is not None:
        print(f"simplified op : {simplified_op_info}")

    print(f"d/dx          : {dx}")
    print(f"d/dx simplify : {dx_s}")
    print(f"d/dy          : {dy}")
    print(f"d/dy simplify : {dy_s}")
    print(f"d/dz          : {dz}")
    print(f"d/dz simplify : {dz_s}")

    with pm.new_context():
        pm.set_value("x", 2.0)
        pm.set_value("y", 3.0)
        pm.set_value("z", 4.0)

        print(f"eval(expr)    : {expr_eval}")
        print(f"eval(simplify): {simplified.eval()}")
        print(f"eval(d/dx)    : {dx_s.eval()}")
        print(f"eval(d/dy)    : {dy_s.eval()}")
        print(f"eval(d/dz)    : {dz_s.eval()}")


def main() -> None:
    x = pm.var("x", dtype=pm.DType.R, shape=())
    y = pm.var("y", dtype=pm.DType.R, shape=())
    z = pm.var("z", dtype=pm.DType.R, shape=())
    k = pm.constants

    c0 = pm.const("zero", dtype=pm.DType.R, data=0.0)
    c1 = pm.const("one", dtype=pm.DType.R, data=1.0)
    cneg1 = pm.const("minus_one", dtype=pm.DType.R, data=-1.0)
    c2 = pm.const("two", dtype=pm.DType.R, data=2.0)
    c3 = pm.const("three", dtype=pm.DType.R, data=3.0)

    cases = [
        ("MERGE_CONSTANTS: 2 * 3", c2 * c3),
        ("MERGE_CONSTANTS nested left: (2 * 3) * x", (c2 * c3) * x),
        ("MERGE_CONSTANTS nested right: x * (2 * 3)", x * (c2 * c3)),
        ("MUL_ONE: x * 1", x * c1),
        ("MUL_ONE: 1 * x", c1 * x),
        ("MUL_ONE nested right: (x * y) * 1", (x * y) * c1),
        ("MUL_ONE nested left: 1 * (x * y)", c1 * (x * y)),
        ("MUL_ZERO: x * 0", x * c0),
        ("MUL_ZERO: 0 * x", c0 * x),
        ("MUL_ZERO nested right: (x * y) * 0", (x * y) * c0),
        ("MUL_ZERO nested left: 0 * (x * y)", c0 * (x * y)),
        ("MUL_ZERO via constants: (2 * 0) * x", (c2 * c0) * x),
        ("MUL_BY_CONSTANT: x * 2", x * c2),
        ("MUL_ITSELF: x * x", x * x),
        ("MUL_NEG: x * (-y)", x * (-y)),
        ("MUL_BY_NEG_ONE: x * (-1)", x * cneg1),
        ("MUL_BY_INV: x * (1 / y)", x * (c1 / y)),
        ("MUL_BY_INV_NEG: x * (-1 / y)", x * (cneg1 / y)),
        ("MUL_BY_NEG_ITSELF: x * (-x)", x * (-x)),
        ("2 vars: x * y", x * y),
        ("3 vars: (x * y) * z", (x * y) * z),
        ("3 vars: x * (y * z)", x * (y * z)),
        ("3 vars + neg: (x * y) * (-z)", (x * y) * (-z)),
        ("with symbolic constants: x * pi * e", x * k.PI * k.E),
        ("symbolic-only: phi * pi * e", k.PHI * k.PI * k.E),
    ]

    print("\nElementwise Mul deep-check (print/simplify/diff/eval)")
    print("Values used for eval: x=2, y=3, z=4")

    for title, expr in cases:
        _print_expr_details(title, expr, x, y, z)

    print("\nDone.")


if __name__ == "__main__":
    main()
