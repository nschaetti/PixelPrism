import pixelprism.math as pm


def _print_expr_details(
    title: str,
    expr: pm.MathExpr,
    x: pm.Variable,
    y: pm.Variable,
    z: pm.Variable,
    w: pm.Variable,
) -> None:
    print("\n" + "=" * 90)
    print(f"CASE: {title}")
    print("-" * 90)

    # Raw / simplified expressions
    simplified = expr.simplify()
    print(f"expr          : {expr}")
    print(f"simplified    : {simplified}")

    # repr() for node/operator visibility
    print(f"expr repr     : {repr(expr)}")
    print(f"simplified repr: {repr(simplified)}")

    expr_op = getattr(expr, "op", None)
    simplified_op = getattr(simplified, "op", None)
    if expr_op is not None:
        print(f"expr op       : {expr_op} | {repr(expr_op)}")
    if simplified_op is not None:
        print(f"simplified op : {simplified_op} | {repr(simplified_op)}")

    # Derivatives
    dx = expr.diff(wrt=x)
    dy = expr.diff(wrt=y)
    dz = expr.diff(wrt=z)
    dw = expr.diff(wrt=w)

    dx_s = dx.simplify()
    dy_s = dy.simplify()
    dz_s = dz.simplify()
    dw_s = dw.simplify()

    print(f"d/dx          : {dx}")
    print(f"d/dx simplify : {dx_s}")
    print(f"d/dy          : {dy}")
    print(f"d/dy simplify : {dy_s}")
    print(f"d/dz          : {dz}")
    print(f"d/dz simplify : {dz_s}")
    print(f"d/dw          : {dw}")
    print(f"d/dw simplify : {dw_s}")

    # Numeric checks
    with pm.new_context():
        pm.set_value("x", 2.0)
        pm.set_value("y", 3.0)
        pm.set_value("z", 4.0)
        pm.set_value("w", 5.0)

        print(f"eval(expr)    : {expr.eval()}")
        print(f"eval(simplify): {simplified.eval()}")
        print(f"eval(d/dx)    : {dx_s.eval()}")
        print(f"eval(d/dy)    : {dy_s.eval()}")
        print(f"eval(d/dz)    : {dz_s.eval()}")
        print(f"eval(d/dw)    : {dw_s.eval()}")


def main() -> None:
    x = pm.var("x", dtype=pm.DType.R, shape=())
    y = pm.var("y", dtype=pm.DType.R, shape=())
    z = pm.var("z", dtype=pm.DType.R, shape=())
    w = pm.var("w", dtype=pm.DType.R, shape=())
    k = pm.constants

    c0 = pm.const("zero", dtype=pm.DType.R, data=0.0)
    c2 = pm.const("two", dtype=pm.DType.R, data=2.0)
    c3 = pm.const("three", dtype=pm.DType.R, data=3.0)

    # Add simplification rules + 1..3 variable coverage
    cases = [
        ("MERGE_CONSTANTS: 2 + 3", c2 + c3),
        ("ADD_ZERO: x + 0", x + c0),
        ("ADD_ZERO: 0 + x", c0 + x),
        ("ADD_ITSELF: x + x", x + x),
        ("ADD_NEG: x + (-y)", x + (-y)),
        ("ADD_AX_BX: (2 * x) + (3 * x)", (c2 * x) + (c3 * x)),
        ("2 vars: x + y", x + y),
        ("2 vars + zero: (x + y) + 0", (x + y) + c0),
        ("3 vars: (x + y) + z", (x + y) + z),
        ("3 vars: x + (y + z)", x + (y + z)),
        ("3 vars + neg: (x + y) + (-z)", (x + y) + (-z)),
        ("3 vars with sub: x - y + z", x - y + z),
        ("with symbolic constants: x + pi + e", x + k.PI + k.E),
        ("symbolic-only: phi + pi + e", k.PHI + k.PI + k.E),
        # ("3 vars with sub: x - (y + z)", x - (y + z)),
        # ("3 vars with sub: (x - y) + z", (x - y) + z),
        # ("4 vars with sub: x - (y - z) + w", x - (y - z) + w),
        # ("4 vars with nested sub: x - (y - (z - w))", x - (y - (z - w))),
    ]

    print("\nElementwise Add deep-check (print/simplify/diff/eval)")
    print("Values used for eval: x=2, y=3, z=4, w=5")

    for title, expr in cases:
        _print_expr_details(title, expr, x, y, z, w)

    print("\nDone.")


if __name__ == "__main__":
    main()
