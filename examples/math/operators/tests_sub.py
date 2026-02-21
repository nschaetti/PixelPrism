import pixelprism.math as pm


def _print_expr_details(title: str, expr: pm.MathExpr, x: pm.MathExpr, y: pm.MathExpr, z: pm.MathExpr) -> None:
    print("\n" + "=" * 90)
    print(f"CASE: {title}")
    print("-" * 90)

    simplified = expr.simplify()
    print(f"expr          : {expr}")
    print(f"simplified    : {simplified}")

    print(f"expr repr     : {repr(expr)}")
    print(f"simplified repr: {repr(simplified)}")

    if hasattr(expr, "op"):
        print(f"expr op       : {expr.op} | {repr(expr.op)}")
    if hasattr(simplified, "op"):
        print(f"simplified op : {simplified.op} | {repr(simplified.op)}")

    dx = expr.diff(wrt=x)
    dy = expr.diff(wrt=y)
    dz = expr.diff(wrt=z)

    dx_s = dx.simplify()
    dy_s = dy.simplify()
    dz_s = dz.simplify()

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

        print(f"eval(expr)    : {expr.eval()}")
        print(f"eval(simplify): {simplified.eval()}")
        print(f"eval(d/dx)    : {dx_s.eval()}")
        print(f"eval(d/dy)    : {dy_s.eval()}")
        print(f"eval(d/dz)    : {dz_s.eval()}")


def main() -> None:
    x = pm.var("x", dtype=pm.DType.R, shape=())
    y = pm.var("y", dtype=pm.DType.R, shape=())
    z = pm.var("z", dtype=pm.DType.R, shape=())

    c0 = pm.const("zero", dtype=pm.DType.R, data=0.0)
    c2 = pm.const("two", dtype=pm.DType.R, data=2.0)
    c3 = pm.const("three", dtype=pm.DType.R, data=3.0)
    c5 = pm.const("five", dtype=pm.DType.R, data=5.0)

    cases = [
        ("MERGE_CONSTANTS: 5 - 2", c5 - c2),
        ("SUB_ZERO: x - 0", x - c0),
        ("SUB_ZERO: 0 - x", c0 - x),
        ("SUB_ITSELF: x - x", x - x),
        ("SUB_NEG: x - (-y)", x - (-y)),
        ("2 vars: x - y", x - y),
        ("2 vars + zero: (x - y) - 0", (x - y) - c0),
        ("3 vars: (x - y) - z", (x - y) - z),
        ("3 vars: x - (y - z)", x - (y - z)),
        ("3 vars + neg: (x - y) - (-z)", (x - y) - (-z)),
    ]

    print("\nElementwise Sub deep-check (print/simplify/diff/eval)")
    print("Values used for eval: x=2, y=3, z=4")

    for title, expr in cases:
        _print_expr_details(title, expr, x, y, z)

    print("\nDone.")


if __name__ == "__main__":
    main()
