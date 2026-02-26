import pixelprism.math as pm
from pixelprism.math.functional.elementwise import sub


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

        print(f"depth expr      : {expr.depth()}")
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
    c1 = pm.const("one", dtype=pm.DType.R, data=1.0)
    c2 = pm.const("two", dtype=pm.DType.R, data=2.0)
    c3 = pm.const("three", dtype=pm.DType.R, data=3.0)
    c4 = pm.const("four", dtype=pm.DType.R, data=4.0)
    c9 = pm.const("nine", dtype=pm.DType.R, data=9.0)
    cneg1 = pm.const("minus_one", dtype=pm.DType.R, data=-1.0)

    cases = [
        # ("N-ARY single constant: sub(2)", sub(c2)),
        # ("N-ARY single variable: sub(x)", sub(x)),
        # ("N-ARY MERGE_CONSTANTS: sub(9, 4, 2)", sub(c9, c4, c2)),
        # ("N-ARY SUB_REMOVE_ZEROS: sub(x, 0, y, 0, z)", sub(x, c0, y, c0, z)),
        # ("N-ARY SUB_FIRST_ZERO: sub(0, x)", sub(c0, x)),
        # ("N-ARY SUB_FIRST_ZERO: sub(0, x, y)", sub(c0, x, y)),
        # ("N-ARY SUB_REMOVE_ZEROS: sub(x, 0)", sub(x, c0)),
        # ("N-ARY MERGE_CONSTANTS: sub(x, -y, z, -1)", sub(x, -y, z, cneg1)),
        # ("N-ARY 3 vars: sub(x, y, z)", sub(x, y, z)),
        # ("N-ARY nested: sub(sub(x, y), z, 0)", sub(sub(x, y), z, c0)),
        # ("N-ARY long chain: sub(x, y, z, 1, 2, -1)", sub(x, y, z, c1, c2, cneg1)),
        ("N-ARY repeated vars: sub(x, x, x)", sub(x, x, x)),
        # ("N-ARY mixed constants only: sub(3, 2, 1)", sub(c3, c2, c1)),
        ("N-ARY mixed constants and vars: sub(x * 2, 3 * x)", sub(x * 2, 3 * x)),
    ]

    print("\nElementwise Sub N-ary deep-check (print/simplify/diff/eval)")
    print("Values used for eval: x=2, y=3, z=4")

    for title, expr in cases:
        _print_expr_details(title, expr, x, y, z)

    print("\nDone.")


if __name__ == "__main__":
    main()
