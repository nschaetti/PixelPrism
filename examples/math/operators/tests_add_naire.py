import pixelprism.math as pm
from pixelprism.math.functional.elementwise import add


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
    k = pm.constants

    c0 = pm.const("zero", dtype=pm.DType.R, data=0.0)
    c1 = pm.const("one", dtype=pm.DType.R, data=1.0)
    c2 = pm.const("two", dtype=pm.DType.R, data=2.0)
    c3 = pm.const("three", dtype=pm.DType.R, data=3.0)
    c4 = pm.const("four", dtype=pm.DType.R, data=4.0)
    c9 = pm.const("nine", dtype=pm.DType.R, data=9.0)
    cneg1 = pm.const("minus_one", dtype=pm.DType.R, data=-1.0)

    cases = [
        ("N-ARY single constant: add(2)", add(c2)),
        ("N-ARY single variable: add(x)", add(x)),
        ("N-ARY MERGE_CONSTANTS: add(2, 3, 4)", add(c2, c3, c4)),
        ("N-ARY ADD_REMOVE_ZEROS: add(x, 0, y, 0, z)", add(x, c0, y, c0, z)),
        ("N-ARY ADD_GROUP_ALIKE: add(x, x, x)", add(x, x, x)),
        ("N-ARY MERGE_CONSTANTS: add(x, -y, z, -1)", add(x, -y, z, cneg1)),
        ("N-ARY ADD_REMOVE_ZEROS: add(x, 0)", add(x, c0)),
        ("N-ARY ADD_GROUP_ALIKE (pure): add(x*2, 3*x, 1*y, y*9)", add(x * c2, c3 * x, c1 * y, y * c9)),
        ("N-ARY ADD_GROUP_ALIKE (mixed): add(2*x, 3*x, y)", add(c2 * x, c3 * x, y)),
        ("N-ARY 3 vars: add(x, y, z)", add(x, y, z)),
        ("N-ARY nested: add(add(x, y), z, 0)", add(add(x, y), z, c0)),
        ("N-ARY long chain: add(x, y, z, 1, 2, -1)", add(x, y, z, c1, c2, cneg1)),
        ("N-ARY with symbolic constants: add(x, pi, e)", add(x, k.PI, k.E)),
        ("N-ARY symbolic-only: add(phi, pi, e)", add(k.PHI, k.PI, k.E)),
    ]

    print("\nElementwise Add N-ary deep-check (print/simplify/diff/eval)")
    print("Values used for eval: x=2, y=3, z=4")

    for title, expr in cases:
        _print_expr_details(title, expr, x, y, z)

    print("\nDone.")


if __name__ == "__main__":
    main()
