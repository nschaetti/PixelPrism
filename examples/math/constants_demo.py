"""Show predefined symbolic constants available in pixelprism.math."""

from __future__ import annotations

import pixelprism.math as pm


def main() -> None:
    x = pm.var("x", dtype=pm.DType.R, shape=())

    print("Predefined symbolic constants (pm.constants.ALL):")
    print("-" * 72)
    for name in sorted(pm.constants.ALL):
        const = pm.constants.ALL[name]
        value = const.eval().value.item()
        print(f"{name:20s} -> {value!r} | foldable={const.is_foldable()} | dtype={const.dtype}")

    print("\nExample expressions:")
    print("-" * 72)

    expr1 = x + pm.constants.PI
    expr2 = pm.constants.PHI * x + pm.constants.E
    expr3 = pm.constants.I * pm.constants.I

    print(f"x + PI            : {expr1}")
    print(f"PHI * x + E       : {expr2}")
    print(f"I * I repr        : {repr(expr3)}")

    with pm.new_context():
        pm.set_value("x", 2.0)
        print(f"eval(x + PI)      : {expr1.eval()}")
        print(f"eval(PHI * x + E) : {expr2.eval()}")
        print(f"eval(I * I)       : {expr3.eval()}")


if __name__ == "__main__":
    main()
