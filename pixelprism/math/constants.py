from __future__ import annotations

import math

from .dtype import DType
from .math_leaves import SymbolicConstant
from .tensor import tensor


def _symbolic_real(name: str, value: float) -> SymbolicConstant:
    return SymbolicConstant(name=name, data=tensor(data=float(value), dtype=DType.R, mutable=False))
# end def _symbolic_real


def _symbolic_complex(name: str, value: complex) -> SymbolicConstant:
    return SymbolicConstant(name=name, data=tensor(data=complex(value), dtype=DType.C, mutable=False))
# end def _symbolic_complex


# Fundamental constants
PI = _symbolic_real("pi", math.pi)
TAU = _symbolic_real("tau", math.tau)
E = _symbolic_real("e", math.e)
I = _symbolic_complex("i", 1j)

# Logarithmic / exponential constants
LN2 = _symbolic_real("ln2", math.log(2.0))
LN10 = _symbolic_real("ln10", math.log(10.0))
LOG2E = _symbolic_real("log2e", math.log2(math.e))
LOG10E = _symbolic_real("log10e", math.log10(math.e))

# Algebraic constants
SQRT2 = _symbolic_real("sqrt2", math.sqrt(2.0))
SQRT3 = _symbolic_real("sqrt3", math.sqrt(3.0))
SQRT5 = _symbolic_real("sqrt5", math.sqrt(5.0))
SILVER_RATIO = _symbolic_real("silver_ratio", 1.0 + math.sqrt(2.0))
PLASTIC_CONSTANT = _symbolic_real("plastic_constant", 1.324717957244746)

# Classical irrational/transcendental constants
PHI = _symbolic_real("phi", (1.0 + math.sqrt(5.0)) / 2.0)
EULER_GAMMA = _symbolic_real("euler_gamma", 0.5772156649015329)
CATALAN = _symbolic_real("catalan", 0.915965594177219)
APERY = _symbolic_real("apery", 1.2020569031595942)
KHINCHIN = _symbolic_real("khinchin", 2.6854520010653062)
GLAISHER_KINKELIN = _symbolic_real("glaisher_kinkelin", 1.2824271291006226)
OMEGA = _symbolic_real("omega", 0.5671432904097838)

# Number-theory / dynamical-system constants
TWIN_PRIME = _symbolic_real("twin_prime", 0.6601618158468696)
MEISSEL_MERTENS = _symbolic_real("meissel_mertens", 0.2614972128476428)
FEIGENBAUM_DELTA = _symbolic_real("feigenbaum_delta", 4.66920160910299)
FEIGENBAUM_ALPHA = _symbolic_real("feigenbaum_alpha", 2.5029078750958926)
LEVY = _symbolic_real("levy", 3.275822918721811)

# Aliases
GOLDEN_RATIO = PHI
GAMMA = EULER_GAMMA
ZETA3 = APERY


ALL = {
    "PI": PI,
    "TAU": TAU,
    "E": E,
    "I": I,
    "LN2": LN2,
    "LN10": LN10,
    "LOG2E": LOG2E,
    "LOG10E": LOG10E,
    "SQRT2": SQRT2,
    "SQRT3": SQRT3,
    "SQRT5": SQRT5,
    "SILVER_RATIO": SILVER_RATIO,
    "PLASTIC_CONSTANT": PLASTIC_CONSTANT,
    "PHI": PHI,
    "EULER_GAMMA": EULER_GAMMA,
    "CATALAN": CATALAN,
    "APERY": APERY,
    "KHINCHIN": KHINCHIN,
    "GLAISHER_KINKELIN": GLAISHER_KINKELIN,
    "OMEGA": OMEGA,
    "TWIN_PRIME": TWIN_PRIME,
    "MEISSEL_MERTENS": MEISSEL_MERTENS,
    "FEIGENBAUM_DELTA": FEIGENBAUM_DELTA,
    "FEIGENBAUM_ALPHA": FEIGENBAUM_ALPHA,
    "LEVY": LEVY,
    "GOLDEN_RATIO": GOLDEN_RATIO,
    "GAMMA": GAMMA,
    "ZETA3": ZETA3,
}


__all__ = list(ALL.keys()) + ["ALL"]
