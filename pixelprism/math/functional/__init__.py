# ####   #####  #   #  #####  #
# #   #    #     # #   #      #
# ####     #      #    #####  #
# #        #     # #   #      #
# #      #####  #   #  #####  #####
#
# ####   ####   #####   ####  #   #
# #   #  #   #    #    #      ## ##
# ####   ####     #     ###   # # #
# #      #  #     #        #  #   #
# #      #   #  #####  ####   #   #
#
# Copyright (C) 2025 Pixel Prism
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#


from .elementwise import (
    add,
    sub,
    mul,
    div,
    pow,
    log,
    log10,
    log2,
    log1p,
    exp,
    exp2,
    expm1,
    sqrt,
    square,
    cbrt,
    reciprocal,
    deg2rad,
    rad2deg,
    absolute,
    abs,
    neg,
)
from .trigo import sin, cos, acos, acosh, atan, cot, atan2, csc, sec, tan, asin, cosh, sinh, tanh, asinh, atanh
from .linear_algebra import matmul, dot, outer, trace
from .reduction import sum, mean, std, summation
from .discretization import (
    sign,
    floor,
    ceil,
    trunc,
    rint,
    round,
    clip,
)


__all__ = [
    # Element wise
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log10",
    "log2",
    "log1p",
    "sqrt",
    "square",
    "cbrt",
    "reciprocal",
    "deg2rad",
    "rad2deg",
    "absolute",
    "abs",
    "neg",
    # Discretization
    "sign",
    "floor",
    "ceil",
    "trunc",
    "rint",
    "round",
    "clip",
    # Trigo
    "sin",
    "cos",
    "acos",
    "acosh",
    "atan",
    "cot",
    "atan2",
    "csc",
    "sec",
    "tan",
    "asin",
    "cosh",
    "sinh",
    "tanh",
    "asinh",
    "atanh",
    # Linear Algebra
    "matmul",
    "dot",
    "outer",
    "trace",
    # Reduction
    "sum",
    "mean",
    "std",
    "summation"
]
