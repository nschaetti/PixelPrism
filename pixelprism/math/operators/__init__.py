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
"""
Operator package exports.
"""

from .base import Operands, Operator, OperatorRegistry, operator_registry
from .elementwise import (
    Add,
    Div,
    ElementwiseOperator,
    Exp,
    Log,
    Log2,
    Log10,
    Mul,
    Neg,
    Pow,
    Sqrt,
    Sub,
    UnaryElementwiseOperator,
)
from .linear_algebra import (
    MatMul,
    Dot,
    Outer
)
from .trigo import (
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atan2,
    Atanh,
    Cos,
    Cosh,
    Cot,
    Csc,
    Sec,
    Sin,
    Sinh,
    Tan,
    Tanh,
)

__all__ = [
    "Operands",
    "Operator",
    "OperatorRegistry",
    "operator_registry",
    "ElementwiseOperator",
    "UnaryElementwiseOperator",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Exp",
    "Log",
    "Log2",
    "Log10",
    "Sqrt",
    "Neg",
    "Sin",
    "Cos",
    "Tan",
    "Asin",
    "Acos",
    "Atan",
    "Atan2",
    "Sec",
    "Csc",
    "Cot",
    "Sinh",
    "Cosh",
    "Tanh",
    "Asinh",
    "Acosh",
    "Atanh",
    "MatMul",
    "Dot",
    "Outer"
]
