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
Shared helpers for math operator tests.
"""

from __future__ import annotations

import numpy as np

import pixelprism.math as pm
from pixelprism.math.math_expr import MathExpr
from pixelprism.math.tensor import Tensor

SCALAR_KINDS = (
    "math_expr",
    "float",
    "int",
    "np_scalar",
    "np_array_scalar",
)

UNARY_SCALAR_KINDS = SCALAR_KINDS

TENSOR_SCALAR_KINDS = (
    "float",
    "int",
    "np_scalar",
    "np_array_scalar",
)


def make_tensor(value: float, name: str) -> MathExpr:
    """Create a scalar constant expression for tests."""
    return pm.const(name=name, data=value, dtype=pm.DType.FLOAT32)


def make_tensor_array(values, name: str) -> MathExpr:
    """Create an array constant expression for tests."""
    data = np.asarray(values, dtype=np.float32)
    return pm.const(name=name, data=data, dtype=pm.DType.FLOAT32)


def eval_expr_value(expr: MathExpr) -> np.ndarray:
    """Evaluate a MathExpr and return its NumPy array payload."""
    result = expr.eval()
    if isinstance(result, Tensor):
        return result.value
    return np.asarray(result)
# end def eval_expr_value


def assert_expr_allclose(expr: MathExpr, expected, **kwargs) -> None:
    """Assert that a MathExpr evaluates close to ``expected``."""
    np.testing.assert_allclose(eval_expr_value(expr), np.asarray(expected), **kwargs)
# end def assert_expr_allclose


def as_expected_array(value):
    if isinstance(value, MathExpr):
        return eval_expr_value(value)
    if isinstance(value, Tensor):
        return value.value
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float64)
    if isinstance(value, (int, float, np.number, bool, complex)):
        return np.asarray(value, dtype=np.float64)
    raise TypeError(f"Unsupported operand type: {type(value)}")


def operand_factory(kind: str, value: float):
    if kind == "math_expr":
        return make_tensor(value, f"t_{value}")
    if kind == "float":
        return float(value)
    if kind == "int":
        return int(value)
    if kind == "np_scalar":
        return np.float32(value)
    if kind == "np_array_scalar":
        return np.array(value, dtype=np.float32)
    if kind == "list":
        return [value]
    raise ValueError(f"Unknown operand kind: {kind}")
