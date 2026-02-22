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
# Copyright (C) 2026 Pixel Prism
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
"""Generic algorithmic operator to execute registered Python callables."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from .. import as_expr
from ..dtype import DType, to_numpy
from ..math_node import MathNode
from ..shape import Shape
from ..tensor import Tensor
from ..typing import MathExpr, LeafKind
from .base import Operands, OperatorBase, ParametricOperator, operator_registry


AlgorithmCallable = Callable[..., Any]


__all__ = [
    "AlgorithmicOperator",
    "register_algorithm",
    "has_algorithm",
    "get_algorithm",
]


_ALGORITHMS: Dict[str, AlgorithmCallable] = {}


def register_algorithm(name: str, fn: AlgorithmCallable) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("Algorithm name must be a non-empty string")
    # end if
    if not callable(fn):
        raise TypeError("Algorithm function must be callable")
    # end if
    _ALGORITHMS[name] = fn
# end def register_algorithm


def has_algorithm(name: str) -> bool:
    return name in _ALGORITHMS
# end def has_algorithm


def get_algorithm(name: str) -> AlgorithmCallable:
    if name not in _ALGORITHMS:
        raise KeyError(f"Algorithm '{name}' is not registered")
    # end if
    return _ALGORITHMS[name]
# end def get_algorithm


def _resolve_param(value: Any) -> Any:
    if value is None:
        return None
    # end if
    if isinstance(value, (bool, int, float, str)):
        return value
    # end if
    if isinstance(value, MathNode):
        data = np.asarray(value.eval().value)
        return data.item() if data.shape == () else data
    # end if
    if isinstance(value, MathExpr):
        data = np.asarray(value.eval().value)
        return data.item() if data.shape == () else data
    # end if
    return value
# end def _resolve_param


class AlgorithmicOperator(OperatorBase, ParametricOperator):
    """Execute a user-registered Python algorithm at eval time."""

    NAME = "algorithm"
    ARITY = 0
    IS_VARIADIC = True

    def __init__(
            self,
            algorithm_name: str,
            out_shape: Shape,
            out_dtype: DType = DType.R,
            params: Optional[Dict[str, Any]] = None,
    ):
        params = params or {}
        super().__init__(algorithm_name=algorithm_name, out_shape=out_shape, out_dtype=out_dtype, params=params)
        self._algorithm_name = algorithm_name
        self._out_shape = Shape.create(out_shape)
        self._out_dtype = out_dtype
        self._params = dict(params)
    # end def __init__

    def contains(self, expr: MathExpr, by_ref: bool = False, look_for: LeafKind = LeafKind.ANY) -> bool:
        for value in self._params.values():
            if isinstance(value, MathExpr) and value.contains(expr, by_ref=by_ref, look_for=look_for):
                return True
            # end if
        # end for
        return False
    # end def contains

    @classmethod
    def check_parameters(
            cls,
            algorithm_name: str,
            out_shape: Shape,
            out_dtype: DType = DType.R,
            params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return has_algorithm(algorithm_name)
    # end def check_parameters

    def check_operands(self, operands: Operands) -> bool:
        return True
    # end def check_operands

    def check_shapes(self, operands: Operands) -> bool:
        return True
    # end def check_shapes

    def infer_shape(self, operands: Operands) -> Shape:
        return self._out_shape
    # end def infer_shape

    def infer_dtype(self, operands: Operands) -> DType:
        return self._out_dtype
    # end def infer_dtype

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        fn = get_algorithm(self._algorithm_name)
        operand_values = [np.asarray(op.eval().value) for op in operands]
        runtime_params = {key: _resolve_param(value) for key, value in self._params.items()}
        result = fn(*operand_values, **runtime_params)
        return Tensor(data=np.asarray(result, dtype=to_numpy(self._out_dtype)), dtype=self._out_dtype)
    # end def _eval

    def _backward(self, out_grad: MathNode, node: MathNode) -> Sequence[MathNode]:
        raise NotImplementedError("AlgorithmicOperator does not support backward propagation.")
    # end def _backward

    def __str__(self) -> str:
        return f"{self.NAME}({self._algorithm_name})"
    # end def __str__

    def __repr__(self) -> str:
        return self.__str__()
    # end def __repr__

# end class AlgorithmicOperator


operator_registry.register(AlgorithmicOperator)
