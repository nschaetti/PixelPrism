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
Structure operator implementations.
"""

# Imports
from abc import ABC
from typing import Optional, Sequence, Union, Any, List

import numpy as np

from ..dtype import DType
from ..shape import Shape
from ..tensor import Tensor, einsum
from ..math_expr import SliceExpr, MathNode
from .base import Operands, Operand, operator_registry, Operator, ParametricOperator

__all__ = [

]


class StructureOperator(Operator, ABC):
    """
    Linear algebra operator.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._parameters: dict[str, Any] = kwargs

        if not self.check_parameters(**kwargs):
            raise ValueError(f"Invalid parameters for operator {self.NAME}: {kwargs}")
        # end if
    # end def __init__

    def contains(
            self,
            expr: "MathNode",
            by_ref: bool = False,
            look_for: Optional[str] = None
    ) -> bool:
        """Does the operator contain the given expression (in parameters)?"""
        raise NotImplementedError("Parametric operators must implement contains(..).")
    # end def contains

    def check_parameters(self, **kwargs) -> bool:
        """Check that the operands have compatible shapes."""
        raise NotImplementedError("Parametric operators must implement check_parameters(..).")
    # end def check_shapes

# end class StructureOperator


class Getitem(StructureOperator):
    """Getitem operator."""

    NAME = "getitem"
    ARITY = 1

    def __init__(self, indices: List[Union[SliceExpr, int]]):
        super().__init__(indices=indices)
        self._indices = indices
    # end def __init__

    # region PUBLIC

    def check_parameters(self, indices: List[Union[SliceExpr, int]]) -> bool:
        def _check_slice(s: Union[SliceExpr, int]) -> bool:
            if isinstance(s, int):
                return True
            # end if
            return s.step is None or self._get_scalar(s.step) != 0
        # end def _check_slice
        return all([
            _check_slice(o)
            for o in indices
        ])
    # end for

    def check_operands(self, operands: Operands) -> bool:
        return len(operands) == 1
    # end def check_operands

    def infer_dtype(self, operands: Operands) -> DType:
        return operands[0].dtype
    # end def infer_dtype

    def infer_shape(self, operands: Operands) -> Shape:
        new_shape = list(operands[0].shape.dims)
        for n_i, (i, n) in enumerate(zip(self._indices, operands[0].shape.dims)):
            if isinstance(i, int):
                new_shape[n_i] = 0
            else:
                new_shape[n_i] = self._compute_new_dim(
                    start=self._get_scalar(i.start) if i.start is not None else 0,
                    stop=self._get_scalar(i.stop) if i.stop is not None else None,
                    step=self._get_scalar(i.step) if i.step is not None else 1,
                    n=n
                )
            # end if
        # end for
        if len(new_shape) > 1 and 0 in new_shape:
            new_shape.remove(0)
        # end if
        return Shape(new_shape)
    # end def infer_shape

    def check_shapes(self, operands: Operands) -> bool:
        for n_i, i in enumerate(self._indices):
            start = self._get_scalar(i.start) if isinstance(i, SliceExpr) else i
            if start < -operands[0].shape[n_i] or start >= operands[0].shape[n_i]:
                return False
            # end if
        # end for
        return True
    # end def check_shapes

    def contains(
            self,
            expr: MathNode,
            by_ref: bool = False,
            look_for: Optional[str] = None
    ) -> bool:
        return any([s.contains(expr, by_ref=by_ref, look_for=look_for) for s in self._indices])
    # end def contains

    # endregion PUBLIC

    # region PRIVATE

    def _get_scalar(self, e: Union[MathNode, int]) -> Union[int, float]:
        if isinstance(e, int):
            return e
        else:
            return e.eval().item()
        # end if
    # end def _get_scalar

    def _compute_new_dim(self, start: int, stop: Optional[int], step: int, n: int):
        """
        Compute the new dimension.
        """
        start = self._compute_start(start=start, n=n)
        stop = self._compute_stop(start=start, stop=stop, step=step, n=n)
        range_is = np.arange(start, stop, step)
        range_is = range_is[range_is >= 0]
        range_is = range_is[range_is < n]
        return range_is.size
    # end if

    def _compute_start(self, start: int, n: int) -> int:
        return start + n if start < 0 else start
    # end def _compute_start

    def _compute_stop(self, start: int, stop: Optional[int], step: int, n: int) -> int:
        """Compute the stop value for a slice."""
        if stop is None:
            return -1 if step < 0 else n
        else:
            return stop + n if stop < 0 else stop
        # end if
    # end def _compute_stop

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        indices: List[Union[SliceExpr, int]] = list()
        for i in self._indices:
            indices.append(i.to_slice() if isinstance(i, SliceExpr) else i)
        # end for
        return operands[0].eval()[*indices]
    # end def _eval

    def _backward(self, out_grad: "MathExpr", node: "MathExpr") -> Sequence["MathExpr"]:
        raise NotImplementedError("GetItem does not support backward.")
    # end def _backward

    # endregion PRIVATE

# end class GetItem


operator_registry.register(Getitem)
