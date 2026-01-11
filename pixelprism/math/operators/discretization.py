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
Discretization-related elementwise operators.
"""
from typing import Sequence, Any

from ..tensor import Tensor
from .base import Operands, Operator, operator_registry
from .elementwise import UnaryElementwiseOperator

__all__ = [
    "Sign",
    "Floor",
    "Ceil",
    "Trunc",
    "Rint",
    "Round",
    "Clip",
]


class Sign(UnaryElementwiseOperator):
    """Element-wise sign operator."""

    NAME = "sign"

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return value.eval().sign()
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Sign does not support backward.")
    # end def _backward
# end class Sign


class Floor(UnaryElementwiseOperator):
    """Element-wise floor operator."""

    NAME = "floor"

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return value.eval().floor()
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Floor does not support backward.")
    # end def _backward

# end class Floor


class Ceil(UnaryElementwiseOperator):
    """Element-wise ceil operator."""

    NAME = "ceil"

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return value.eval().ceil()
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Ceil does not support backward.")
    # end def _backward

# end class Ceil


class Trunc(UnaryElementwiseOperator):
    """Element-wise truncation operator."""

    NAME = "trunc"

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return value.eval().trunc()
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Trunc does not support backward.")
    # end def _backward
# end class Trunc


class Rint(UnaryElementwiseOperator):
    """Element-wise rounding to nearest integer."""

    NAME = "rint"

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return value.eval().rint()
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Rint does not support backward.")
    # end def _backward
# end class Rint


class Round(UnaryElementwiseOperator):
    """Element-wise rounding with configurable decimals."""

    NAME = "round"

    def __init__(self, *, decimals: int = 0, **kwargs: Any):
        super().__init__(**kwargs)
        self._decimals = decimals
    # end def __init__

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return value.eval().round(decimals=self._decimals)
    # end def _eval

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Round does not support backward.")
    # end def _backward
# end class Round


class Clip(UnaryElementwiseOperator):
    """Element-wise clipping operator."""

    NAME = "clip"

    def __init__(
            self,
            *,
            min_value: Any = None,
            max_value: Any = None,
            **kwargs: Any
    ):
        if min_value is None and max_value is None:
            raise ValueError("Clip requires at least one of min_value or max_value.")
        super().__init__(**kwargs)
        self._min_value = min_value
        self._max_value = max_value
    # end def __init__

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        tensor = value.eval()
        return tensor.clip(
            min_value=self._resolve_bound(self._min_value),
            max_value=self._resolve_bound(self._max_value)
        )
    # end def _eval

    @staticmethod
    def _resolve_bound(bound: Any):
        if bound is None:
            return None
        try:
            from ..math_expr import MathExpr  # local import to avoid cycles
        except Exception:  # pragma: no cover - defensive
            MathExpr = None  # type: ignore
        # end try
        if MathExpr is not None and isinstance(bound, MathExpr):
            return bound.eval()
        return bound
    # end def _resolve_bound

    def _backward(
            self,
            out_grad: "MathExpr",
            node: "MathExpr",
    ) -> Sequence["MathExpr"]:
        raise NotImplementedError("Clip does not support backward.")
    # end def _backward
# end class Clip


operator_registry.register(Sign)
operator_registry.register(Floor)
operator_registry.register(Ceil)
operator_registry.register(Trunc)
operator_registry.register(Rint)
operator_registry.register(Round)
operator_registry.register(Clip)

