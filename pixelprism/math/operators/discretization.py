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

from typing import Optional, Union

from ..tensor import Tensor
from ..math_node import MathNode
from .base import Operands, operator_registry
from .elementwise import UnaryElementwiseOperator, UnaryElementwiseParametricOperator


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
    IS_VARIADIC = False
    IS_DIFF = False

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return value.eval().sign()
    # end def _eval

# end class Sign


class Floor(UnaryElementwiseOperator):
    """Element-wise floor operator."""

    NAME = "floor"
    IS_VARIADIC = False
    IS_DIFF = False

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return value.eval().floor()
    # end def _eval

# end class Floor


class Ceil(UnaryElementwiseOperator):
    """Element-wise ceil operator."""

    NAME = "ceil"
    IS_DIFF = False

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return value.eval().ceil()
    # end def _eval

# end class Ceil


class Trunc(UnaryElementwiseOperator):
    """Element-wise truncation operator."""

    NAME = "trunc"
    IS_DIFF = False

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return value.eval().trunc()
    # end def _eval
# end class Trunc


class Rint(UnaryElementwiseOperator):
    """Element-wise rounding to nearest integer."""

    NAME = "rint"
    IS_DIFF = False

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return value.eval().rint()
    # end def _eval
# end class Rint


class Round(UnaryElementwiseParametricOperator):
    """Element-wise rounding with configurable decimals."""

    NAME = "round"
    IS_DIFF = False

    def __init__(
            self,
            *,
            decimals: Optional[Union[MathNode, int]] = None
    ):
        super().__init__(decimals=decimals)
        from ..utils import const, random_const_name
        from ..math_base import MathNode
        if decimals is None:
            decimals = const(random_const_name("round-decimals-"), 0)
        # end if
        self._decimals: MathNode = decimals if isinstance(decimals, MathNode) \
            else const(name=random_const_name("round-decimals-"), data=decimals)
    # end def __init__

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return value.eval().round(decimals=self._resolve_parameter(self._decimals))
    # end def _eval

    def __str__(self) -> str:
        decimals = self._resolve_parameter(self._decimals)
        return f"{self.NAME}(decimals={decimals})"
    # end def __str__

    def __repr__(self) -> str:
        decimals = self._resolve_parameter(self._decimals)
        return f"{self.__class__.__name__}(decimals={decimals})"
    # end def __repr__
# end class Round


class Clip(UnaryElementwiseParametricOperator):
    """Element-wise clipping operator."""

    NAME = "clip"
    IS_DIFF = False

    def __init__(
            self,
            *,
            min_value: Optional[Union[MathNode, int]] = None,
            max_value: Optional[Union[MathNode, int]] = None
    ):
        super().__init__(min_value=min_value, max_value=max_value)
        from ..utils import const, random_const_name
        from ..math_base import MathNode
        if min_value is None and max_value is None:
            raise ValueError("Clip requires at least one of min_value or max_value.")
        # end if
        self._min_value = min_value if isinstance(min_value, MathNode) or min_value is None \
            else const(random_const_name("clip-min-"), min_value)
        self._max_value = max_value if isinstance(max_value, MathNode) or max_value is None \
            else const(random_const_name("clip-max-"), max_value)
    # end def __init__

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        tensor = value.eval()
        return tensor.clip(
            min_value=self._resolve_parameter(self._min_value),
            max_value=self._resolve_parameter(self._max_value)
        )
    # end def _eval

    def __str__(self) -> str:
        return (
            f"{self.NAME}(min_value={self._format_bound(self._min_value)}, "
            f"max_value={self._format_bound(self._max_value)})"
        )
    # end def __str__

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(min_value={self._format_bound(self._min_value)}, "
            f"max_value={self._format_bound(self._max_value)})"
        )
    # end def __repr__

    def _format_bound(self, bound):
        if bound is None:
            return None
        return self._resolve_parameter(bound)
    # end def _format_bound
# end class Clip


operator_registry.register(Sign)
operator_registry.register(Floor)
operator_registry.register(Ceil)
operator_registry.register(Trunc)
operator_registry.register(Rint)
operator_registry.register(Round)
operator_registry.register(Clip)
