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
Trigonometric operator implementations.
"""

# Imports
from .base import Operands, operator_registry
from .elementwise import ElementwiseOperator, UnaryElementwiseOperator, Sqrt
from ..tensor import Tensor
from ..math_expr import Variable, Constant, MathNode


__all__ = [
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
]


class Sin(UnaryElementwiseOperator):
    """
    Element-wise sine operator.

    Examples
    --------
    >>> import numpy as np
    >>> op = Sin()
    >>> op._eval([np.array([0.0, np.pi / 2])])
    array([0., 1.])
    """

    NAME = "sin"
    IS_VARIADIC = False
    IS_DIFF = True

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.sin(value.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        (x,) = operands
        return Cos.create_node(operands=(x,)) * x.diff(wrt)
    # end def _diff

# end class Sin


class Cos(UnaryElementwiseOperator):
    """
    Element-wise cosine operator.

    Examples
    --------
    >>> import numpy as np
    >>> Cos()._eval([np.array([0.0])])
    array([1.])
    """

    NAME = "cos"
    IS_VARIADIC = False
    IS_DIFF = True

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.cos(value.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        (x,) = operands
        return -Sin.create_node(operands=(x,)) * x.diff(wrt)
    # end def _diff

# end class Cos


class Tan(UnaryElementwiseOperator):
    """
    Element-wise tangent operator.

    Examples
    --------
    >>> import numpy as np
    >>> Tan()._eval([np.array([0.0])])
    array([0.])
    """

    NAME = "tan"
    IS_DIFF = True

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.tan(value.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        (x,) = operands
        sec_x = Sec.create_node(operands=(x,))
        return sec_x * sec_x * x.diff(wrt)
    # end def _diff

# end class Tan


class Asin(UnaryElementwiseOperator):
    """
    Element-wise arcsine operator.

    Examples
    --------
    >>> import numpy as np
    >>> Asin()._eval([np.array([0.0])])
    array([0.])
    """

    NAME = "asin"
    IS_DIFF = True

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.arcsin(value.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        (x,) = operands
        denom = Sqrt.create_node(operands=(Constant.new(1) - x * x,))
        return (Constant.new(1) / denom) * x.diff(wrt)
    # end def _diff

# end class Asin


class Acos(UnaryElementwiseOperator):
    """
    Element-wise arccosine operator.

    Examples
    --------
    >>> import numpy as np
    >>> Acos()._eval([np.array([1.0])])
    array([0.])
    """

    NAME = "acos"
    IS_DIFF = True

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.arccos(value.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        (x,) = operands
        denom = Sqrt.create_node(operands=(Constant.new(1) - x * x,))
        return (-Constant.new(1) / denom) * x.diff(wrt)
    # end def _diff

# end class Acos


class Atan(UnaryElementwiseOperator):
    """
    Element-wise arctangent operator.

    Examples
    --------
    >>> import numpy as np
    >>> Atan()._eval([np.array([0.0])])
    array([0.])
    """

    NAME = "atan"
    IS_DIFF = True

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.arctan(value.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        (x,) = operands
        denom = Constant.new(1) + x * x
        return (Constant.new(1) / denom) * x.diff(wrt)
    # end def _diff

# end class Atan


class Atan2(ElementwiseOperator):
    """
    Element-wise two-argument arctangent operator.

    Examples
    --------
    >>> import numpy as np
    >>> Atan2()._eval([np.array([1.0]), np.array([1.0])])
    array([0.78539816])
    """

    NAME = "atan2"
    ARITY = 2
    IS_DIFF = True

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        y, x = operands
        return Tensor.arctan2(y.eval(), x.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        y, x = operands
        denom = x * x + y * y
        dy_term = (x / denom) * y.diff(wrt)
        dx_term = (-y / denom) * x.diff(wrt)
        return dy_term + dx_term
    # end def _diff

# end class Atan2


class Sec(UnaryElementwiseOperator):
    """
    Element-wise secant operator.

    Examples
    --------
    >>> import numpy as np
    >>> Sec()._eval([np.array([0.0])])
    array([1.])
    """

    NAME = "sec"
    IS_DIFF = True

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return 1.0 / Tensor.cos(value.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        (x,) = operands
        return Sec.create_node(operands=(x,)) * Tan.create_node(operands=(x,)) * x.diff(wrt)
    # end def _diff

# end class Sec


class Csc(UnaryElementwiseOperator):
    """
    Element-wise cosecant operator.

    Examples
    --------
    >>> import numpy as np
    >>> Csc()._eval([np.array([np.pi / 2])])
    array([1.])
    """

    NAME = "csc"
    IS_DIFF = True

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return 1.0 / Tensor.sin(value.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        (x,) = operands
        return -Csc.create_node(operands=(x,)) * Cot.create_node(operands=(x,)) * x.diff(wrt)
    # end def _diff

# end class Csc


class Cot(UnaryElementwiseOperator):
    """
    Element-wise cotangent operator.

    Examples
    --------
    >>> import numpy as np
    >>> Cot()._eval([np.array([np.pi / 4])])
    array([1.])
    """

    NAME = "cot"
    IS_DIFF = True

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return 1.0 / Tensor.tan(value.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        (x,) = operands
        csc_x = Csc.create_node(operands=(x,))
        return -csc_x * csc_x * x.diff(wrt)
    # end def _diff

# end class Cot


class Sinh(UnaryElementwiseOperator):
    """
    Element-wise hyperbolic sine operator.

    Examples
    --------
    >>> import numpy as np
    >>> Sinh()._eval([np.array([0.0])])
    array([0.])
    """

    NAME = "sinh"
    IS_DIFF = True

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.sinh(value.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        (x,) = operands
        return Cosh.create_node(operands=(x,)) * x.diff(wrt)
    # end def _diff

# end class Sinh


class Cosh(UnaryElementwiseOperator):
    """
    Element-wise hyperbolic cosine operator.

    Examples
    --------
    >>> import numpy as np
    >>> Cosh()._eval([np.array([0.0])])
    array([1.])
    """

    NAME = "cosh"
    IS_DIFF = True

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.cosh(value.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        (x,) = operands
        return Sinh.create_node(operands=(x,)) * x.diff(wrt)
    # end def _diff

# end class Cosh


class Tanh(UnaryElementwiseOperator):
    """
    Element-wise hyperbolic tangent operator.

    Examples
    --------
    >>> import numpy as np
    >>> Tanh()._eval([np.array([0.0])])
    array([0.])
    """

    NAME = "tanh"
    IS_DIFF = True

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.tanh(value.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        (x,) = operands
        cosh_x = Cosh.create_node(operands=(x,))
        return (Constant.new(1) / (cosh_x * cosh_x)) * x.diff(wrt)
    # end def _diff

# end class Tanh


class Asinh(UnaryElementwiseOperator):
    """
    Element-wise inverse hyperbolic sine operator.

    Examples
    --------
    >>> import numpy as np
    >>> Asinh()._eval([np.array([0.0])])
    array([0.])
    """

    NAME = "asinh"
    IS_DIFF = True

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.arcsinh(value.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        (x,) = operands
        denom = Sqrt.create_node(operands=(x * x + Constant.new(1),))
        return (Constant.new(1) / denom) * x.diff(wrt)
    # end def _diff

# end class Asinh


class Acosh(UnaryElementwiseOperator):
    """
    Element-wise inverse hyperbolic cosine operator.

    Examples
    --------
    >>> import numpy as np
    >>> Acosh()._eval([np.array([1.5])])
    array([0.96242365])
    """

    NAME = "acosh"
    IS_DIFF = True

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.arccosh(value.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        (x,) = operands
        denom = (
            Sqrt.create_node(operands=(x - Constant.new(1),))
            * Sqrt.create_node(operands=(x + Constant.new(1),))
        )
        return (Constant.new(1) / denom) * x.diff(wrt)
    # end def _diff

# end class Acosh


class Atanh(UnaryElementwiseOperator):
    """
    Element-wise inverse hyperbolic tangent operator.

    Examples
    --------
    >>> import numpy as np
    >>> Atanh()._eval([np.array([0.2])])
    array([0.20273255])
    """

    NAME = "atanh"
    IS_DIFF = True

    def _eval(self, operands: Operands, **kwargs) -> Tensor:
        (value,) = operands
        return Tensor.arctanh(value.eval())
    # end def _eval

    def _diff(self, wrt: Variable, operands: Operands) -> MathNode:
        (x,) = operands
        denom = Constant.new(1) - x * x
        return (Constant.new(1) / denom) * x.diff(wrt)
    # end def _diff

# end class Atanh


operator_registry.register(Sin)
operator_registry.register(Cos)
operator_registry.register(Tan)
operator_registry.register(Asin)
operator_registry.register(Acos)
operator_registry.register(Atan)
operator_registry.register(Atan2)
operator_registry.register(Sec)
operator_registry.register(Csc)
operator_registry.register(Cot)
operator_registry.register(Sinh)
operator_registry.register(Cosh)
operator_registry.register(Tanh)
operator_registry.register(Asinh)
operator_registry.register(Acosh)
operator_registry.register(Atanh)
