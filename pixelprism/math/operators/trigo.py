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

from typing import Sequence

import numpy as np

from ..dtype import DType
from ..shape import Shape
from .base import Operands, operator_registry
from .elementwise import ElementwiseOperator, UnaryElementwiseOperator

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

    def _eval(self, values: np.ndarray) -> np.ndarray:
        (value,) = values
        return np.sin(value)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("Sin does not support backward.")
    # end def _backward

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

    def _eval(self, values: np.ndarray) -> np.ndarray:
        (value,) = values
        return np.cos(value)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("Cos does not support backward.")
    # end def _backward

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

    def _eval(self, values: np.ndarray) -> np.ndarray:
        (value,) = values
        return np.tan(value)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("Tan does not support backward.")
    # end def _backward

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

    def _eval(self, values: np.ndarray) -> np.ndarray:
        (value,) = values
        return np.arcsin(value)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("Asin does not support backward.")
    # end def _backward

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

    def _eval(self, values: np.ndarray) -> np.ndarray:
        (value,) = values
        return np.arccos(value)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("Acos does not support backward.")
    # end def _backward

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

    def _eval(self, values: np.ndarray) -> np.ndarray:
        (value,) = values
        return np.arctan(value)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("Atan does not support backward.")
    # end def _backward

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

    def _eval(self, values: np.ndarray) -> np.ndarray:
        y, x = values
        return np.arctan2(y, x)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("Atan2 does not support backward.")
    # end def _backward

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

    def _eval(self, values: np.ndarray) -> np.ndarray:
        (value,) = values
        return 1.0 / np.cos(value)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("Sec does not support backward.")
    # end def _backward

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

    def _eval(self, values: np.ndarray) -> np.ndarray:
        (value,) = values
        return 1.0 / np.sin(value)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("Csc does not support backward.")
    # end def _backward

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

    def _eval(self, values: np.ndarray) -> np.ndarray:
        (value,) = values
        return 1.0 / np.tan(value)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("Cot does not support backward.")
    # end def _backward

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

    def _eval(self, values: np.ndarray) -> np.ndarray:
        (value,) = values
        return np.sinh(value)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("Sinh does not support backward.")
    # end def _backward

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

    def _eval(self, values: np.ndarray) -> np.ndarray:
        (value,) = values
        return np.cosh(value)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("Cosh does not support backward.")
    # end def _backward

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

    def _eval(self, values: np.ndarray) -> np.ndarray:
        (value,) = values
        return np.tanh(value)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("Tanh does not support backward.")
    # end def _backward

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

    def _eval(self, values: np.ndarray) -> np.ndarray:
        (value,) = values
        return np.arcsinh(value)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("Asinh does not support backward.")
    # end def _backward

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

    def _eval(self, values: np.ndarray) -> np.ndarray:
        (value,) = values
        return np.arccosh(value)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("Acosh does not support backward.")
    # end def _backward

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

    def _eval(self, values: np.ndarray) -> np.ndarray:
        (value,) = values
        return np.arctanh(value)
    # end def _eval

    def _backward(self, out_grad, node):
        raise NotImplementedError("Atanh does not support backward.")
    # end def _backward

# end class Atanh


operator_registry.register(Sin())
operator_registry.register(Cos())
operator_registry.register(Tan())
operator_registry.register(Asin())
operator_registry.register(Acos())
operator_registry.register(Atan())
operator_registry.register(Atan2())
operator_registry.register(Sec())
operator_registry.register(Csc())
operator_registry.register(Cot())
operator_registry.register(Sinh())
operator_registry.register(Cosh())
operator_registry.register(Tanh())
operator_registry.register(Asinh())
operator_registry.register(Acosh())
operator_registry.register(Atanh())
