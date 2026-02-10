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

from typing import Any, Union, Optional
import numpy as np

from .math_node import MathNode
from .math_leaves import MathLeaf, const
from .dtype import ScalarLike, DType, TypeLike
from .tensor import TensorLike, Tensor
from .typing import MathExpr


__all__ = [
    "as_expr"
]


def as_expr(
        obj: Union[MathExpr, TensorLike],
        dtype: Optional[TypeLike] = None,
) -> MathExpr:
    """
    Convert Python and NumPy inputs to a :class:`~pixelprism.math.MathExpr`.

    This helper follows NumPy-style conversion rules: scalars and nested
    Python sequences are treated as array-like data and are converted via
    ``numpy.asarray`` inside :func:`pixelprism.math.tensor`. NumPy arrays are
    wrapped as :class:`~pixelprism.math.Tensor` instances; when ``dtype`` is
    provided, the array is cast to that dtype.

    Conversion rules (by input type)
    --------------------------------
    - ``MathExpr``: returned unchanged.
    - Scalar (Python or ``np.number``): converted to a scalar ``Tensor`` using
      ``dtype``.
    - ``numpy.ndarray``: wrapped as a ``Tensor``; if ``dtype`` is provided,
      it overrides the array's dtype.
    - Nested Python lists: converted to a ``Tensor`` using ``dtype``; when
      ``dtype`` is ``None``, defaults to ``R``.

    Parameters
    ----------
    obj : MathNode | DataType | numpy.ndarray
        Input object to convert.
    dtype : AnyDType | None, default None
        Target dtype for scalar, list, and array inputs. When ``None``,
        lists default to ``R`` and arrays keep their existing dtype.

    Returns
    -------
    MathNode
        A math expression node representing the input.

    Examples
    --------
    >>> from pixelprism.math import as_expr, DType
    >>> as_expr(3.5, dtype=DType.R).dtype
    <DType.R: 'R'>

    >>> import numpy as np
    >>> arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
    >>> as_expr(arr, dtype=DType.R).dtype
    <DType.R: 'R'>

    >>> as_expr([[1, 2], [3, 4]], dtype=None).dtype
    <DType.R: 'R'>
    """
    if isinstance(obj, MathExpr):
        return obj
    # end if
    if isinstance(obj, ScalarLike) or isinstance(obj, np.ndarray):
        return const(
            name=f"constant_{MathNode.next_id()}",
            data=obj,
            dtype=dtype
        )
    # end if
    if isinstance(obj, (list, tuple)):
        return const(
            name=f"constant_{MathNode.next_id()}",
            data=obj,
            dtype=dtype or DType.R
        )
    # end if
    if isinstance(obj, Tensor):
        return const(
            name=f"constant_{MathNode.next_id()}",
            data=obj.value,
        )
    # end if
    raise TypeError(f"Cannot convert {type(obj)} to MathExpr")
    # end if
# end def as_expr
