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
Symbolic math primitives used throughout Pixel Prism.

The :mod:`pixelprism.math.math_expr` module defines the canonical interfaces
implemented by every symbolic expression node in the system.  These nodes form
the immutable directed acyclic graphs that power algebraic manipulation,
automatic differentiation, and execution planning.  Each node advertises a
``DType`` and ``Shape`` describing the element type and tensor extent of its
value, and exposes helpers for structured traversal.

The documentation below mirrors the runtime implementation.  Every public
method follows the NumPy docstring convention to make the API consumable by the
interactive help system, by static tooling, and by our reference docs.
"""

# Imports
from __future__ import annotations
from typing import Optional
from .dtype import DType
from .shape import Shape


__all__ = [
    "MathBase",
]


class MathBase:
    """
    TODO: Add documentation.
    """

    __slots__ = (
        "_id",
        "_name",
        "_dtype",
        "_shape",
    )

    # Global counter
    _next_id = 0

    def __init__(
            self,
            name: Optional[str],
            *,
            dtype: DType,
            shape: Shape
    ) -> None:
        """
        Build a symbolic node (typically invoked by subclasses).

        Parameters
        ----------
        name : str or None
            Optional debugging label for the node.
        dtype : DType
            Element dtype advertised by the node.
        shape : Shape
            Symbolic tensor shape describing the node's extent.
        """
        self._id: int = MathBase._next_id
        self._name: str = name
        self._dtype: DType = dtype
        self._shape: Shape = shape
        MathBase._next_id += 1
    # end __init__

    @property
    def identifier(self) -> int:
        """
        Returns
        -------
        'int'
            Unique identifier for the node.
        """
        return self._id
    # end def identifier

    @property
    def dtype(self) -> DType:
        """
        Returns
        -------
        'DType'
            Element dtype for this expression.
        """
        return self._dtype
    # end def dtype

    @property
    def shape(self) -> Shape:
        """
        Returns
        -------
        'Shape'
            Symbolic shape describing the tensor extent.
        """
        return self._shape
    # end def shape

    @property
    def name(self) -> Optional[str]:
        """
        Returns
        -------
        str or None
            Optional user-provided display name.
        """
        return self._name
    # end def name

    def __hash__(self) -> int:
        """
        Returns
        -------
        int
            Identity hash suitable for storing nodes in sets/dicts.
        """
        return hash(self._id)
    # end __hash__

    @staticmethod
    def next_id() -> int:
        """
        Returns
        -------
        int
            Next identifier that will be assigned to a :class:`MathExpr`.
        """
        return MathBase._next_id
    # end def next_id

# end class MathBase
