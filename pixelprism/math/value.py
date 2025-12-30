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
"""Runtime tensor value container."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .shape import Shape

__all__ = ["Value"]


class Value:
    """Mutable runtime tensor value."""

    def __init__(
        self,
        data: Any,
        shape: Shape,
        dtype: Any | None = None,
        ops: Any | None = None,
    ):
        """Initialize a Value.

        Parameters
        ----------
        data : Any
            Backend tensor data or array-like object.
        shape : Shape
            Symbolic tensor shape describing ``data``.
        dtype : Any | None, optional
            Backend dtype metadata associated with ``data``. Can be ``None`` if
            unknown.
        ops : Any | None, optional
            Optional backend operations helper carrying tensor operations used
            by higher-level functions. May be ``None`` when no backend is
            registered.

        Raises
        ------
        TypeError
            If ``shape`` is not an instance of :class:`Shape`.
        """
        if not isinstance(shape, Shape):
            raise TypeError("shape must be an instance of Shape.")
        # end if
        self._data = data
        self._shape = shape
        self._dtype = dtype
        self._ops = ops
    # end def __init__

    @property
    def shape(self) -> Shape:
        """Return the associated Shape.

        Returns
        -------
        Shape
            Symbolic shape describing the value.
        """
        return self._shape
    # end def shape

    @property
    def dtype(self) -> Any:
        """Return the backend dtype metadata.

        Returns
        -------
        Any
            Backend dtype descriptor stored for this value (may be ``None``).
        """
        return self._dtype
    # end def dtype

    def get(self) -> Any:
        """Return the backend data.

        Returns
        -------
        Any
            Underlying backend tensor data.
        """
        return self._data
    # end def get

    def set(self, data: Any) -> None:
        """Update the backend data.

        Parameters
        ----------
        data : Any
            New backend data sharing the same shape contract.
        """
        self._data = data
    # end def set

    def copy(self) -> "Value":
        """Return a deep copy of the Value.

        Returns
        -------
        Value
            Independent Value containing deep-copied data and the same metadata.
        """
        data_copy = deepcopy(self._data)
        return Value(data_copy, self._shape, self._dtype, self._ops)
    # end def copy

# end class Value

